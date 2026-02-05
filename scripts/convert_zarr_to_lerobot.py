#!/usr/bin/env python3
"""
将 ViTaMin-B Zarr 格式数据转换为 LeRobot 格式（优化版：内存预加载 + 异步写入）
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import zarr
from zarr.storage import ZipStore
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 注册图像解码器
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()
from utils.pose_util import pose_to_mat, mat_to_pose


class ZarrToLeRobotConverter:
    """ViTaMin-B Zarr 格式到 LeRobot 格式的转换器（优化版）"""
    
    def __init__(self, 
                 zarr_path, 
                 output_repo_id, 
                 fps=30, 
                 state_dim=20, 
                 action_dim=20,
                 language_instruction=["perform bimanual manipulation task"]):
        """初始化转换器"""
        self.zarr_path = Path(zarr_path)
        self.output_repo_id = output_repo_id

        if not self.zarr_path.exists():
            raise ValueError(f"Zarr 文件不存在: {self.zarr_path}")
        
        # 加载 Zarr 数据
        print(f"加载 Zarr 数据: {self.zarr_path}")
        store = ZipStore(self.zarr_path, mode="r")
        self.zarr_root = zarr.open_group(store=store, mode="r")
        self.data = self.zarr_root["data"]
        
        print(f"Zarr 数据包含的键: {list(self.data.keys())}")

        # 分析结构
        self.robot_keys, self.camera_keys, self.tactile_keys, \
        self.num_robots, self.num_cameras, self.num_tactiles = self.analyze_zarr_structure()

        # 获取图像形状
        self.img_size = (224, 224, 3)
        if len(self.camera_keys) > 0:
            first_camera_rgb = self._process_image(self.data[self.camera_keys[0]][0])
            self.img_size = first_camera_rgb.shape
        
        self.fps = fps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.language_instruction = language_instruction
        
        # 🔥 预加载所有数据到内存
        self._preload_data()
        
        # 关闭 store，避免后续访问
        if hasattr(self.zarr_root, 'store'):
            self.zarr_root.store.close()
    
    def _preload_data(self):
        """🔥 预加载所有需要的数据到内存"""
        print("\n" + "="*70)
        print("预加载数据到内存（避免 I/O 阻塞）")
        print("="*70)
        
        self.data_cache = {}
        
        # 收集所有需要加载的键
        keys_to_load = set()
        
        # 机器人数据
        for i in range(self.num_robots):
            keys_to_load.add(f"robot{i}_eef_pos")
            keys_to_load.add(f"robot{i}_eef_rot_axis_angle")
            keys_to_load.add(f"robot{i}_gripper_width")
            keys_to_load.add(f"robot{i}_demo_start_pose")
        
        # 相机和触觉数据
        keys_to_load.update(self.camera_keys)
        keys_to_load.update(self.tactile_keys)
        
        # 添加其他可能的键
        camera_mappings = ["camera0_rgb", "camera1_rgb"]
        tactile_mappings = [
            "camera0_left_tactile", "camera0_right_tactile",
            "camera1_left_tactile", "camera1_right_tactile"
        ]
        keys_to_load.update(camera_mappings)
        keys_to_load.update(tactile_mappings)
        
        # 过滤出实际存在的键
        existing_keys = [k for k in keys_to_load if k in self.data.keys()]
        
        print(f"需要加载 {len(existing_keys)} 个数据数组...")
        
        # 批量加载到内存
        for key in tqdm(existing_keys, desc="加载中", ncols=70):
            try:
                self.data_cache[key] = self.data[key][:]  # 完整读取到内存
            except Exception as e:
                print(f"警告: 加载 {key} 失败: {e}")
                self.data_cache[key] = None
        
        # 加载 episode 信息
        self.episode_ends = self.zarr_root["meta"]["episode_ends"][:]
        
        print(f"✓ 已加载 {len(self.data_cache)} 个数组到内存")
        
        # 计算内存使用
        total_bytes = sum(
            arr.nbytes for arr in self.data_cache.values() 
            if arr is not None and hasattr(arr, 'nbytes')
        )
        print(f"✓ 总内存使用: {total_bytes / 1024**3:.2f} GB")
        print("="*70)
    
    def get_episode_info(self):
        """从内存缓存获取 episode 信息"""
        n_episodes = len(self.episode_ends)
        n_steps = self.episode_ends[-1] if len(self.episode_ends) > 0 else 0
        return n_episodes, n_steps, self.episode_ends
    
    def get_episode_slice(self, episode_idx, episode_ends):
        """返回给定 episode 索引的切片范围"""
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]
        return slice(start_idx, end_idx)
    
    def analyze_zarr_structure(self):
        """分析 Zarr 数据结构"""
        print("\n" + "="*70)
        print("分析 Zarr 数据结构")
        print("="*70)
        print(f"\n检测到:")

        keys = list(self.data.keys())
        
        robot_keys = [k for k in keys if k.startswith('robot') and 'eef_pos' in k]
        num_robots = len(robot_keys)
        print(f"  - 机器人数量: {num_robots}")

        camera_keys = [k for k in keys if k.startswith('camera') and ('rgb' in k)]
        num_cameras = len(camera_keys)
        print(f"  - 相机数量: {num_cameras}")

        tactile_keys = [k for k in keys if k.startswith('camera') and ('tactile' in k)]
        num_tactiles = len(tactile_keys)
        print(f"  - 触觉传感器数量: {num_tactiles}")
        
        episode_ends = self.zarr_root["meta"]["episode_ends"][:]
        n_episodes = len(episode_ends)
        n_steps = episode_ends[-1] if len(episode_ends) > 0 else 0
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        
        return robot_keys, camera_keys, tactile_keys, num_robots, num_cameras, num_tactiles
    
    def create_lerobot_dataset(self):
        """创建 LeRobot 数据集结构"""
        print(f"\n创建 LeRobot 数据集:")
        print(f"  - 数据集 ID: {self.output_repo_id}")
        print(f"  - 图像形状: {self.img_size}")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim}")
        print(f"  - 采集频率: {self.fps} Hz")
        
        features = {
            "observation.images.camera0": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.images.camera1": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.images.tactile_left_0": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.images.tactile_right_0": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.images.tactile_left_1": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.images.tactile_right_1": {
                "dtype": "image",
                "shape": self.img_size,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": ["observation.state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": ["actions"],
            },
        }
        
        dataset = LeRobotDataset.create(
            repo_id=self.output_repo_id,
            fps=self.fps,
            robot_type="bimanual",
            features=features,
            use_videos=False,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        return dataset
    
    def convert_all_episodes(self, episode_workers=None):
        """
        🔥 优化版：预加载数据 + 异步写入
        
        Args:
            episode_workers: 并行处理的 episode 数量
        """
        n_episodes, n_steps, episode_ends = self.get_episode_info()
        
        print(f"\n数据维度:")
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        print(f"  - 图像形状: {self.img_size}")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim}")
        
        # 创建数据集
        dataset = self.create_lerobot_dataset()
        
        # 设置并行度
        if episode_workers is None:
            episode_workers = min(8, os.cpu_count() or 4)
        episode_workers = max(1, min(episode_workers, n_episodes))
        
        print(f"\n{'='*70}")
        print(f"🚀 开始转换（优化模式）")
        print(f"{'='*70}")
        print(f"  - Episode 并行度: {episode_workers}")
        print(f"  - 数据源: 内存缓存（无 I/O 阻塞）")
        print(f"  - 写入模式: 异步写入（读写并行）")
        print(f"{'='*70}\n")
        
        # 🔥 异步写入队列和线程
        write_queue = Queue(maxsize=episode_workers * 2)
        write_complete = {'value': False}
        total_frames = {'value': 0}
        
        def writer_thread():
            """后台写入线程"""
            while True:
                item = write_queue.get()
                if item is None:  # 结束信号
                    write_complete['value'] = True
                    break
                
                ep_idx, frame_list = item
                for frame_data in frame_list:
                    dataset.add_frame(frame_data)
                dataset.save_episode()
                total_frames['value'] += len(frame_list)
                write_queue.task_done()
        
        # 启动写入线程
        writer = Thread(target=writer_thread, daemon=True)
        writer.start()
        
        # 🔥 多线程读取 + 构建帧数据
        with ThreadPoolExecutor(max_workers=episode_workers) as executor:
            # 提交所有任务（但保持顺序）
            future_to_idx = {
                executor.submit(self._build_episode_frames, ep_idx, episode_ends): ep_idx
                for ep_idx in range(n_episodes)
            }
            
            # 按完成顺序收集，但按 episode 顺序写入
            results_by_idx = [None] * n_episodes
            
            for future in tqdm(as_completed(future_to_idx), 
                             total=n_episodes, 
                             desc="构建帧数据",
                             ncols=70):
                ep_idx = future_to_idx[future]
                try:
                    frame_list = future.result()
                    results_by_idx[ep_idx] = frame_list
                except Exception as e:
                    raise RuntimeError(f"Episode {ep_idx} 处理失败: {e}") from e
        
        # 按顺序放入写入队列
        print("\n正在写入数据集...")
        for ep_idx in tqdm(range(n_episodes), desc="写入 episodes", ncols=70):
            frame_list = results_by_idx[ep_idx]
            write_queue.put((ep_idx, frame_list))
        
        # 等待写入完成
        write_queue.join()
        write_queue.put(None)
        writer.join()
        
        print(f"\n{'='*70}")
        print(f"✓ 转换完成!")
        print(f"{'='*70}")
        print(f"数据集保存位置: {dataset.root}")
        print(f"总 episodes: {n_episodes}")
        print(f"总帧数: {total_frames['value']}")
        print(f"平均每个 episode 帧数: {total_frames['value'] / n_episodes:.1f}")
        
        return dataset
    
    def _build_episode_frames(self, ep_idx, episode_ends):
        """
        🔥 从内存缓存构建单个 episode 的所有帧（无 I/O 阻塞）
        """
        episode_slice = self.get_episode_slice(ep_idx, episode_ends)
        start_idx, stop_idx = episode_slice.start, episode_slice.stop
        
        frame_list = []
        for step_idx in range(start_idx, stop_idx):
            frame_data = self._build_frame_data(step_idx, stop_idx)
            frame_list.append(frame_data)
        
        return frame_list
    
    def _build_frame_data(self, step_idx, stop_idx):
        """
        🔥 从内存缓存构建单帧数据（所有 self.data 改为 self.data_cache）
        """
        frame_data = {}
        
        # 语言指令
        if step_idx < len(self.language_instruction):
            frame_data["task"] = self.language_instruction[step_idx]
        else:
            frame_data["task"] = self.language_instruction[-1]

        # 图像 - 从内存读取
        camera_mappings = {
            "camera0_rgb": "observation.images.camera0",
            "camera1_rgb": "observation.images.camera1",
        }

        for cam_key, feature_key in camera_mappings.items():
            if cam_key in self.data_cache and self.data_cache[cam_key] is not None:
                img_data = self.data_cache[cam_key][step_idx]
                frame_data[feature_key] = self._process_image(img_data)
            else:
                frame_data[feature_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        tactile_mappings = {
            "camera0_left_tactile": "observation.images.tactile_left_0",
            "camera0_right_tactile": "observation.images.tactile_right_0",
            "camera1_left_tactile": "observation.images.tactile_left_1",
            "camera1_right_tactile": "observation.images.tactile_right_1",
        }

        for tac_key, feature_key in tactile_mappings.items():
            if tac_key in self.data_cache and self.data_cache[tac_key] is not None:
                img_data = self.data_cache[tac_key][step_idx]
                frame_data[feature_key] = self._process_image(img_data)
            else:
                frame_data[feature_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 状态向量 - 从内存读取
        state_features = []
        curr2world_mat_0 = None
        curr2world_mat_1 = None

        for i in range(self.num_robots):
            # 1. 相对初始位姿
            init_pose_key = f"robot{i}_demo_start_pose"
            if init_pose_key in self.data_cache and self.data_cache[init_pose_key] is not None:
                init2world_mat = pose_to_mat(self.data_cache[init_pose_key][0])
            else:
                init2world_mat = np.eye(4)
            
            pos_key = f"robot{i}_eef_pos"
            rot_key = f"robot{i}_eef_rot_axis_angle"
            
            if (pos_key in self.data_cache and rot_key in self.data_cache and 
                self.data_cache[pos_key] is not None and self.data_cache[rot_key] is not None):
                curr2world_mat = pose_to_mat(
                    np.concatenate([
                        self.data_cache[pos_key][step_idx],
                        self.data_cache[rot_key][step_idx],
                    ], axis=-1)
                )
            else:
                curr2world_mat = np.eye(4)
            
            if i == 0:
                curr2world_mat_0 = curr2world_mat
            else:
                curr2world_mat_1 = curr2world_mat

            curr2init_mat = np.linalg.inv(init2world_mat) @ curr2world_mat
            curr2init_pose = mat_to_pose(curr2init_mat)
            state_features.extend(curr2init_pose)

            # 2. 夹爪距离
            grip_key = f"robot{i}_gripper_width"
            if grip_key in self.data_cache and self.data_cache[grip_key] is not None:
                grip_data = self.data_cache[grip_key][step_idx]
                try:
                    if hasattr(grip_data, "__len__"):
                        state_features.append(float(grip_data[0]))
                    else:
                        state_features.append(float(grip_data))
                except Exception:
                    state_features.append(0.0)
            else:
                state_features.append(0.0)

        # 3. 两个末端执行器相对位姿
        if curr2world_mat_0 is not None and curr2world_mat_1 is not None:
            rel_0to1_pose = mat_to_pose(
                np.linalg.inv(curr2world_mat_1) @ curr2world_mat_0
            )
            state_features.extend(rel_0to1_pose)

        # 维度调整
        expected_state_dim = self.state_dim
        if len(state_features) < expected_state_dim:
            state_features.extend([0.0] * (expected_state_dim - len(state_features)))
        elif len(state_features) > expected_state_dim:
            state_features = state_features[:expected_state_dim]

        frame_data["observation.state"] = np.asarray(state_features, dtype=np.float32)

        # 动作（变化量）- 从内存读取
        if step_idx < stop_idx - 1:
            action_features = []
            for i in range(self.num_robots):
                pos_key = f"robot{i}_eef_pos"
                rot_key = f"robot{i}_eef_rot_axis_angle"
                
                if (pos_key in self.data_cache and rot_key in self.data_cache and
                    self.data_cache[pos_key] is not None and self.data_cache[rot_key] is not None):
                    next2world_mat = pose_to_mat(
                        np.concatenate([
                            self.data_cache[pos_key][step_idx + 1],
                            self.data_cache[rot_key][step_idx + 1],
                        ], axis=-1)
                    )
                    curr2world_mat = pose_to_mat(
                        np.concatenate([
                            self.data_cache[pos_key][step_idx],
                            self.data_cache[rot_key][step_idx],
                        ], axis=-1)
                    )

                    next2curr_mat = np.linalg.inv(curr2world_mat) @ next2world_mat
                    next2curr_pos = mat_to_pose(next2curr_mat)[:3]
                    rot_cols = next2curr_mat[:3, :2].reshape(-1)
                    action_feature_9d = np.concatenate([next2curr_pos, rot_cols], axis=0)
                    action_features.extend(action_feature_9d)
                else:
                    action_features.extend([0.0] * 9)

                # Δ gripper
                grip_key = f"robot{i}_gripper_width"
                if grip_key in self.data_cache and self.data_cache[grip_key] is not None:
                    next_grip = self.data_cache[grip_key][step_idx + 1]
                    curr_grip = self.data_cache[grip_key][step_idx]
                    try:
                        if hasattr(next_grip, "__len__") and hasattr(curr_grip, "__len__"):
                            delta_grip = float(next_grip[0] - curr_grip[0])
                        elif hasattr(next_grip, "__len__"):
                            delta_grip = float(next_grip[0] - curr_grip)
                        elif hasattr(curr_grip, "__len__"):
                            delta_grip = float(next_grip - curr_grip[0])
                        else:
                            delta_grip = float(next_grip - curr_grip)
                        action_features.append(delta_grip)
                    except Exception:
                        action_features.append(0.0)
                else:
                    action_features.append(0.0)

            frame_data["actions"] = np.asarray(action_features, dtype=np.float32)
        else:
            action_dim = self.action_dim
            frame_data["actions"] = np.zeros(action_dim, dtype=np.float32)

        return frame_data
    
    def _process_image(self, image_data, target_h=224, target_w=224):
        """处理图像数据：解码并调整大小"""
        if isinstance(image_data, bytes):
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        elif hasattr(image_data, "shape"):
            img = image_data
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        if img.dtype == np.uint8:
            pass
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        try:
            img = cv2.resize(img, (target_w, target_h))
        except Exception:
            img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        return img


def main(data_name="_0118"):
    parser = argparse.ArgumentParser(
        description='转换 ViTaMin-B Zarr 数据到 LeRobot 格式（优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--zarr_path',
        type=str,
        default=f'data/{data_name}.zarr.zip',
        help='Zarr 文件路径'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default=f'data/lerobot/chaoyi/{data_name}',
        help='LeRobot 数据集 ID'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='数据采集频率 (Hz)'
    )
    parser.add_argument(
        '--language_instruction',
        type=str,
        default=["perform bimanual manipulation task"],
        help='任务描述'
    )
    parser.add_argument(
        '--episode_workers',
        type=int,
        default=None,
        help='并行处理的 episode 数量'
    )
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"错误: 找不到 Zarr 文件: {zarr_path}")
        sys.exit(1)
    
    print("="*70)
    print("ViTaMin-B Zarr → LeRobot 转换（优化版）")
    print("="*70)
    print(f"Zarr 文件: {zarr_path.absolute()}")
    print(f"目标数据集: {args.repo_id}")
    print(f"采集频率: {args.fps} Hz")
    print("="*70)
    print()
    
    try:
        converter = ZarrToLeRobotConverter(
            zarr_path=args.zarr_path,
            output_repo_id=args.repo_id,
            fps=args.fps,
            state_dim=20,
            action_dim=20,
            language_instruction=args.language_instruction
        )
        
        dataset = converter.convert_all_episodes(
            episode_workers=args.episode_workers,
        )
        
    except Exception as e:
        print(f"\n错误: 转换失败")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main(data_name="example")