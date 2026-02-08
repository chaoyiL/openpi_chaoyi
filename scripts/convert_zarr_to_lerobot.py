#!/usr/bin/env python3
"""
将 ViTaMin-B Zarr 格式数据转换为 LeRobot 格式（内存节约版：进程池 + 批量读取）
"""

import argparse
import sys
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import zarr
from zarr.storage import ZipStore
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 注册图像解码器
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()
from utils.pose_util import pose_to_mat, mat_to_pose


# ============================================================================
# 进程池全局变量（每个进程独立的 store）
# ============================================================================
_PROCESS_ZARR_ROOT = None
_PROCESS_DATA = None


def _worker_init(zarr_path):
    """
    🔥 进程池初始化函数：每个进程独立打开自己的 ZipStore
    避免多进程共享同一个文件句柄导致的锁争用
    """
    global _PROCESS_ZARR_ROOT, _PROCESS_DATA
    
    # 每个进程打开自己的只读 store
    store = ZipStore(zarr_path, mode="r")
    _PROCESS_ZARR_ROOT = zarr.open_group(store=store, mode="r")
    _PROCESS_DATA = _PROCESS_ZARR_ROOT["data"]
    
    # 重新注册编解码器（每个进程需要独立注册）
    register_codecs()


def _build_episode_frames_worker(args):
    """
    🔥 进程池 worker：批量读取单个 episode 的数据并构建所有帧
    
    这个函数必须是模块级函数（不能是类方法），因为 multiprocessing 需要 pickle
    """
    (ep_idx, start_idx, stop_idx, num_robots, state_dim, action_dim, 
     language_instruction, img_size) = args
    
    global _PROCESS_DATA
    
    # ========================================================================
    # 🔥 关键优化：批量读取该 episode 的所有数据切片（一次 I/O）
    # ========================================================================
    episode_data = {}
    
    # 需要读取的所有键
    keys_to_load = []
    
    # 机器人数据
    for i in range(num_robots):
        keys_to_load.extend([
            f"robot{i}_eef_pos",
            f"robot{i}_eef_rot_axis_angle",
            f"robot{i}_gripper_width",
            f"robot{i}_demo_start_pose",
        ])
    
    # 相机和触觉数据
    camera_keys = ["camera0_rgb", "camera1_rgb"]
    tactile_keys = [
        "camera0_left_tactile", "camera0_right_tactile",
        "camera1_left_tactile", "camera1_right_tactile"
    ]
    keys_to_load.extend(camera_keys)
    keys_to_load.extend(tactile_keys)
    
    # 批量切片读取（关键：一次读取整个 episode 范围，而不是逐帧读取）
    for key in keys_to_load:
        if key in _PROCESS_DATA.keys():
            try:
                # 只读取当前 episode 的数据范围
                if key.endswith('_demo_start_pose'):
                    # demo_start_pose 只需要第一个值
                    episode_data[key] = _PROCESS_DATA[key][0:1]
                else:
                    episode_data[key] = _PROCESS_DATA[key][start_idx:stop_idx]
            except Exception as e:
                print(f"警告: 读取 {key} 失败: {e}")
                episode_data[key] = None
        else:
            episode_data[key] = None
    
    # ========================================================================
    # 基于批量读取的数据构建所有帧（纯内存操作，无 I/O）
    # ========================================================================
    frame_list = []
    episode_length = stop_idx - start_idx
    
    for local_idx in range(episode_length):
        global_idx = start_idx + local_idx
        
        frame_data = _build_single_frame(
            episode_data=episode_data,
            local_idx=local_idx,
            global_idx=global_idx,
            episode_length=episode_length,
            num_robots=num_robots,
            state_dim=state_dim,
            action_dim=action_dim,
            language_instruction=language_instruction,
            img_size=img_size
        )
        
        frame_list.append(frame_data)
    
    return ep_idx, frame_list


def _build_single_frame(episode_data, local_idx, global_idx, episode_length,
                        num_robots, state_dim, action_dim, language_instruction, img_size):
    """
    从批量读取的 episode 数据中构建单帧（纯内存操作）
    
    Args:
        episode_data: 该 episode 的所有数据（已批量读取）
        local_idx: 在 episode 内的索引（0-based）
        global_idx: 全局索引（用于 language_instruction）
        episode_length: episode 总帧数
        其他参数: 配置信息
    """
    frame_data = {}
    
    # 语言指令
    if global_idx < len(language_instruction):
        frame_data["task"] = language_instruction[global_idx]
    else:
        frame_data["task"] = language_instruction[-1]
    
    # ========================================================================
    # 图像数据
    # ========================================================================
    camera_mappings = {
        "camera0_rgb": "observation.images.camera0",
        "camera1_rgb": "observation.images.camera1",
    }
    
    for cam_key, feature_key in camera_mappings.items():
        if cam_key in episode_data and episode_data[cam_key] is not None:
            img_data = episode_data[cam_key][local_idx]
            frame_data[feature_key] = _process_image(img_data, img_size)
        else:
            frame_data[feature_key] = np.zeros(img_size, dtype=np.uint8)
    
    tactile_mappings = {
        "camera0_left_tactile": "observation.images.tactile_left_0",
        "camera0_right_tactile": "observation.images.tactile_right_0",
        "camera1_left_tactile": "observation.images.tactile_left_1",
        "camera1_right_tactile": "observation.images.tactile_right_1",
    }
    
    for tac_key, feature_key in tactile_mappings.items():
        if tac_key in episode_data and episode_data[tac_key] is not None:
            img_data = episode_data[tac_key][local_idx]
            frame_data[feature_key] = _process_image(img_data, img_size)
        else:
            frame_data[feature_key] = np.zeros(img_size, dtype=np.uint8)
    
    # ========================================================================
    # 状态向量
    # ========================================================================
    state_features = []
    curr2world_mat_0 = None
    curr2world_mat_1 = None
    
    for i in range(num_robots):
        # 1. 相对初始位姿
        init_pose_key = f"robot{i}_demo_start_pose"
        if init_pose_key in episode_data and episode_data[init_pose_key] is not None:
            init2world_mat = pose_to_mat(episode_data[init_pose_key][0])
        else:
            init2world_mat = np.eye(4)
        
        pos_key = f"robot{i}_eef_pos"
        rot_key = f"robot{i}_eef_rot_axis_angle"
        
        if (pos_key in episode_data and rot_key in episode_data and
            episode_data[pos_key] is not None and episode_data[rot_key] is not None):
            curr2world_mat = pose_to_mat(
                np.concatenate([
                    episode_data[pos_key][local_idx],
                    episode_data[rot_key][local_idx],
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
        if grip_key in episode_data and episode_data[grip_key] is not None:
            grip_data = episode_data[grip_key][local_idx]
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
    if len(state_features) < state_dim:
        state_features.extend([0.0] * (state_dim - len(state_features)))
    elif len(state_features) > state_dim:
        state_features = state_features[:state_dim]
    
    frame_data["observation.state"] = np.asarray(state_features, dtype=np.float32)
    
    # ========================================================================
    # 动作向量
    # ========================================================================
    if local_idx < episode_length - 1:
        action_features = []
        for i in range(num_robots):
            pos_key = f"robot{i}_eef_pos"
            rot_key = f"robot{i}_eef_rot_axis_angle"
            
            if (pos_key in episode_data and rot_key in episode_data and
                episode_data[pos_key] is not None and episode_data[rot_key] is not None):
                next2world_mat = pose_to_mat(
                    np.concatenate([
                        episode_data[pos_key][local_idx + 1],
                        episode_data[rot_key][local_idx + 1],
                    ], axis=-1)
                )
                curr2world_mat = pose_to_mat(
                    np.concatenate([
                        episode_data[pos_key][local_idx],
                        episode_data[rot_key][local_idx],
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
            if grip_key in episode_data and episode_data[grip_key] is not None:
                next_grip = episode_data[grip_key][local_idx + 1]
                curr_grip = episode_data[grip_key][local_idx]
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
        frame_data["actions"] = np.zeros(action_dim, dtype=np.float32)
    
    return frame_data


def _process_image(image_data, img_size, target_h=224, target_w=224):
    """处理图像数据"""
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


# ============================================================================
# 主转换器类
# ============================================================================
class ZarrToLeRobotConverter:
    """ViTaMin-B Zarr 格式到 LeRobot 格式的转换器（内存节约版）"""
    
    def __init__(self, 
                 zarr_path, 
                 output_repo_id, 
                 fps=30, 
                 state_dim=20, 
                 action_dim=20,
                 language_instruction=["perform bimanual manipulation task"]):
        """初始化转换器（仅分析结构，不加载数据）"""
        self.zarr_path = Path(zarr_path)
        self.output_repo_id = output_repo_id
        
        if not self.zarr_path.exists():
            raise ValueError(f"Zarr 文件不存在: {self.zarr_path}")
        
        # 临时打开 store 用于分析结构
        print(f"分析 Zarr 数据结构: {self.zarr_path}")
        store = ZipStore(self.zarr_path, mode="r")
        self.zarr_root = zarr.open_group(store=store, mode="r")
        self.data = self.zarr_root["data"]
        
        # 分析结构
        self.analyze_zarr_structure()
        
        # 获取图像形状
        self.img_size = (224, 224, 3)
        if len(self.camera_keys) > 0:
            first_camera_rgb = _process_image(self.data[self.camera_keys[0]][0], (224, 224, 3))
            self.img_size = first_camera_rgb.shape
        
        # 关闭临时 store（进程池会重新打开）
        store.close()
        
        self.fps = fps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.language_instruction = language_instruction
    
    def analyze_zarr_structure(self):
        """分析 Zarr 数据结构"""
        print("\n" + "="*70)
        print("分析 Zarr 数据结构")
        print("="*70)
        
        keys = list(self.data.keys())
        
        # 检测机器人数量
        self.robot_keys = [k for k in keys if k.startswith('robot') and 'eef_pos' in k]
        self.num_robots = len(self.robot_keys)
        print(f"  - 机器人数量: {self.num_robots}")
        
        # 检测相机数量
        self.camera_keys = [k for k in keys if k.startswith('camera') and ('rgb' in k)]
        self.num_cameras = len(self.camera_keys)
        print(f"  - 相机数量: {self.num_cameras}")
        
        # 检测触觉传感器数量
        self.tactile_keys = [k for k in keys if k.startswith('camera') and ('tactile' in k)]
        self.num_tactiles = len(self.tactile_keys)
        print(f"  - 触觉传感器数量: {self.num_tactiles}")
        
        # 获取 episode 信息
        self.episode_ends = self.zarr_root["meta"]["episode_ends"][:]
        n_episodes = len(self.episode_ends)
        n_steps = self.episode_ends[-1] if len(self.episode_ends) > 0 else 0
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        print("="*70)
    
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
    
    def convert_all_episodes(self, num_workers=None):
        """
        🔥 使用进程池 + 批量读取转换所有 episodes
        
        Args:
            num_workers: 进程池大小，None 时自动设置
        """
        n_episodes = len(self.episode_ends)
        n_steps = self.episode_ends[-1] if len(self.episode_ends) > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🚀 开始转换（内存节约模式）")
        print(f"{'='*70}")
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        print(f"  - 转换策略: 进程池 + 按 episode 批量读取")
        
        # 设置进程数
        if num_workers is None:
            num_workers = min(4, cpu_count() or 4)
        num_workers = max(1, min(num_workers, n_episodes))
        
        print(f"  - 进程数: {num_workers}")
        print(f"{'='*70}\n")
        
        # 创建数据集
        dataset = self.create_lerobot_dataset()
        
        # 准备参数列表
        args_list = []
        for ep_idx in range(n_episodes):
            if ep_idx == 0:
                start_idx = 0
            else:
                start_idx = self.episode_ends[ep_idx - 1]
            stop_idx = self.episode_ends[ep_idx]
            
            args_list.append((
                ep_idx,
                start_idx,
                stop_idx,
                self.num_robots,
                self.state_dim,
                self.action_dim,
                self.language_instruction,
                self.img_size
            ))
        
        # ====================================================================
        # 🔥 使用进程池并行处理（每个进程独立 store，批量读取 episode）
        # ====================================================================
        print("使用进程池处理 episodes...")
        results_by_idx = [None] * n_episodes
        
        with Pool(processes=num_workers, 
                  initializer=_worker_init, 
                  initargs=(str(self.zarr_path),)) as pool:
            
            # 使用 imap_unordered 提高效率（结果无序，但我们会重排）
            for ep_idx, frame_list in tqdm(
                pool.imap_unordered(_build_episode_frames_worker, args_list),
                total=n_episodes,
                desc="转换 episodes",
                ncols=70
            ):
                results_by_idx[ep_idx] = frame_list
        
        # ====================================================================
        # 主进程按顺序写入 dataset
        # ====================================================================
        print("\n写入数据集...")
        total_frames = 0
        
        for ep_idx in tqdm(range(n_episodes), desc="保存 episodes", ncols=70):
            frame_list = results_by_idx[ep_idx]
            for frame_data in frame_list:
                dataset.add_frame(frame_data)
            dataset.save_episode()
            total_frames += len(frame_list)
        
        print(f"\n{'='*70}")
        print(f"✓ 转换完成!")
        print(f"{'='*70}")
        print(f"数据集保存位置: {dataset.root}")
        print(f"总 episodes: {n_episodes}")
        print(f"总帧数: {total_frames}")
        print(f"平均每个 episode 帧数: {total_frames / n_episodes:.1f}")
        
        return dataset


def main(data_name="_0118"):
    parser = argparse.ArgumentParser(
        description='转换 ViTaMin-B Zarr 数据到 LeRobot 格式（内存节约版）',
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
        default=f'chaoyi/{data_name}',
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
        '--num_workers',
        type=int,
        default=None,
        help='进程池大小（默认 min(4, CPU核心数)）'
    )
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"错误: 找不到 Zarr 文件: {zarr_path}")
        sys.exit(1)
    
    print("="*70)
    print("ViTaMin-B Zarr → LeRobot 转换（内存节约版）")
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
            num_workers=args.num_workers,
        )
        
    except Exception as e:
        print(f"\n错误: 转换失败")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main(data_name="example")