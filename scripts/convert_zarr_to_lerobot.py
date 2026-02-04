#!/usr/bin/env python3
"""
将 ViTaMin-B Zarr 格式数据转换为 LeRobot 格式

Zarr 格式结构:
- root["data"]: 包含所有数据键
  - robot{i}_eef_pos: 末端执行器位置 (按全局步骤索引)
  - robot{i}_eef_rot_axis_angle: 末端执行器旋转 (按全局步骤索引)
  - robot{i}_gripper_width: 夹爪宽度 (按全局步骤索引)
  - camera{i}_rgb: 视觉图像 (按全局步骤索引)
- root["meta"]["episode_ends"]: episode 结束位置数组

LeRobot 格式结构:
- 数据集按 episode 组织，每个 episode 包含多个帧
- 每帧包含以下特征:
  - observation.images.camera0/camera1: RGB 相机图像 (H, W, 3)
  - observation.images.tactile_left_0/tactile_right_0/tactile_left_1/tactile_right_1: 触觉传感器图像 (H, W, 3)
  - observation.state: 状态向量，依次包含每个机器人相对初始位置的位姿 + 夹爪宽度(2 * 7维)、左机械臂相对右机械臂的位姿(6维)，共20维
  - action: 动作向量，依次包含每个机器人的位姿变化量 + 夹爪宽度变化量(2 * 7维)，共14维
  - language_instruction: 任务描述文本
  - task: 任务名称
- 数据集元数据包含 fps (采集频率) 和 robot_type (机器人类型)

"""

import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import zarr
from zarr.storage import ZipStore
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    if "libavformat" in str(e) or "libavcodec" in str(e):
        sys.exit(
            "PyAV 需要 FFmpeg 6.x。请先安装 FFmpeg：\n"
            "  方案 A（推荐）：conda install -c conda-forge 'ffmpeg>=6.1'\n"
            "  然后使用 bash data/run_convert.sh 运行（会自动使用 conda 的 FFmpeg）\n"
            "  方案 B：bash install_sys_deps.sh（系统 apt 安装）"
        )
    raise
import imagecodecs

# 注册图像解码器
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()
from utils.pose_util import pose_to_mat, mat_to_pose

class ZarrToLeRobotConverter:
    """ViTaMin-B Zarr 格式到 LeRobot 格式的转换器"""
    
    def __init__(self, 
    zarr_path, output_repo_id, fps=30, 
    state_dim=20, 
    action_dim=20,
    language_instruction=["perform bimanual manipulation task"]):
        """
        初始化转换器
        
        Args:
            zarr_path: Zarr 文件路径 (.zarr 或 .zarr.zip)
            output_repo_id: LeRobot 数据集名称，格式: "username/dataset-name"
            fps: 数据采集频率 (Hz)
        """
        self.zarr_path = Path(zarr_path)
        self.output_repo_id = output_repo_id

        if not self.zarr_path.exists():
            raise ValueError(f"Zarr 文件不存在: {self.zarr_path}")
        
        # 加载 Zarr 数据
        print(f"加载 Zarr 数据: {self.zarr_path}")
        store = ZipStore(self.zarr_path, mode="a")
        self.zarr_root = zarr.group(store)
        self.data = self.zarr_root["data"]  # 数据在 root["data"] 下
        
        print(f"Zarr 数据包含的键: {list(self.data.keys())}")

        # 分析 Zarr 结构
        self.robot_keys, self.camera_keys, self.tactile_keys, self.num_robots, self.num_cameras, self.num_tactiles = self.analyze_zarr_structure()

        # 获取图像形状（从第一个相机数据的第一帧）
        self.img_size=(224,224,3)
        if len(self.camera_keys) > 0:
            first_camera_rgb = self._process_image(self.data[self.camera_keys[0]][0])
            self.img_size = first_camera_rgb.shape  # (H, W, 3)
        
        self.fps = fps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.language_instruction = language_instruction
    
    def get_episode_info(self):
        """从 zarr root 获取 episode 结构信息"""
        episode_ends = self.zarr_root["meta"]["episode_ends"][:]
        n_episodes = len(episode_ends)
        n_steps = episode_ends[-1] if len(episode_ends) > 0 else 0
        return n_episodes, n_steps, episode_ends
    
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
        
        # 检测机器人数量
        robot_keys = [k for k in keys if k.startswith('robot') and 'eef_pos' in k]
        num_robots = len(robot_keys)
        print(f"  - 机器人数量: {num_robots}")

        # 检测相机数量
        camera_keys = [k for k in keys if k.startswith('camera') and ('rgb' in k)]
        num_cameras = len(camera_keys)
        print(f"  - 相机数量: {num_cameras}")

        # 检测触觉传感器数量
        tactile_keys = [k for k in keys if k.startswith('camera') and ('tactile' in k)]
        num_tactiles = len(tactile_keys)
        print(f"  - 触觉传感器数量: {num_tactiles}")
        
        # 获取 episode 信息
        n_episodes, n_steps, episode_ends = self.get_episode_info()
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        
        return robot_keys, camera_keys, tactile_keys, num_robots, num_cameras, num_tactiles
    
    def create_lerobot_dataset(self):
        """
        创建 LeRobot 数据集结构
        
        Args:
            image_shape: 图像尺寸 (height, width, channels)
            state_dim: 状态向量维度
            action_dim: 动作向量维度
        """
        print(f"\n创建 LeRobot 数据集:")
        print(f"  - 数据集 ID: {self.output_repo_id}")
        print(f"  - 图像形状: {self.img_size}")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim}")
        print(f"  - 采集频率: {self.fps} Hz")
        
        # 定义数据集特征
        features = {
            # 图像数据（根据实际相机数量调整）
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
            # 机器人状态
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": ["observation.state"],
            },
            # 动作
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
    
    def convert_all_episodes(self):
        """
        转换所有 episodes
        
        Args:
            language_instruction: 任务描述
        """
        
        # 获取 episode 信息
        n_episodes, n_steps, episode_ends = self.get_episode_info()
        
        print(f"\n数据维度:")
        print(f"  - Episodes: {n_episodes}")
        print(f"  - 总步骤数: {n_steps}")
        print(f"  - 图像形状: {self.img_size}")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim}")
        
        # 创建 LeRobot 数据集
        dataset = self.create_lerobot_dataset()
        
        # 转换每个 episode
        print(f"\n开始转换 {n_episodes} 个 episodes...")
        total_frames = 0
        
        for ep_idx in tqdm(range(n_episodes), desc="转换 episodes"):
            num_frames = self._convert_episode(
                dataset,
                ep_idx,
                episode_ends,
            )
            dataset.save_episode()
            total_frames += num_frames
        
        # 关闭 store
        store = self.zarr_root.store
        if hasattr(store, 'close'):
            store.close()
        
        print(f"\n{'='*70}")
        print(f"✓ 转换完成!")
        print(f"{'='*70}")
        print(f"数据集保存位置: {dataset.root}")
        print(f"总 episodes: {n_episodes}")
        print(f"总帧数: {total_frames}")
        print(f"平均每个 episode 帧数: {total_frames / n_episodes:.1f}")
        
        return dataset
    
    def _build_frame_data(self, step_idx, stop_idx):
        """
        构建单帧数据（线程安全地只读取 zarr / 计算，不写入 dataset）
        """
        frame_data = {}
        # 语言指令 / 任务名（按全局 step 对齐）
        if step_idx < len(self.language_instruction):
            frame_data["task"] = self.language_instruction[step_idx]
        else:
            frame_data["task"] = self.language_instruction[-1]

        # 图像
        camera_mappings = {
            "camera0_rgb": "observation.images.camera0",
            "camera1_rgb": "observation.images.camera1",
        }

        for cam_key, feature_key in camera_mappings.items():
            if cam_key in self.data.keys():
                img_data = self.data[cam_key][step_idx]
                frame_data[feature_key] = self._process_image(img_data)
            else:
                # 如果没有对应的相机，使用零图像
                frame_data[feature_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        tactile_mappings = {
            "camera0_left_tactile": "observation.images.tactile_left_0",
            "camera0_right_tactile": "observation.images.tactile_right_0",
            "camera1_left_tactile": "observation.images.tactile_left_1",
            "camera1_right_tactile": "observation.images.tactile_right_1",
        }

        for tac_key, feature_key in tactile_mappings.items():
            if tac_key in self.data.keys():
                img_data = self.data[tac_key][step_idx]
                frame_data[feature_key] = self._process_image(img_data)
            else:
                # 如果没有对应的相机，使用零图像
                frame_data[feature_key] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 状态向量：拼接所有机器人 1. 相对初始位置的位姿 2. 夹爪距离 3. 相对另一个夹爪的位姿
        state_features = []
        curr2world_mat_0 = None
        curr2world_mat_1 = None

        for i in range(self.num_robots):
            # 1. 相对初始位姿
            init2world_mat = pose_to_mat(self.data[f"robot{i}_demo_start_pose"][0])
            curr2world_mat = pose_to_mat(
                np.concatenate(
                    [
                        self.data[f"robot{i}_eef_pos"][step_idx],
                        self.data[f"robot{i}_eef_rot_axis_angle"][step_idx],
                    ],
                    axis=-1,
                )
            )
            if i == 0:
                curr2world_mat_0 = curr2world_mat
            else:
                curr2world_mat_1 = curr2world_mat

            curr2init_mat = np.linalg.inv(init2world_mat) @ curr2world_mat
            curr2init_pose = mat_to_pose(curr2init_mat)
            state_features.extend(curr2init_pose)  # rel pos + rel rot_vec, 6d

            # 2. 夹爪距离
            grip_key = f"robot{i}_gripper_width"
            if grip_key in self.data.keys():
                grip_data = self.data[grip_key][step_idx]
                try:
                    if hasattr(grip_data, "__len__"):
                        state_features.append(float(grip_data[0]))  # gripper width, 1d
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
            state_features.extend(rel_0to1_pose)  # rel pos + rel rot_vec, 6d

        # 状态维度裁剪 / 补零
        expected_state_dim = self.state_dim
        if len(state_features) < expected_state_dim:
            state_features.extend([0.0] * (expected_state_dim - len(state_features)))
        elif len(state_features) > expected_state_dim:
            state_features = state_features[:expected_state_dim]

        frame_data["observation.state"] = np.asarray(
            state_features, dtype=np.float32
        )  # totally 20d

        # 动作（变化量）
        if step_idx < stop_idx - 1:
            action_features = []
            for i in range(self.num_robots):
                # Δ action
                pos_key = f"robot{i}_eef_pos"
                rot_key = f"robot{i}_eef_rot_axis_angle"
                next2world_mat = pose_to_mat(
                    np.concatenate(
                        [
                            self.data[pos_key][step_idx + 1],
                            self.data[rot_key][step_idx + 1],
                        ],
                        axis=-1,
                    )
                )
                curr2world_mat = pose_to_mat(
                    np.concatenate(
                        [
                            self.data[pos_key][step_idx],
                            self.data[rot_key][step_idx],
                        ],
                        axis=-1,
                    )
                )

                next2curr_mat = np.linalg.inv(curr2world_mat) @ next2world_mat
                next2curr_pos = mat_to_pose(next2curr_mat)[:3]
                # 提取旋转矩阵前两列并展平（6d）
                rot_cols = next2curr_mat[:3, :2].reshape(-1)
                action_feature_9d = np.concatenate(
                    [next2curr_pos, rot_cols], axis=0
                )  # 拼接为9d向量
                action_features.extend(action_feature_9d)  # rel pos + rel mat first two cols, 9d

                # Δ gripper
                grip_key = f"robot{i}_gripper_width"
                if grip_key in self.data.keys():
                    next_grip = self.data[grip_key][step_idx + 1]
                    curr_grip = self.data[grip_key][step_idx]
                    try:
                        if hasattr(next_grip, "__len__") and hasattr(
                            curr_grip, "__len__"
                        ):
                            delta_grip = float(next_grip[0] - curr_grip[0])
                        elif hasattr(next_grip, "__len__"):
                            delta_grip = float(next_grip[0] - curr_grip)
                        elif hasattr(curr_grip, "__len__"):
                            delta_grip = float(next_grip - curr_grip[0])
                        else:
                            delta_grip = float(next_grip - curr_grip)
                        action_features.append(delta_grip)  # gripper width, 1d
                    except Exception:
                        action_features.append(0.0)
                else:
                    action_features.append(0.0)

            frame_data["actions"] = np.asarray(
                action_features, dtype=np.float32
            )  # totally 20d
        else:
            # 最后一帧：零动作
            action_dim = self.action_dim
            frame_data["actions"] = np.zeros(action_dim, dtype=np.float32)

        return frame_data

    def _convert_episode(self, dataset, ep_idx, episode_ends):
        """
        转换单个 episode
        
        Args:
            dataset: LeRobotDataset 对象
            ep_idx: episode 索引
            episode_ends: episode 结束位置数组
        """
        
        # 获取该 episode 的步骤范围
        episode_slice = self.get_episode_slice(ep_idx, episode_ends)
        start_idx, stop_idx = episode_slice.start, episode_slice.stop

        # 先使用多线程并行构建每一帧的数据（只读 Zarr，不写磁盘）
        indices = list(range(start_idx, stop_idx))
        # 适当限制线程数，避免过多线程竞争
        max_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 结果顺序与 indices 保持一致，保证帧顺序与原始数据一致
            frame_list = list(
                executor.map(
                    lambda s: self._build_frame_data(s, stop_idx),
                    indices,
                )
            )

        # 再在主线程中按顺序写入数据集，避免多线程写 dataset 的并发问题
        for frame_data in frame_list:
            dataset.add_frame(frame_data)

        return stop_idx - start_idx
    
    def _process_image(self, image_data, target_h=224, target_w=224):
        """
        处理图像数据：解码并调整大小
        
        Args:
            image_data: 图像数据（可能是 numpy 数组或 bytes）
            target_h: 目标高度
            target_w: 目标宽度
        
        Returns:
            HxWx3 uint8 RGB numpy 数组
        """
        # 解码图像数据
        if isinstance(image_data, bytes):
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        elif hasattr(image_data, "shape"):
            img = image_data
            # 确保是 RGB 格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 如果已经是 RGB，直接使用
                pass
            elif len(img.shape) == 2:
                # 灰度图转 RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 确保是 uint8 类型
        if img.dtype == np.uint8:
            pass
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # 调整大小到目标尺寸
        try:
            img = cv2.resize(img, (target_w, target_h))
        except Exception:
            img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        return img


def main():
    parser = argparse.ArgumentParser(
        description='转换 ViTaMin-B Zarr 数据到 LeRobot 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--zarr_path',
        type=str,
        default=r'/home/liuchaoyi/openpi_chaoyi/openpi_chaoyi/data/example.zarr.zip',
        help='Zarr 文件路径 (.zarr 或 .zarr.zip)'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default=r'/home/liuchaoyi/openpi_chaoyi/openpi_chaoyi/data/lerobot/chaoyi/_test_127',
        help='LeRobot 数据集 ID，格式: username/dataset-name'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='数据采集频率 (Hz)，默认: 30'
    )
    parser.add_argument(
        '--language_instruction',
        type=str,
        default=["perform bimanual manipulation task"],
        help='任务描述'
    )
    
    args = parser.parse_args()
    
    # 验证 repo_id 格式
    if '/' not in args.repo_id:
        print(f"错误: repo_id 格式必须是 'username/dataset-name'，当前: {args.repo_id}")
        sys.exit(1)
    
    # 检查 Zarr 文件
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"错误: 找不到 Zarr 文件: {zarr_path}")
        sys.exit(1)
    
    print("="*70)
    print("ViTaMin-B Zarr 数据转换为 LeRobot 格式")
    print("="*70)
    print(f"Zarr 文件: {zarr_path.absolute()}")
    print(f"目标数据集 ID: {args.repo_id}")
    print(f"采集频率: {args.fps} Hz")
    print(f"任务描述: {args.language_instruction}")
    print("="*70)
    print()
    
    try:
        # 执行转换
        converter = ZarrToLeRobotConverter(
            zarr_path=args.zarr_path,
            output_repo_id=args.repo_id,
            fps=args.fps,
            state_dim=20,
            action_dim=20,
            language_instruction=args.language_instruction
        )
        
        dataset = converter.convert_all_episodes()
        
    except Exception as e:
        print(f"\n错误: 转换失败")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()