#!/usr/bin/env python3
"""
创建decaying turbulence演化视频的脚本
从保存的npz文件中读取数据并生成视频

Usage:
    python create_turbulence_video.py --base_path /T7/2048_2 --start_step 1000 --end_step 12000 --step_interval 100
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import logging

# Try to import tqdm, fallback to basic progress if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing"):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            else:
                print(f"\r{desc}: {i+1}", end="", flush=True)
            yield item
        print()  # New line after completion

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_format(file_path):
    """检查npz文件的格式和内容"""
    try:
        data = np.load(file_path)
        logger.info(f"检查文件: {file_path}")
        logger.info(f"数据键: {list(data.keys())}")
        
        for key in data.keys():
            logger.info(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
            
        return data
    except Exception as e:
        logger.error(f"无法读取文件 {file_path}: {e}")
        return None

def get_simulation_file_list(base_path, start_step, end_step):
    """获取simulation文件列表 - 每1000步的文件"""
    base_path = Path(base_path)
    file_list = []
    
    # 查找从start_step到end_step范围内每1000步的文件
    for step in range(start_step, end_step + 1, 1000):
        filename = f"turbulence_2048_step{step}_2048x2048_index_1.npz"
        file_path = base_path / filename
        
        if file_path.exists():
            file_list.append((step, file_path))
            logger.info(f"找到文件: {filename}")
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    logger.info(f"总共找到 {len(file_list)} 个simulation文件")
    
    if not file_list:
        logger.info("未找到任何目标文件，列出目录中的一些npz文件:")
        all_npz = list(base_path.glob("*.npz"))
        for f in all_npz[:10]:
            logger.info(f"  {f.name}")
        if len(all_npz) > 10:
            logger.info(f"  ... 还有 {len(all_npz) - 10} 个文件")
    
    return file_list

def generate_frame_list(file_list, time_interval):
    """根据文件列表和时间间隔生成帧列表"""
    frame_list = []
    
    for file_step, file_path in file_list:
        # 检查文件内有多少时间步
        try:
            data = np.load(file_path)
            if 'u' in data:
                time_steps = data['u'].shape[0]  # 获取时间步数
                logger.info(f"文件 {file_path.name} 包含 {time_steps} 个时间步")
                
                # 根据时间间隔采样
                for t_idx in range(0, time_steps, time_interval):
                    # 计算全局时间步 (假设每个文件从该step开始的100步)
                    global_step = file_step + t_idx
                    frame_list.append((global_step, file_path, t_idx))
                    
        except Exception as e:
            logger.error(f"无法读取文件 {file_path}: {e}")
    
    logger.info(f"总共将生成 {len(frame_list)} 帧")
    return frame_list

def load_velocity_field(file_path, time_index=None):
    """从simulation npz文件中加载速度场数据"""
    try:
        data = np.load(file_path)
        
        # 检查是否有u, v分量
        if 'u' in data and 'v' in data:
            u = data['u']
            v = data['v']
            
            # 如果是3D数据 [time, y, x]，使用指定的时间索引
            if len(u.shape) > 2:
                if time_index is not None and time_index < u.shape[0]:
                    u = u[time_index]
                    v = v[time_index]
                else:
                    # 如果没有指定时间索引或索引超出范围，取最后一个时间步
                    u = u[-1]
                    v = v[-1]
                
            return u, v
        else:
            logger.error(f"文件 {file_path} 中没有找到 'u' 和 'v' 数据")
            return None, None
            
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {e}")
        return None, None

def calculate_velocity_magnitude(u, v):
    """计算速度幅值"""
    return np.sqrt(u**2 + v**2)

def calculate_vorticity(u, v):
    """计算涡度 ω = ∂v/∂x - ∂u/∂y"""
    # 使用有限差分计算偏导数
    dudx, dudy = np.gradient(u)
    dvdx, dvdy = np.gradient(v)
    
    vorticity = dvdx - dudy
    return vorticity

def create_single_frame(u, v, step, visualization_type='velocity_magnitude'):
    """创建单帧图像"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    if visualization_type == 'velocity_magnitude':
        field = calculate_velocity_magnitude(u, v)
        title = f'Velocity Magnitude - Step {step}'
        cmap = 'viridis'
        vmin, vmax = 0, None
    elif visualization_type == 'vorticity':
        field = calculate_vorticity(u, v)
        title = f'Vorticity - Step {step}'
        cmap = 'RdBu_r'
        vmax = np.abs(field).max()
        vmin = -vmax
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")
    
    im = ax.imshow(field, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 去除边距
    plt.tight_layout()
    
    return fig

def setup_ffmpeg():
    """设置ffmpeg路径"""
    import shutil
    
    # 尝试找到ffmpeg路径
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        return True
    
    # 尝试常见路径
    common_paths = [
        '/usr/local/bin/ffmpeg',
        '/opt/homebrew/bin/ffmpeg',
        '/usr/bin/ffmpeg'
    ]
    
    for path in common_paths:
        if Path(path).exists():
            plt.rcParams['animation.ffmpeg_path'] = path
            return True
    
    return False

def create_turbulence_video(base_path, start_step, end_step, time_interval, 
                          output_file='turbulence_evolution.mp4', 
                          visualization_type='velocity_magnitude',
                          fps=10):
    """创建湍流演化视频
    
    Args:
        base_path: 数据文件路径
        start_step: 开始步数 
        end_step: 结束步数
        time_interval: 每个文件内的时间步采样间隔
        output_file: 输出视频文件名
        visualization_type: 可视化类型
        fps: 视频帧率
    """
    
    # 设置ffmpeg
    if not setup_ffmpeg():
        logger.error("无法找到ffmpeg，请确保ffmpeg已安装并在PATH中")
        return
    
    # 获取simulation文件列表
    file_list = get_simulation_file_list(base_path, start_step, end_step)
    
    if not file_list:
        logger.error("没有找到任何simulation文件")
        return
    
    # 生成帧列表
    frame_list = generate_frame_list(file_list, time_interval)
    
    if not frame_list:
        logger.error("没有生成任何帧")
        return
    
    # 检查第一个文件的格式
    logger.info("检查数据格式...")
    first_data = check_data_format(file_list[0][1])
    if first_data is None:
        return
    
    logger.info(f"使用ffmpeg生成MP4视频")
    
    # 确保输出文件是MP4格式
    if not output_file.endswith('.mp4'):
        output_file = output_file.rsplit('.', 1)[0] + '.mp4'
    
    # 创建临时图像文件夹
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    logger.info("生成帧图像...")
    frame_files = []
    
    for i, (global_step, file_path, time_idx) in enumerate(tqdm(frame_list, desc="处理帧")):
        u, v = load_velocity_field(file_path, time_idx)
        
        if u is None or v is None:
            continue
            
        # 创建单帧图像
        fig = create_single_frame(u, v, global_step, visualization_type)
        
        # 保存帧
        frame_file = temp_dir / f"frame_{i:05d}.png"
        fig.savefig(frame_file, dpi=100, bbox_inches='tight')
        frame_files.append(frame_file)
        
        plt.close(fig)  # 释放内存
    
    logger.info(f"生成了 {len(frame_files)} 帧")
    
    # 使用matplotlib的animation模块创建视频
    if frame_files:
        logger.info("创建MP4视频...")
        
        try:
            # 读取第一帧来获取图像大小
            first_frame = plt.imread(frame_files[0])
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis('off')
            
            # 初始化图像显示
            im = ax.imshow(first_frame)
            
            def animate(frame_idx):
                if frame_idx < len(frame_files):
                    img = plt.imread(frame_files[frame_idx])
                    im.set_array(img)
                return [im]
            
            # 创建动画
            anim = animation.FuncAnimation(fig, animate, frames=len(frame_files), 
                                         interval=1000/fps, blit=True)
            
            # 使用ffmpeg写入器
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='PICT'), bitrate=1800)
            
            output_path = Path(output_file)
            anim.save(output_path, writer=writer)
            
            logger.info(f"MP4视频已保存到: {output_path}")
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"创建视频时出错: {e}")
            logger.info("请检查ffmpeg安装或尝试重新安装")
            return
    
    # 清理临时文件
    logger.info("清理临时文件...")
    for frame_file in frame_files:
        if frame_file.exists():
            frame_file.unlink()
    # 清理可能遗留的其他文件
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='创建decaying turbulence演化视频')
    
    parser.add_argument('--base_path', type=str, required=True,
                       help='数据文件的基础路径 (例如: /T7/2048_2)')
    parser.add_argument('--start_step', type=int, default=1000,
                       help='开始步数')
    parser.add_argument('--end_step', type=int, default=12000,
                       help='结束步数')
    parser.add_argument('--time_interval', type=int, default=100,
                       help='每个文件内的时间步采样间隔 (1=所有帧, 10=每10帧取1帧, 100=每个文件只取1帧)')
    parser.add_argument('--output_file', type=str, default='turbulence_evolution.mp4',
                       help='输出视频文件名')
    parser.add_argument('--visualization_type', type=str, default='velocity_magnitude',
                       choices=['velocity_magnitude', 'vorticity'],
                       help='可视化类型')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率')
    parser.add_argument('--check_format', action='store_true',
                       help='仅检查第一个文件的数据格式')
    
    args = parser.parse_args()
    
    if args.check_format:
        # 仅检查格式
        first_file = Path(args.base_path) / f"turbulence_2048_step{args.start_step}_2048x2048_index_1.npz"
        check_data_format(first_file)
    else:
        # 创建视频
        create_turbulence_video(
            base_path=args.base_path,
            start_step=args.start_step,
            end_step=args.end_step,
            time_interval=args.time_interval,
            output_file=args.output_file,
            visualization_type=args.visualization_type,
            fps=args.fps
        )

if __name__ == '__main__':
    main()
