import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import OpenEXR
import json
import time
import logging
from datetime import datetime  # 用于获取当前小时分钟生成日志文件名

from ExampleModel import ExampleModel
import Utils

# ===================== 日志配置 =====================
def setup_logger():
    """配置日志：文件名为train-小时分钟-日-月-年.log，同时输出到控制台和文件"""
    # 获取当前时间的小时和分钟（例如：14点35分 → 1435）
    current_time = datetime.now()
    # 核心改动：新增日、月、年的格式符，文件名规则变为 train-小时分钟-日-月-年.log
    # %H=小时(24制)、%M=分钟、%d=日、%m=月、%Y=4位年份
    log_filename = f"train-{current_time.strftime('%H%M-%d-%m-%Y')}.log"
    
    # 日志格式：时间 + 级别 + 消息
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 写入文件
            logging.StreamHandler()  # 输出到控制台（可选，如需仅文件输出可删除此行）
        ]
    )
    return logging.getLogger(__name__)

# 初始化日志器
logger = setup_logger()

# ===================== 时间格式化工具 =====================
def format_time(seconds):
    """将秒数格式化为 小时h 分钟m 秒.xxs"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"

def parse_times(time_str):
    return int(time_str) / 100.0

def main():
    # 记录程序总开始时间（总训练时间起点）
    total_start_time = time.time()
    logger.info("="*50)
    logger.info("程序启动，开始统计总训练时间")
    logger.info(f"使用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info("="*50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default='G:/Tencent_HPRC/HPRC_Test1/trunk/Data/Data_HPRC')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存参数和结果的文件夹
    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
     
    # 读取数据集下的配置文件
    config_file = 'config.json'
    times = ["0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1810", "1900", "2000", "2100", "2200", "2300"] 
    time_count = len(times) + 1
    with open(os.path.join(args.dataset, config_file), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 初始化整个数据集的指标list
    total_psnr = []
    total_ssim = []
    total_lpips = []

    # 统计每个光照贴图的耗时（存储字典：key=lightmap_id, value=总耗时/训练耗时）
    lightmap_time_stats = {}

    # 分别训练每张lightmap
    for idx, lightmap in enumerate(config['lightmap_list']):
        lightmap_id = f"{lightmap['level']}_{lightmap['id']}"
        logger.info(f"\n{'='*20} 开始处理光照贴图: {lightmap_id} (第{idx+1}/{len(config['lightmap_list'])}) {'='*20}")
        
        # 记录当前光照贴图的总处理开始时间（数据准备+训练+测试）
        lightmap_total_start = time.time()

        # 从配置文件中获取lightmap的基础信息
        id = lightmap['id']
        lightmap_names = lightmap['lightmaps']
        mask_names = lightmap['masks']
        resolution = lightmap['resolution']

        # 读取每张lightmap在不同时间的数据
        lightmap_in_different_time = []
        for time_idx in range(time_count):
            lightmap_path = os.path.join(args.dataset, "Data", lightmap_names[times[time_idx % (time_count - 1)]])
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))
        lightmap_data = torch.from_numpy(np.concatenate(lightmap_in_different_time, axis=0)).to(torch.float32).to(device)

        # 读取mask数据
        mask_in_different_time = []
        for time_idx in range(time_count):
            mask_path = os.path.join(args.dataset, "Data", mask_names[times[time_idx % (time_count - 1)]])
            mask_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_in_different_time.append(mask_bin.reshape(resolution['height'], resolution['width']))
        mask_data = np.concatenate(mask_in_different_time, axis=0).reshape(time_count, resolution['height'], resolution['width'])

        # 初始化模型
        model = ExampleModel(input_dim=6, output_dim=3, hidden_dim=args.hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 生成归一化坐标，并组合时间信息
        xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
        coords = np.stack([ys / (resolution['height'] - 1), xs / (resolution['width'] - 1)], axis=-1).reshape(-1, 2)
        coords = torch.from_numpy(coords).to(torch.float32).to(device)
        total_coords = []
        for time_idx in range(time_count):
            if time_idx < len(times):
                current_time = times[time_idx]
            else:
                current_time = "2400"
            # print(f"parse_times(current_time): {parse_times(current_time)}")
            alpha = torch.full((resolution['width'] * resolution['height'], 1), (parse_times(current_time) / 24)).to(device)
            coords_with_time = torch.cat([coords, alpha], dim=-1)
            total_coords.append(coords_with_time)
        total_coords = torch.cat(total_coords, dim=0)
        total_data = torch.cat([total_coords, lightmap_data], dim=-1)
        total_data = total_data[torch.randperm(total_data.shape[0])]
        
        # 记录训练循环开始时间（纯训练耗时起点）
        train_loop_start = time.time()
        
        # 训练循环
        batch_start = 0
        for it in range(args.iterations):
            batch_end = min(batch_start + args.batch_size, total_coords.shape[0])
            batch_data = total_data[batch_start:batch_end]

            pred = model(batch_data[:, :3])
            loss = criterion(pred, batch_data[:, 3:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start = batch_end
            if batch_start >= total_coords.shape[0]:
                batch_start = 0
                total_data = total_data[torch.randperm(total_data.shape[0])]

            if (it + 1) % 1000 == 0:
                logger.info(f"光照贴图 {lightmap_id} - 迭代 {it + 1} | 损失: {loss.item():.6f}")

        # 计算当前光照贴图的纯训练耗时
        train_loop_duration = time.time() - train_loop_start
        # 计算当前光照贴图的总处理耗时（数据准备+训练+测试）
        lightmap_total_duration = time.time() - lightmap_total_start

        # 保存模型参数
        all_params = []
        for param in model.parameters():
            all_params.append(param.detach().cpu().numpy().flatten())
        params_array = np.concatenate(all_params)
        params_array.astype(np.float32).tofile(f"./Parameters/model_{lightmap['level']}_{id}_params.bin")

        # 测试阶段
        with torch.no_grad():
            # 完整重建整张lightmap
            model.eval()
            pred_list = []
            for i in range((total_coords.shape[0] + args.batch_size - 1) // args.batch_size):
                batch_start = i * args.batch_size
                batch_end = min(batch_start + args.batch_size, total_coords.shape[0])
                batch_data = total_coords[batch_start:batch_end]
                pred = model(batch_data[:, :3])
                pred_list.append(pred)
            pred = torch.cat(pred_list, dim=0)
            pred = pred.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)

            # 将lightmap数据reshape为[time_count, height, width, 3]方便计算指标
            lightmap_data = lightmap_data.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)
            
            # 计算指标
            psnr_list = []
            ssim_list = []
            lpips_list = []

            # 计算指标，每256*256的区域计算一次
            part_size = 256
            rows = (lightmap_data.shape[2] + part_size - 1) // part_size
            cols = (lightmap_data.shape[3] + part_size - 1) // part_size
            for time_idx in range(time_count):
                # 无效区域置0
                pred[time_idx, :, mask_data[time_idx] <= 0] = 0
                for i in range(rows):
                    for j in range(cols):
                        start_row = i * part_size
                        end_row = min((i + 1) * part_size, lightmap_data.shape[2])
                        start_col = j * part_size
                        end_col = min((j + 1) * part_size, lightmap_data.shape[3])

                        lightmap_part = lightmap_data[[time_idx], :, start_row:end_row, start_col:end_col]
                        lightmap_reconstruct_part = pred[[time_idx], :, start_row:end_row, start_col:end_col]
                        mask_part = mask_data[time_idx, start_row:end_row, start_col:end_col]
                        valid_mask = mask_part >= 127

                        # 可以忽略完全无效的区域
                        if (np.any(valid_mask) and lightmap_part.max() != 0):
                            psnr_list.append(Utils.cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                            ssim_list.append(Utils.cal_ssim(lightmap_part, lightmap_reconstruct_part))
                            lpips_list.append(Utils.cal_lpips(lightmap_part, lightmap_reconstruct_part))

            # 汇总指标
            total_psnr.extend(psnr_list)
            total_ssim.extend(ssim_list)
            total_lpips.extend(lpips_list)
            
            # ===================== 记录当前光照贴图的时间统计 =====================
            lightmap_time_stats[lightmap_id] = {
                "纯训练耗时": format_time(train_loop_duration),
                "总处理耗时(数据+训练+测试)": format_time(lightmap_total_duration),
                "迭代次数": args.iterations,
                "批次大小": args.batch_size
            }
            logger.info(f"\n{'='*20} 光照贴图 {lightmap_id} 时间统计 {'='*20}")
            logger.info(f"纯训练耗时: {lightmap_time_stats[lightmap_id]['纯训练耗时']}")
            logger.info(f"总处理耗时: {lightmap_time_stats[lightmap_id]['总处理耗时(数据+训练+测试)']}")
            logger.info(f"PSNR: {np.mean(psnr_list):.2f} | SSIM: {np.mean(ssim_list):.4f} | LPIPS: {np.mean(lpips_list):.4f}")
            logger.info(f"模型大小: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")
            logger.info(f"数据大小: {lightmap_data.shape[0] * lightmap_data.shape[1] * lightmap_data.shape[2] * lightmap_data.shape[3] * 4 / 1024 / 1024:.2f} MB")

            # 保存拟合结果为exr文件
            pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            for time_idx in range(time_count):
                path = f'./ResultImages/reconstructed_{id}_{time_idx}.exr'
                header = OpenEXR.Header(pred[time_idx].shape[1], pred[time_idx].shape[0])
                channels = ['R', 'G', 'B']
                exr = OpenEXR.OutputFile(path, header)
                exr.writePixels({
                    c: pred[time_idx][..., i].tobytes()
                    for i, c in enumerate(channels)
                })
                exr.close()
        # break  # 测试用，如需训练所有lightmap请删除此行
            
    # ===================== 总训练时间统计 =====================
    total_duration = time.time() - total_start_time
    logger.info(f"\n{'='*30} 全局时间统计 {'='*30}")
    logger.info(f"程序总运行时间（总训练时间）: {format_time(total_duration)}")
    logger.info(f"\n各光照贴图耗时明细:")
    for lm_id, stats in lightmap_time_stats.items():
        logger.info(f"贴图id: {lm_id}:")
        logger.info(f"纯训练耗时: {stats['纯训练耗时']}")
        # logger.info(f"    - 总处理耗时: {stats['总处理耗时(数据+训练+测试)']}")
    logger.info(f"\n全局指标:")
    logger.info(f"PSNR (所有光照贴图): {np.mean(total_psnr):.2f}")
    logger.info(f"SSIM (所有光照贴图): {np.mean(total_ssim):.4f}")
    logger.info(f"LPIPS (所有光照贴图): {np.mean(total_lpips):.4f}")
    # logger.info("="*60)

if __name__ == "__main__":
    main()