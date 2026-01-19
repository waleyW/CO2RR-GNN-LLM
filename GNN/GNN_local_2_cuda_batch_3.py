#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用OCP模块和本地GemNet-T模型进行批量优化
分批处理版：支持指定文件范围，每次脚本运行都从原始模型开始
"""

import os
import ase.io
import torch
import gc
import sys
import logging
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from ase.optimize import BFGS
import pandas as pd
from tqdm import tqdm

# ✅ 强力OCP初始化修复
# 设置环境变量
os.environ['OCP_ROOT'] = '/nesi/nobackup/uoa04335/WXY/Software/ocp'
sys.path.insert(0, '/nesi/nobackup/uoa04335/WXY/Software/ocp')

# 强制导入所有OCP模块以触发注册
print("正在初始化OCP模块...")
try:
    # 先导入注册相关模块
    from ocpmodels.common.registry import registry
    from ocpmodels.common.utils import setup_imports, setup_logging
    
    # 设置导入
    setup_imports()
    
    # 强制导入所有模块以触发注册
    import ocpmodels.models
    import ocpmodels.trainers
    import ocpmodels.datasets
    import ocpmodels.tasks
    
    # 检查注册状态
    trainer_count = len(registry.mapping.get('trainer', {}))
    model_count = len(registry.mapping.get('model', {}))
    print(f"✓ 注册完成: {trainer_count} trainers, {model_count} models")
    
    if trainer_count == 0:
        print("⚠️ 警告: trainer注册仍然为空，尝试手动注册...")
        # 手动导入具体的trainer
        from ocpmodels.trainers import ForcesTrainer, EnergyTrainer
        print(f"手动注册后: {len(registry.mapping.get('trainer', {}))} trainers")
    
except Exception as e:
    print(f"初始化过程中出现警告: {e}")

# 然后导入OCPCalculator
from ocpmodels.common.relaxation.ase_utils import OCPCalculator

# 推荐设置（减少显存碎片）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ✅ 添加CUDA调试设置
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步执行，便于调试

# 配置参数
chk_path = "/nesi/nobackup/uoa04335/WXY/GNN/Model/gemnet_t_direct_h512_all.pt"
ini_folder = '/nesi/nobackup/uoa04335/WXY/CO2RRChat/GNN_OPT/PDS_Sele'
fin_folder = '/nesi/nobackup/uoa04335/WXY/CO2RRChat/GNN_OPT/PDS_OPT_GPU'
fmax = 0.05
max_steps = 200

def setup_logging(output_dir, batch_id):
    """设置日志记录"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建带时间戳和批次ID的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ml_optimization_batch_{batch_id}_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"批次 {batch_id} 日志文件: {log_file}")
    return logger

def get_all_poscar_files(ini_folder):
    """获取所有POSCAR文件的路径列表"""
    poscar_files = []
    ini_path = Path(ini_folder)
    
    for lvl1 in ini_path.iterdir():
        if lvl1.is_dir():
            for lvl2 in lvl1.iterdir():
                if lvl2.is_dir():
                    for lvl3 in lvl2.iterdir():
                        if lvl3.is_dir():
                            for poscar_file in lvl3.glob("POSCAR"):
                                # 存储文件路径和相对路径信息
                                poscar_files.append({
                                    'file_path': poscar_file,
                                    'lvl1': lvl1.name,
                                    'lvl2': lvl2.name,
                                    'lvl3': lvl3.name,
                                    'relative_path': f"{lvl1.name}/{lvl2.name}/{lvl3.name}"
                                })
    
    return poscar_files

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OCP + GemNet-T 分批优化工具')
    parser.add_argument('--start', type=int, required=True, help='起始文件索引 (1-based)')
    parser.add_argument('--end', type=int, required=True, help='结束文件索引 (inclusive)')
    parser.add_argument('--batch_id', type=int, required=True, help='批次ID')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"OCP + GemNet-T 分批优化工具 - 批次 {args.batch_id}")
    print(f"处理文件范围: {args.start} - {args.end}")
    print("="*70)
    
    # ✅ 设置日志记录
    logger = setup_logging(fin_folder, args.batch_id)
    logger.info("="*70)
    logger.info(f"开始批次 {args.batch_id} 优化任务")
    logger.info(f"文件范围: {args.start} - {args.end}")
    logger.info(f"模型文件: {chk_path}")
    logger.info(f"输入目录: {ini_folder}")
    logger.info(f"输出目录: {fin_folder}")
    logger.info(f"收敛标准: {fmax} eV/Å")
    logger.info(f"最大步数: {max_steps}")
    
    if not os.path.exists(chk_path):
        logger.error(f"模型文件不存在: {chk_path}")
        return
    
    if not os.path.exists(ini_folder):
        logger.error(f"输入目录不存在: {ini_folder}")
        return
    
    print("正在获取所有POSCAR文件列表...")
    all_poscar_files = get_all_poscar_files(ini_folder)
    total_files = len(all_poscar_files)
    
    logger.info(f"总文件数: {total_files}")
    
    # 验证索引范围
    if args.start < 1 or args.end > total_files or args.start > args.end:
        logger.error(f"无效的文件范围: {args.start}-{args.end}, 总文件数: {total_files}")
        return
    
    # 获取当前批次要处理的文件 (转换为0-based索引)
    batch_files = all_poscar_files[args.start-1:args.end]
    batch_size = len(batch_files)
    
    logger.info(f"当前批次文件数: {batch_size}")
    print(f"当前批次将处理 {batch_size} 个文件")
    
    os.makedirs(fin_folder, exist_ok=True)
    
    results_data = []
    energy_stats = {
        'CHCOH': [], 'CHCHOH': [], 'CCH': []
    }
    success_count = 0
    failed_count = 0
    skipped_count = 0
    processing_times = []
    
    fin_path = Path(fin_folder)
    
    with tqdm(total=batch_size, desc=f"批次 {args.batch_id}", unit="files") as pbar:
        for i, file_info in enumerate(batch_files):
            poscar_file = file_info['file_path']
            lvl1_name = file_info['lvl1']
            lvl2_name = file_info['lvl2']
            lvl3_name = file_info['lvl3']
            
            file_start_time = time.time()
            global_index = args.start + i  # 全局文件索引
            
            pbar.set_description(f"批次{args.batch_id} [{global_index}/{total_files}] {lvl3_name}")
            
            # 创建输出目录
            output_subfolder = fin_path / lvl1_name / lvl2_name / lvl3_name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已处理
            existing_files = list(output_subfolder.glob(f"{poscar_file.name}_*"))
            if existing_files:
                logger.info(f"跳过 {poscar_file.name} (已处理)")
                skipped_count += 1
                pbar.update(1)
                continue
            
            try:
                atoms = ase.io.read(poscar_file)
                logger.info(f"处理: {poscar_file.name} ({len(atoms)} atoms) - 全局索引: {global_index}")
                
                if len(atoms) == 0:
                    raise ValueError("结构为空")
                if len(atoms) > 200:
                    raise ValueError(f"结构过大: {len(atoms)} atoms")
                
                positions = atoms.get_positions()
                if torch.isnan(torch.tensor(positions)).any() or torch.isinf(torch.tensor(positions)).any():
                    raise ValueError("原子坐标包含NaN或Inf值")
                
                # 尝试GPU，失败则切换CPU
                try:
                    calc = OCPCalculator(checkpoint_path=chk_path, cpu=False, max_neighbors=30)
                    atoms.calc = calc
                    _ = atoms.get_potential_energy()
                except Exception as cuda_error:
                    if "CUDA" in str(cuda_error):
                        logger.warning(f"CUDA错误，切换到CPU模式: {cuda_error}")
                        calc = OCPCalculator(checkpoint_path=chk_path, cpu=True, max_neighbors=30)
                        atoms.calc = calc
                    else:
                        raise cuda_error
                
                opt = BFGS(atoms, trajectory=None, logfile=None)
                is_converged = opt.run(fmax=fmax, steps=max_steps)
                
                # 获取能量
                try:
                    energy = atoms.get_potential_energy()
                    energy_str = f"_{energy:.6f}eV"
                    molecule_type = lvl3_name.split('_')[0]  # 从 CHC_01 提取 CHC
                    if molecule_type in energy_stats:
                        energy_stats[molecule_type].append({
                            'slab': f"{lvl1_name}/{lvl2_name}/{lvl3_name}",
                            'energy': energy,
                            'atoms': len(atoms),
                            'converged': is_converged
                        })
                except Exception as e:
                    energy = None
                    energy_str = ""
                    logger.warning(f"无法获取 {poscar_file.name} 的能量: {e}")
                
                # 保存优化结构
                output_file = output_subfolder / f"{poscar_file.name}_opt{energy_str}.vasp"
                ase.io.write(output_file, atoms, format="vasp", sort=True)
                
                process_time = time.time() - file_start_time
                processing_times.append(process_time)
                
                # 更新进度条
                if len(processing_times) >= 3:
                    avg_time = sum(processing_times[-10:]) / min(len(processing_times), 10)
                    remaining_files = batch_size - (success_count + failed_count + skipped_count + 1)
                    eta_seconds = avg_time * remaining_files
                    eta_str = format_time(eta_seconds)
                    
                    pbar.set_postfix({
                        'ETA': eta_str,
                        'avg_time': f"{avg_time:.1f}s/file",
                        'success': success_count + 1
                    })
                
                # 记录结果
                result_record = {
                    'batch_id': args.batch_id,
                    'global_index': global_index,
                    'slab': f"{lvl1_name}/{lvl2_name}/{lvl3_name}",
                    'molecule': molecule_type,
                    'filename': poscar_file.name,
                    'atoms_count': len(atoms),
                    'energy_eV': energy,
                    'converged': is_converged,
                    'process_time_s': process_time,
                    'output_file': output_file.name,
                    'status': 'success'
                }
                results_data.append(result_record)
                success_count += 1
                conv_status = "✓收敛" if is_converged else "✗未收敛"
                logger.info(f"  → {conv_status}, E={energy:.6f}eV, 用时:{process_time:.1f}s")
                
            except Exception as e:
                failed_count += 1
                process_time = time.time() - file_start_time
                processing_times.append(process_time)
                logger.error(f"处理 {poscar_file.name} 失败: {str(e)}")
                
                # 保存错误信息
                error_file = output_subfolder / f"{poscar_file.name}_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"优化失败: {str(e)}\n")
                    f.write(f"处理时间: {process_time:.1f}s\n")
                    f.write(f"时间戳: {datetime.now()}\n")
                    f.write(f"批次ID: {args.batch_id}\n")
                    f.write(f"全局索引: {global_index}\n")
                
                result_record = {
                    'batch_id': args.batch_id,
                    'global_index': global_index,
                    'slab': f"{lvl1_name}/{lvl2_name}/{lvl3_name}",
                    'molecule': lvl3_name,
                    'filename': poscar_file.name,
                    'atoms_count': 0,
                    'energy_eV': None,
                    'converged': False,
                    'process_time_s': process_time,
                    'output_file': error_file.name,
                    'status': f'failed: {str(e)[:50]}'
                }
                results_data.append(result_record)
                
            finally:
                # 清理内存
                try:
                    # 安全地删除变量
                    if 'atoms' in locals():
                        del atoms
                    if 'calc' in locals():
                        del calc
                    
                    # GPU内存清理
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # 确保GPU操作完成
                    
                    gc.collect()
                except Exception as cleanup_error:
                    # 记录清理过程中的错误，但不影响主流程
                    logger.debug(f"清理内存时出现错误: {cleanup_error}")
                
                pbar.update(1)
    
    # 保存结果
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_csv = Path(fin_folder) / f"optimization_results_batch_{args.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"批次 {args.batch_id} 详细结果已保存到: {results_csv}")
    
    # 计算统计信息
    total_time = time.time() - start_time
    total_processed = success_count + failed_count + skipped_count
    avg_time_per_file = total_time / total_processed if total_processed > 0 else 0
    
    logger.info("\n" + "="*70)
    logger.info(f"批次 {args.batch_id} 优化完成!")
    logger.info(f"文件范围: {args.start} - {args.end}")
    logger.info(f"总处理时间: {format_time(total_time)}")
    logger.info(f"平均每文件: {avg_time_per_file:.1f}s")
    logger.info(f"批次文件数: {batch_size}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"跳过文件: {skipped_count}")
    logger.info(f"失败文件: {failed_count}")
    if total_processed > 0:
        logger.info(f"成功率: {success_count/total_processed*100:.1f}%")
    
    # 能量统计
    if any(energy_stats.values()):
        logger.info("\n" + "="*50)
        logger.info(f"批次 {args.batch_id} 分子类型能量统计:")
        
        energy_summary = []
        for molecule, data_list in energy_stats.items():
            if data_list:
                energies = [d['energy'] for d in data_list]
                convergence_rate = sum(1 for d in data_list if d['converged']) / len(data_list) * 100
                stats = {
                    'batch_id': args.batch_id,
                    'molecule': molecule,
                    'count': len(data_list),
                    'avg_energy': sum(energies) / len(energies),
                    'min_energy': min(energies),
                    'max_energy': max(energies),
                    'std_energy': (sum((e - sum(energies)/len(energies))**2 for e in energies) / len(energies))**0.5,
                    'convergence_rate': convergence_rate
                }
                energy_summary.append(stats)
                
                logger.info(f"{molecule:>8}: {len(data_list):>3}个, "
                           f"平均: {stats['avg_energy']:>8.3f}±{stats['std_energy']:>5.3f}eV, "
                           f"范围: [{stats['min_energy']:>7.3f}, {stats['max_energy']:>7.3f}]eV, "
                           f"收敛率: {convergence_rate:>5.1f}%")
        
        if energy_summary:
            energy_df = pd.DataFrame(energy_summary)
            energy_csv = Path(fin_folder) / f"energy_statistics_batch_{args.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            energy_df.to_csv(energy_csv, index=False)
            logger.info(f"批次 {args.batch_id} 能量统计已保存到: {energy_csv}")
    
    logger.info(f"批次 {args.batch_id} 结果保存在: {fin_folder}")
    logger.info("="*70)


if __name__ == "__main__":
    main()