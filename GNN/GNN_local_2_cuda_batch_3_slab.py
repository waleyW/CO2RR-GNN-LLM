#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量Slab优化（无分批版）
支持多个父文件夹，用户可指定使用POSCAR或CONTCAR
例：
python slab_opt_all.py \
  --input-folders /path/SlabSet1 /path/SlabSet2 \
  --out-folder /path/Optimized \
  --model /path/gemnet_t_direct_h512_all.pt \
  --file-type POSCAR
"""

import os
import ase.io
import torch
import gc
import sys
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path
from ase.optimize import BFGS
import pandas as pd
from tqdm import tqdm

# ========== 初始化OCP ==========
os.environ['OCP_ROOT'] = '/nesi/nobackup/uoa04335/WXY/Software/ocp'
sys.path.insert(0, '/nesi/nobackup/uoa04335/WXY/Software/ocp')

print("正在初始化OCP模块...")
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports
setup_imports()
import ocpmodels.models, ocpmodels.trainers, ocpmodels.datasets, ocpmodels.tasks
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
print(f"✓ 注册完成: {len(registry.mapping.get('trainer', {}))} trainers, {len(registry.mapping.get('model', {}))} models")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ========== 工具函数 ==========
def setup_logging(output_dir):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"slab_optimization_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger

def find_structure_files(input_folders, file_type):
    """
    递归查找多个父目录下指定类型文件 (POSCAR或CONTCAR)
    """
    all_files = []
    for folder in input_folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"⚠️ 跳过不存在的文件夹: {folder}")
            continue
        for root, _, files in os.walk(folder):
            if file_type.lower() in [f.lower() for f in files]:
                all_files.append(Path(root) / file_type)
    return all_files

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='OCP + GemNet-T Slab批量优化')
    parser.add_argument('--input-folders', nargs='+', required=True, help='多个输入父目录')
    parser.add_argument('--out-folder', required=True, help='输出目录')
    parser.add_argument('--model', required=True, help='GemNet-T模型路径')
    parser.add_argument('--file-type', choices=['POSCAR', 'CONTCAR'], default='POSCAR', help='指定读取文件类型')
    parser.add_argument('--fmax', type=float, default=0.05, help='收敛标准')
    parser.add_argument('--steps', type=int, default=200, help='最大步数')
    args = parser.parse_args()

    logger = setup_logging(args.out_folder)
    start_time = time.time()
    os.makedirs(args.out_folder, exist_ok=True)

    # 搜索所有结构文件
    all_files = find_structure_files(args.input-folders, args.file_type)
    total_files = len(all_files)
    logger.info(f"共发现 {total_files} 个 {args.file_type} 文件")

    if total_files == 0:
        logger.error("未找到任何符合条件的文件")
        return

    results = []
    success, failed = 0, 0

    with tqdm(total=total_files, desc="Slab优化中", unit="slab") as pbar:
        for file_path in all_files:
            slab_name = file_path.parent.name
            out_dir = Path(args.out_folder) / slab_name
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            try:
                atoms = ase.io.read(file_path)
                calc = None
                try:
                    calc = OCPCalculator(checkpoint_path=args.model, cpu=False)
                    atoms.calc = calc
                    _ = atoms.get_potential_energy()
                except Exception as e:
                    logger.warning(f"CUDA错误，使用CPU: {e}")
                    calc = OCPCalculator(checkpoint_path=args.model, cpu=True)
                    atoms.calc = calc

                opt = BFGS(atoms, trajectory=None, logfile=None)
                is_conv = opt.run(fmax=args.fmax, steps=args.steps)
                energy = atoms.get_potential_energy()
                energy_str = f"{energy:.5f}eV"
                ase.io.write(out_dir / f"POSCAR_opt_{energy_str}.vasp", atoms, format="vasp", sort=True)
                elapsed = time.time() - t0

                results.append({
                    "slab": slab_name,
                    "energy_eV": energy,
                    "atoms": len(atoms),
                    "converged": is_conv,
                    "time_s": elapsed,
                    "path": str(out_dir)
                })
                success += 1
                logger.info(f"{slab_name}: ✓收敛 E={energy:.3f}eV 用时{elapsed:.1f}s")
            except Exception as e:
                failed += 1
                logger.error(f"{slab_name}: 失败 ({e})")
                with open(out_dir / "POSCAR_error.txt", "w") as f:
                    f.write(f"Error: {e}\n")
            finally:
                if 'calc' in locals():
                    del calc
                if 'atoms' in locals():
                    del atoms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                pbar.update(1)

    # 保存结果
    df = pd.DataFrame(results)
    out_csv = Path(args.out_folder) / f"slab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"结果已保存: {out_csv}")
    total_time = format_time(time.time() - start_time)
    logger.info(f"✅ 全部完成: 成功 {success}, 失败 {failed}, 总时间 {total_time}")

if __name__ == "__main__":
    main()
