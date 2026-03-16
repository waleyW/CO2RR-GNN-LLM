#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch slab optimization (non-batch-splitting version).
Supports multiple parent folders, and the user can specify whether to use POSCAR or CONTCAR.

Example:
python slab_opt_all.py \
  --input-folders /path/to/SlabSet1 /path/to/SlabSet2 \
  --out-folder /path/to/Optimized \
  --model /path/to/gemnet_t_direct_h512_all.pt \
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

# ========== Initialize OCP ==========
os.environ['OCP_ROOT'] = '/path/to/ocp'
sys.path.insert(0, '/path/to/ocp')

print("Initializing OCP modules...")
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports
setup_imports()
import ocpmodels.models, ocpmodels.trainers, ocpmodels.datasets, ocpmodels.tasks
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
print(f"✓ Registration completed: {len(registry.mapping.get('trainer', {}))} trainers, {len(registry.mapping.get('model', {}))} models")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ========== Utility functions ==========
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
    logger.info(f"Log file: {log_file}")
    return logger

def find_structure_files(input_folders, file_type):
    """
    Recursively search for the specified file type (POSCAR or CONTCAR)
    under multiple parent directories.
    """
    all_files = []
    for folder in input_folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"⚠️ Skipping non-existent folder: {folder}")
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

# ========== Main function ==========
def main():
    parser = argparse.ArgumentParser(description='OCP + GemNet-T slab batch optimization')
    parser.add_argument('--input-folders', nargs='+', required=True, help='Multiple input parent directories')
    parser.add_argument('--out-folder', required=True, help='Output directory')
    parser.add_argument('--model', required=True, help='Path to the GemNet-T model')
    parser.add_argument('--file-type', choices=['POSCAR', 'CONTCAR'], default='POSCAR', help='Specify the input file type')
    parser.add_argument('--fmax', type=float, default=0.05, help='Convergence criterion')
    parser.add_argument('--steps', type=int, default=200, help='Maximum optimization steps')
    args = parser.parse_args()

    logger = setup_logging(args.out_folder)
    start_time = time.time()
    os.makedirs(args.out_folder, exist_ok=True)

    # Search for all structure files
    all_files = find_structure_files(args.input_folders, args.file_type)
    total_files = len(all_files)
    logger.info(f"Found a total of {total_files} {args.file_type} files")

    if total_files == 0:
        logger.error("No matching files were found")
        return

    results = []
    success, failed = 0, 0

    with tqdm(total=total_files, desc="Optimizing slabs", unit="slab") as pbar:
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
                    logger.warning(f"CUDA error, switching to CPU: {e}")
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
                logger.info(f"{slab_name}: ✓ Converged, E={energy:.3f} eV, time={elapsed:.1f} s")
            except Exception as e:
                failed += 1
                logger.error(f"{slab_name}: Failed ({e})")
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

    # Save results
    df = pd.DataFrame(results)
    out_csv = Path(args.out_folder) / f"slab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to: {out_csv}")
    total_time = format_time(time.time() - start_time)
    logger.info(f"✅ All tasks completed: success={success}, failed={failed}, total time={total_time}")

if __name__ == "__main__":
    main()
