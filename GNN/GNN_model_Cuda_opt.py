#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using the OCP module and the local GemNet-T model for batch structure optimization.
Batch-processing version: supports specifying a range of files, and each script run starts from the original model.
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

# Robust OCP initialization fix
# Set environment variables
os.environ['OCP_ROOT'] = 'your ocp path'
sys.path.insert(0, 'your ocp path')


print("Initializing OCP modules...")
try:
    # First import registry-related modules
    from ocpmodels.common.registry import registry
    from ocpmodels.common.utils import setup_imports, setup_logging
    
    # Set up imports
    setup_imports()
    
    # Force import all modules to trigger registration
    import ocpmodels.models
    import ocpmodels.trainers
    import ocpmodels.datasets
    import ocpmodels.tasks
    
    # Check registration status
    trainer_count = len(registry.mapping.get('trainer', {}))
    model_count = len(registry.mapping.get('model', {}))
    print(f"✓ Registration completed: {trainer_count} trainers, {model_count} models")
    
    if trainer_count == 0:
        print(" Warning: trainer registry is still empty, attempting manual registration...")
        # Manually import specific trainers
        from ocpmodels.trainers import ForcesTrainer, EnergyTrainer
        print(f"After manual registration: {len(registry.mapping.get('trainer', {}))} trainers")
    
except Exception as e:
    print(f"Warning during initialization: {e}")

# Import OCPCalculator
from ocpmodels.common.relaxation.ase_utils import OCPCalculator

# Recommended setting (reduce GPU memory fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous execution for easier debugging

# Configuration parameters
chk_path = "gemnet_t_direct_h512_all.pt(model_path)"
ini_folder = 'structures_path_folder'
fin_folder = 'output_folder'

fmax = 0.05
max_steps = 200

def setup_logging(output_dir, batch_id):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ml_optimization_batch_{batch_id}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"batch {batch_id} log_file: {log_file}")
    return logger

def get_all_poscar_files(ini_folder):
    poscar_files = []
    ini_path = Path(ini_folder)
    
    for lvl1 in ini_path.iterdir():
        if lvl1.is_dir():
            for lvl2 in lvl1.iterdir():
                if lvl2.is_dir():
                    for lvl3 in lvl2.iterdir():
                        if lvl3.is_dir():
                            for poscar_file in lvl3.glob("POSCAR"):
                                poscar_files.append({
                                    'file_path': poscar_file,
                                    'lvl1': lvl1.name,
                                    'lvl2': lvl2.name,
                                    'lvl3': lvl3.name,
                                    'relative_path': f"{lvl1.name}/{lvl2.name}/{lvl3.name}"
                                })
    
    return poscar_files

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def main():
    parser = argparse.ArgumentParser(description='OCP + GemNet-T')
    parser.add_argument('--start', type=int, required=True, help='start (1-based)')
    parser.add_argument('--end', type=int, required=True, help='final (inclusive)')
    parser.add_argument('--batch_id', type=int, required=True, help='batch_ID')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"OCP + GemNet-T - batch {args.batch_id}")
    print(f"scale: {args.start} - {args.end}")
    print("="*70)
    

    logger = setup_logging(fin_folder, args.batch_id)
    logger.info("="*70)
    logger.info(f"Starting optimization task for batch {args.batch_id}")
    logger.info(f"File range: {args.start} - {args.end}")
    logger.info(f"Model checkpoint: {chk_path}")
    logger.info(f"Input directory: {ini_folder}")
    logger.info(f"Output directory: {fin_folder}")
    logger.info(f"Convergence criterion: {fmax} eV/Å")
    logger.info(f"Maximum optimization steps: {max_steps}")
    
    if not os.path.exists(chk_path):
        logger.error(f"Model checkpoint not found: {chk_path}")
        return
    
    if not os.path.exists(ini_folder):
        logger.error(f"Input directory not found: {ini_folder}")
        return
    

    all_poscar_files = get_all_poscar_files(ini_folder)
    total_files = len(all_poscar_files)
    
    logger.info(f"total_files{total_files}")
    

    if args.start < 1 or args.end > total_files or args.start > args.end:
        logger.error(f"Invalid file range: {args.start}-{args.end}, total files: {total_files}")
        return
    

    batch_files = all_poscar_files[args.start-1:args.end]
    batch_size = len(batch_files)
    
    logger.info(f"Current batch size: {batch_size}")

    
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
    
    with tqdm(total=batch_size, desc=f"batch {args.batch_id}", unit="files") as pbar:
        for i, file_info in enumerate(batch_files):
            poscar_file = file_info['file_path']
            lvl1_name = file_info['lvl1']
            lvl2_name = file_info['lvl2']
            lvl3_name = file_info['lvl3']
            
            file_start_time = time.time()
            global_index = args.start + i 
            
            pbar.set_description(f"批次{args.batch_id} [{global_index}/{total_files}] {lvl3_name}")
            
            
            output_subfolder = fin_path / lvl1_name / lvl2_name / lvl3_name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            

            existing_files = list(output_subfolder.glob(f"{poscar_file.name}_*"))
            if existing_files:
                logger.info(f"Skipping {poscar_file.name} (already processed)")
                skipped_count += 1
                pbar.update(1)
                continue
            
            try:
                atoms = ase.io.read(poscar_file)
                logger.info(f"process: {poscar_file.name} ({len(atoms)} atoms) - Index: {global_index}")
                
                if len(atoms) == 0:
                    raise ValueError("empty")
                if len(atoms) > 200:
                    raise ValueError(f"Structure too large: {len(atoms)} atoms")
                
                positions = atoms.get_positions()
                if torch.isnan(torch.tensor(positions)).any() or torch.isinf(torch.tensor(positions)).any():
                    raise ValueError("Atomic coordinates contain NaN or Inf values")
                
                
                try:
                    calc = OCPCalculator(checkpoint_path=chk_path, cpu=False, max_neighbors=30)
                    atoms.calc = calc
                    _ = atoms.get_potential_energy()
                except Exception as cuda_error:
                    if "CUDA" in str(cuda_error):
                        logger.warning(f"CUDA error, switching to CPU mode: {cuda_error}")
                        calc = OCPCalculator(checkpoint_path=chk_path, cpu=True, max_neighbors=30)
                        atoms.calc = calc
                    else:
                        raise cuda_error
                
                opt = BFGS(atoms, trajectory=None, logfile=None)
                is_converged = opt.run(fmax=fmax, steps=max_steps)
                

                try:
                    energy = atoms.get_potential_energy()
                    energy_str = f"_{energy:.6f}eV"
                    molecule_type = lvl3_name.split('_')[0] 
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
                    logger.warning(f"cannot get {poscar_file.name} energys: {e}")
                
     
                output_file = output_subfolder / f"{poscar_file.name}_opt{energy_str}.vasp"
                ase.io.write(output_file, atoms, format="vasp", sort=True)
                
                process_time = time.time() - file_start_time
                processing_times.append(process_time)
                

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
                conv_status = "✓ Converged" if is_converged else "✗ Not converged"
                logger.info(f"  → {conv_status}, E={energy:.6f} eV, time: {process_time:.1f} s")
                
            except Exception as e:
                failed_count += 1
                process_time = time.time() - file_start_time
                processing_times.append(process_time)
                logger.error(f"Failed to process {poscar_file.name}: {str(e)}")
                
                # 保存错误信息
                error_file = output_subfolder / f"{poscar_file.name}_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"{str(e)}\n")
                    f.write(f"{process_time:.1f}s\n")
                    f.write(f" {datetime.now()}\n")
                    f.write(f" {args.batch_id}\n")
                    f.write(f"{global_index}\n")
                
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

                try:

                    if 'atoms' in locals():
                        del atoms
                    if 'calc' in locals():
                        del calc
                    

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize() 
                    
                    gc.collect()
                except Exception as cleanup_error:

                    logger.debug(f"cleanup_error: {cleanup_error}")
                
                pbar.update(1)
    

    if results_data:
        results_df = pd.DataFrame(results_data)
        results_csv = Path(fin_folder) / f"optimization_results_batch_{args.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Detailed results for batch {args.batch_id} have been saved to: {results_csv}")
    

    total_time = time.time() - start_time
    total_processed = success_count + failed_count + skipped_count
    avg_time_per_file = total_time / total_processed if total_processed > 0 else 0
    
    logger.info("\n" + "="*70)
    logger.info(f" {args.batch_id} success")
    logger.info(f" {args.start} - {args.end}")
    logger.info(f"{format_time(total_time)}")
    logger.info(f" {avg_time_per_file:.1f}s")
    logger.info(f" {batch_size}")
    logger.info(f"{success_count}")
    logger.info(f"{skipped_count}")
    logger.info(f" {failed_count}")
    if total_processed > 0:
        logger.info(f"success_rate: {success_count/total_processed*100:.1f}%")
    

    if any(energy_stats.values()):
        logger.info("\n" + "="*50)
        logger.info(f"{args.batch_id} energy:")
        
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
                           f"average: {stats['avg_energy']:>8.3f}±{stats['std_energy']:>5.3f}eV, "
                           f"scale: [{stats['min_energy']:>7.3f}, {stats['max_energy']:>7.3f}]eV, "
                           f"convergence_rate: {convergence_rate:>5.1f}%")
        
        if energy_summary:
            energy_df = pd.DataFrame(energy_summary)
            energy_csv = Path(fin_folder) / f"energy_statistics_batch_{args.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            energy_df.to_csv(energy_csv, index=False)
            logger.info(f"Batch {args.batch_id} energy statistics saved to: {energy_csv}")
    
    logger.info(f"batch {args.batch_id} results are saved in: {fin_folder}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
