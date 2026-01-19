#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO adsorption parallel analyzer (English column names)
- Multiprocessing (Pool + tqdm)
- Thresholds:
    * --co_bond_cutoff (Å): max allowed CO bond length
    * --c_m_top_cutoff (Å): max allowed min C–Metal distance
- Outputs:
    1) Main CSV (--out): full results, English headers
    2) Filtered CSV (--filtered_out): rows satisfying BOTH thresholds
    3) Copy POSCARs into two folders (preserving tree):
       - CO_pass/ (satisfy BOTH)
       - CO_fail/ (others)
"""

import os
import re
import time
import shutil
import psutil
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

BOND_THRESHOLDS = {
    'C-O': {
        'single': 1.43,      # for labeling only
        'double': 1.23,
        'triple': 1.13,
        'max_bond': 1.6
    },
    'C-Metal': {
        'strong': 2.0,       # for labeling only
        'weak': 2.5,
        'max_interaction': 3.0
    }
}

def read_poscar(filepath: Path):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        comment = lines[0].strip()
        scale = float(lines[1].strip())

        lattice_vectors = []
        for i in range(2, 5):
            lattice_vectors.append([float(x) for x in lines[i].split()])
        lattice_vectors = np.array(lattice_vectors) * scale

        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]

        coord_start = 8
        if lines[7].strip().lower().startswith('s'):
            coord_start = 9

        coord_type = lines[coord_start - 1].strip().lower()
        is_direct = coord_type.startswith('d')

        positions, atom_types = [], []
        atom_index = 0
        for element, count in zip(elements, counts):
            for _ in range(count):
                if coord_start + atom_index >= len(lines):
                    break
                toks = lines[coord_start + atom_index].split()
                pos = [float(x) for x in toks[:3]]
                positions.append(pos)
                atom_types.append(element)
                atom_index += 1

        positions = np.array(positions)
        if is_direct:
            positions = np.dot(positions, lattice_vectors)

        return {
            'lattice_vectors': lattice_vectors,
            'positions': positions,
            'atom_types': atom_types,
            'elements': elements,
            'counts': counts,
            'comment': comment,
            'is_direct': is_direct
        }
    except Exception as e:
        return {'error': f"Read error: {e}"}

def dist(a, b): return float(np.linalg.norm(a - b))

def idx_by_element(structure, elem):
    return [i for i, t in enumerate(structure['atom_types']) if t.upper() == elem.upper()]

def metal_indices(structure):
    non_metal = {'C', 'O', 'H', 'N', 'F', 'CL', 'BR', 'I', 'S', 'P', 'B'}
    return [i for i, t in enumerate(structure['atom_types']) if t.upper() not in non_metal]

def label_co(distance):
    if distance is None:
        return 'No_Bond', 'No bond'
    th = BOND_THRESHOLDS['C-O']
    if distance <= th['triple'] + 0.05: return 'Triple', 'C≡O'
    if distance <= th['double'] + 0.05: return 'Double', 'C=O'
    if distance <= th['single'] + 0.05: return 'Single', 'C–O'
    if distance <= th['max_bond']:      return 'Weak', 'Weak C–O'
    return 'No_Bond', 'No bond'

def label_c_m(distance):
    if distance is None:
        return 'No_Interaction', 'No interaction'
    th = BOND_THRESHOLDS['C-Metal']
    if distance <= th['strong']:           return 'Strong', 'Strong'
    if distance <= th['weak']:             return 'Weak', 'Weak'
    if distance <= th['max_interaction']:  return 'Very_Weak', 'Very weak'
    return 'No_Interaction', 'No interaction'

def analyze_one(structure):
    if structure is None or 'error' in structure:
        return structure

    c_idxs = idx_by_element(structure, 'C')
    o_idxs = idx_by_element(structure, 'O')
    m_idxs = metal_indices(structure)

    if len(c_idxs) == 0: return {'error': 'No carbon found'}
    if len(o_idxs) == 0: return {'error': 'No oxygen found'}

    pos = structure['positions']

    # CO distances (min)
    co_d, co_pairs = [], []
    for ci in c_idxs:
        for oi in o_idxs:
            d = dist(pos[ci], pos[oi])
            co_d.append(d); co_pairs.append((ci, oi))
    if co_d:
        k = int(np.argmin(co_d))
        min_co = float(co_d[k]); min_pair = co_pairs[k]
        co_type, co_desc = label_co(min_co)
        avg_co = float(np.mean(co_d))
    else:
        min_co = None; min_pair = None
        co_type, co_desc = 'No_Bond', 'No bond'
        avg_co = None

    # C to nearest metal (per carbon)
    c_m = []
    for ci in c_idxs:
        best = float('inf'); best_m = None
        for mi in m_idxs:
            d = dist(pos[ci], pos[mi])
            if d < best:
                best = d; best_m = mi
        if best != float('inf'):
            best = float(best)
            itype, idesc = label_c_m(best)
            melem = structure['atom_types'][best_m]
        else:
            best = None; itype, idesc = 'No_Interaction', 'No interaction'
            melem = None
        c_m.append({
            'carbon_idx': ci,
            'closest_metal_idx': best_m,
            'closest_metal_element': melem,
            'distance': best,
            'interaction_type': itype,
            'interaction_desc': idesc
        })

    all_cm = [x['distance'] for x in c_m if x['distance'] is not None]
    min_cm = float(np.min(all_cm)) if all_cm else None

    return {
        'carbon_count': len(c_idxs),
        'oxygen_count': len(o_idxs),
        'metal_count': len(m_idxs),
        'co_analysis': {
            'min_co_distance': min_co,
            'min_co_pair': min_pair,
            'avg_co_distance': avg_co,
            'co_bond_type': co_type,
            'co_bond_desc': co_desc
        },
        'c_metal_analysis': c_m,
        'metal_elements': list(set(structure['atom_types'][i] for i in m_idxs)),
        'min_c_m_distance': min_cm
    }

def extract_energy(fname):
    m = re.search(r'POSCAR_opt_(-?\d+\.?\d*)eV\.vasp', fname)
    return float(m.group(1)) if m else None

def worker(args):
    poscar_file, co_dir = args
    try:
        slab_folder = co_dir.name
        rel = poscar_file.relative_to(co_dir)

        # ==== SlabPath: keep ONLY the first directory (before first "/") ====
        if rel.parent == Path('.'):
            slab_path = 'root'
        else:
            parent_parts = rel.parent.parts
            slab_path = parent_parts[0] if parent_parts else 'root'

        # ==== Site: keep ONLY digits after the last underscore, e.g., "CO_05" -> "05" ====
        site = 'unknown_site'
        for part in rel.parts:
            m_site = re.search(r'_(\d+)$', part)
            if m_site:
                site = m_site.group(1)   # digits only
                break
        if site == 'unknown_site' and len(rel.parts) > 0:
            last_part = rel.parts[-1]
            m_site = re.search(r'_(\d+)$', last_part)
            site = m_site.group(1) if m_site else last_part

        energy = extract_energy(poscar_file.name)
        s = read_poscar(poscar_file)
        a = analyze_one(s)

        out = {
            'SlabFolder': slab_folder,
            'SlabPath': slab_path,   # modified
            'Site': site,            # modified
            'POSCAR': poscar_file.name,
            'Energy_eV': energy,
            'FullPath': str(poscar_file),
            'Error': None
        }

        if a is None or 'error' in a:
            out['Error'] = None if a is None else a.get('error', 'Unknown error')
            return out

        co = a['co_analysis']
        out.update({
            'C_Count': a['carbon_count'],
            'O_Count': a['oxygen_count'],
            'Metal_Count': a['metal_count'],
            'Metal_Elements': ', '.join(a['metal_elements']),
            'CO_BondLength_Ang': co['min_co_distance'],
            'CO_BondType': co['co_bond_type'],
            'CO_BondDesc': co['co_bond_desc'],
            'Avg_CO_Dist_Ang': co['avg_co_distance'],
            'Min_AllC_Metal_Dist_Ang': a.get('min_c_m_distance', None)
        })

        for i, cm in enumerate(a['c_metal_analysis']):
            prefix = f'C{i+1}' if len(a['c_metal_analysis']) > 1 else 'C'
            out[f'{prefix}_Nearest_Metal_Dist_Ang'] = cm['distance']
            out[f'{prefix}_Nearest_Metal_Elem'] = cm['closest_metal_element']
            out[f'{prefix}_Metal_Interaction_Type'] = cm['interaction_type']
            out[f'{prefix}_Metal_Interaction_Desc'] = cm['interaction_desc']

        return out

    except Exception as e:
        return {
            'SlabFolder': co_dir.name if 'co_dir' in locals() else 'Unknown',
            'POSCAR': poscar_file.name if 'poscar_file' in locals() else 'Unknown',
            'FullPath': str(poscar_file) if 'poscar_file' in locals() else 'Unknown',
            'Error': f'Exception: {e}'
        }

def cpu_info():
    cpu_count = mp.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    return {'cpu_count': cpu_count, 'cpu_usage': cpu_usage,
            'mem_total_gb': mem.total/(1024**3), 'mem_avail_gb': mem.available/(1024**3)}

def collect_files(root_path: str):
    root = Path(root_path)
    files = []
    co_dirs = []
    for item in root.iterdir():
        if item.is_dir() and item.name.upper() == 'SLAB_CO':
            co_dirs.append(item)
    if not co_dirs:
        print("❌ No SLAB_CO folder found"); return []
    print(f"Found {len(co_dirs)} SLAB_CO folder(s)")
    for co_dir in co_dirs:
        for f in co_dir.rglob('POSCAR_opt_*.vasp'):
            files.append((f, co_dir))
    return files

def analyze_parallel(root_path: str, nproc: int):
    info = cpu_info()
    print(f"🖥️ CPU {info['cpu_count']} | Mem avail {info['mem_avail_gb']:.1f} GB")
    nproc = min(max(1, nproc), info['cpu_count'])
    print(f"🚀 Using {nproc} processes")

    file_list = collect_files(root_path)
    if not file_list:
        return []

    print(f"Total POSCAR files: {len(file_list)}")
    t0 = time.time(); results = []
    with Pool(processes=nproc) as pool:
        with tqdm(total=len(file_list), desc="Analyzing CO", unit="file") as pbar:
            for r in pool.imap_unordered(worker, file_list, chunksize=10):
                if r is not None: results.append(r)
                pbar.update(1)
    dt = time.time() - t0
    print(f"✅ Done in {dt:.1f}s | {len(file_list)/dt:.1f} files/s")
    return results

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def copy_with_tree(src: Path, base_dir: Path, out_root: Path):
    try:
        rel = src.relative_to(base_dir)
    except Exception:
        rel = src.name
    dst = out_root / rel
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser(description="CO adsorption parallel analyzer (English headers)")
    ap.add_argument('--root', type=str, default='PDS_OPT_GPU', help='Root folder containing SLAB_CO')
    ap.add_argument('--out', type=str, default='co_structure_analysis.csv', help='Main CSV path')
    ap.add_argument('--filtered_out', type=str, default='co_structure_filtered.csv', help='Filtered CSV path')
    ap.add_argument('--co_bond_cutoff', type=float, default=1.30, help='CO bond length cutoff (Å)')
    ap.add_argument('--c_m_top_cutoff', type=float, default=2.20, help='Min C–Metal distance cutoff (Å)')
    ap.add_argument('--pass_dir', type=str, default='CO_pass', help='Folder for passing POSCARs')
    ap.add_argument('--fail_dir', type=str, default='CO_fail', help='Folder for failing POSCARs')
    ap.add_argument('--workers', type=int, default=0, help='Processes (0=auto)')
    args = ap.parse_args()

    if not os.path.exists(args.root):
        print(f"❌ Path not found: {args.root}"); return

    cpu_total = mp.cpu_count()
    nproc = args.workers if args.workers > 0 else cpu_total

    try:
        results = analyze_parallel(args.root, nproc)
        if not results:
            print("❌ No results"); return

        df = pd.DataFrame(results)
        df.to_csv(args.out, index=False, encoding='utf-8-sig')
        print(f"✅ Main CSV saved: {args.out} (rows={len(df)})")

        # Build filter columns (English)
        if 'Min_AllC_Metal_Dist_Ang' in df.columns:
            min_cm = df['Min_AllC_Metal_Dist_Ang']
        else:
            cm_cols = [c for c in df.columns if c.endswith('_Nearest_Metal_Dist_Ang')]
            min_cm = df[cm_cols].min(axis=1, skipna=True) if cm_cols else pd.Series([np.nan]*len(df))
        co_len = df['CO_BondLength_Ang'] if 'CO_BondLength_Ang' in df.columns else pd.Series([np.nan]*len(df))

        mask = (co_len.notna() & min_cm.notna() &
                (co_len <= args.co_bond_cutoff) &
                (min_cm <= args.c_m_top_cutoff))

        filtered = df[mask].copy()
        filtered.to_csv(args.filtered_out, index=False, encoding='utf-8-sig')
        print(f"✅ Filtered CSV saved: {args.filtered_out} (rows={len(filtered)})")

        pass_root = Path(args.pass_dir); fail_root = Path(args.fail_dir)
        ensure_dir(pass_root); ensure_dir(fail_root)

        slab_co_dirs = [p for p in Path(args.root).iterdir() if p.is_dir() and p.name.upper() == 'SLAB_CO']
        base_dir = slab_co_dirs[0] if slab_co_dirs else Path(args.root)

        print("📂 Copying POSCAR files into CO_pass/ and CO_fail/ ...")
        copied_pass = copied_fail = 0
        for i, row in df.iterrows():
            src = Path(str(row.get('FullPath', '')))
            if not src.exists():
                continue
            if mask.loc[i]:
                copy_with_tree(src, base_dir, pass_root); copied_pass += 1
            else:
                copy_with_tree(src, base_dir, fail_root); copied_fail += 1
        print(f"✅ Copy done: CO_pass={copied_pass} | CO_fail={copied_fail}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
