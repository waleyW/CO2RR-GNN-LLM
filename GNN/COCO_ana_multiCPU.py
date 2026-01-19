#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO structure parallel analyzer (English headers + bond/site flags + filtered table)
- Multiprocessing; uses SLURM_CPUS_PER_TASK if available
- Keeps original output columns/CSV unchanged
- Additionally writes a *filtered* CSV containing rows where:
    Min_CC_Ang <= cc_bond_cutoff  AND  Min_C_Metal_Dist_Ang <= c_m_top_cutoff
"""

import os, re, math, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

# -------------------- core helpers --------------------

def read_poscar(filepath):
    """Read POSCAR and return structure info dict, or None on failure."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        comment = lines[0].strip()
        scale = float(lines[1].strip())

        # lattice vectors (3x3)
        lattice_vectors = []
        for i in range(2, 5):
            lattice_vectors.append([float(x) for x in lines[i].split()])
        lattice_vectors = np.array(lattice_vectors) * scale

        # elements & counts (VASP5)
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]

        # selective dynamics?
        coord_start = 8
        if lines[7].strip().lower().startswith('s'):
            coord_start = 9

        # coord type
        coord_type = lines[coord_start - 1].strip().lower()
        is_direct = coord_type.startswith('d')

        # atom positions
        positions, atom_types = [], []
        atom_index = 0
        for element, count in zip(elements, counts):
            for _ in range(count):
                line = lines[coord_start + atom_index].split()
                pos = [float(x) for x in line[:3]]
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
    except Exception:
        return None


def distance(a, b):
    return float(np.linalg.norm(a - b))


def find_carbon_indices(structure):
    return [i for i, t in enumerate(structure['atom_types']) if t.upper() == 'C']


def find_metal_indices(structure):
    # Non-metal set uppercase to avoid misclassification
    non_metal = {'C', 'O', 'H', 'N', 'F', 'CL', 'BR', 'I', 'S', 'P'}
    return [i for i, t in enumerate(structure['atom_types']) if t.upper() not in non_metal]


def analyze_coco_structure(structure):
    """Return analysis dict or {'error': ...}"""
    if structure is None:
        return {'error': 'Failed to read POSCAR'}

    carbons = find_carbon_indices(structure)
    metals = find_metal_indices(structure)

    if len(carbons) < 2:
        return {
            'error': f'Only {len(carbons)} carbon atoms found (<2)',
            'carbon_count': len(carbons),
            'metal_count': len(metals)
        }

    pos = structure['positions']

    # all C–C distances
    cc_dists, cc_pairs = [], []
    for i in range(len(carbons)):
        for j in range(i + 1, len(carbons)):
            i1, i2 = carbons[i], carbons[j]
            d = distance(pos[i1], pos[i2])
            cc_dists.append(d)
            cc_pairs.append((i1, i2))

    if cc_dists:
        min_cc_idx = int(np.argmin(cc_dists))
        min_cc_dist = float(cc_dists[min_cc_idx])
        min_cc_pair = cc_pairs[min_cc_idx]
        avg_cc_dist = float(np.mean(cc_dists))
        max_cc_dist = float(np.max(cc_dists))
    else:
        min_cc_dist = avg_cc_dist = max_cc_dist = None
        min_cc_pair = None

    # each C to nearest metal
    c_metal = []
    for c_idx in carbons:
        cpos = pos[c_idx]
        best = float('inf')
        best_m = None
        for m_idx in metals:
            d = distance(cpos, pos[m_idx])
            if d < best:
                best = d
                best_m = m_idx
        c_metal.append({
            'carbon_idx': c_idx,
            'closest_metal_idx': best_m,
            'distance': (None if math.isinf(best) else float(best)),
            'metal_element': (structure['atom_types'][best_m] if best_m is not None else None)
        })

    # global min C–M among all carbons
    all_cm = [x['distance'] for x in c_metal if x['distance'] is not None]
    min_c_m = float(np.min(all_cm)) if all_cm else None

    return {
        'carbon_count': len(carbons),
        'metal_count': len(metals),
        'carbon_indices': carbons,
        'metal_indices': metals,
        'cc_analysis': {
            'all_cc_distances': [float(x) for x in cc_dists],
            'min_cc_distance': min_cc_dist,
            'min_cc_pair': min_cc_pair,
            'avg_cc_distance': avg_cc_dist,
            'max_cc_distance': max_cc_dist
        },
        'c_metal_analysis': c_metal,
        'metal_elements': list(sorted(set(structure['atom_types'][i] for i in metals))),
        'min_c_m_distance': min_c_m
    }


def extract_energy_from_filename(filename):
    m = re.search(r'POSCAR_opt_(-?\d+\.?\d*)eV\.vasp', filename)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

# -------------------- per-file worker --------------------

def process_one_file(args):
    coco_dir, poscar_file, cc_cut, cm_top_cut = args
    try:
        slab_name = coco_dir.name
        rel = poscar_file.relative_to(coco_dir)

        # ---------- SlabPath: keep ONLY the first directory (before the first "/") ----------
        if rel.parent == Path('.'):
            slab_path = 'root'
        else:
            parent_parts = rel.parent.parts
            slab_path = parent_parts[0] if parent_parts else 'root'

        # ---------- Site: keep ONLY the trailing digits after the last underscore ----------
        site = 'unknown_site'
        for part in rel.parts:
            m_site = re.search(r'_(\d+)$', part)
            if m_site:
                site = m_site.group(1)  # digits only, e.g., "05"
                break
        if site == 'unknown_site' and len(rel.parts) > 0:
            # fallback to last part; still try to strip to digits if pattern matches
            last_part = rel.parts[-1]
            m_site = re.search(r'_(\d+)$', last_part)
            site = m_site.group(1) if m_site else last_part

        energy = extract_energy_from_filename(poscar_file.name)
        structure = read_poscar(poscar_file)
        analysis = analyze_coco_structure(structure)

        result = {
            'SlabFolder': slab_name,
            'SlabPath': slab_path,          # <= modified
            'Site': site,                   # <= modified
            'POSCAR': poscar_file.name,
            'Energy_eV': energy,
            'FullPath': str(poscar_file),
            'Error': None,
            'C_Count': None,
            'Metal_Count': None
        }

        if 'error' in analysis:
            result.update({
                'Error': analysis.get('error', 'Unknown error'),
                'C_Count': analysis.get('carbon_count'),
                'Metal_Count': analysis.get('metal_count'),
                'CC_Bonded': False,
                'C_OnTop': False,
                'Min_C_Metal_Dist_Ang': None
            })
            return result

        cc = analysis['cc_analysis']
        cm = analysis['c_metal_analysis']
        metals_list = analysis.get('metal_elements', [])
        min_c_m = analysis.get('min_c_m_distance', None)

        # CC bonded flag (use min C–C distance)
        min_cc = cc['min_cc_distance']
        cc_bonded = (min_cc is not None and min_cc <= cc_cut)

        # C on-top flag (use global min C–M among all carbons)
        c_on_top = (min_c_m is not None and min_c_m <= cm_top_cut)

        result.update({
            'C_Count': analysis['carbon_count'],
            'Metal_Count': analysis['metal_count'],
            'Metal_Elements': ', '.join(metals_list) if metals_list else None,
            'Min_CC_Ang': min_cc,
            'Avg_CC_Ang': cc['avg_cc_distance'],
            'Max_CC_Ang': cc['max_cc_distance'],
            'Num_CC_Dist': len(cc['all_cc_distances']) if cc['all_cc_distances'] is not None else 0,
            'CC_Bonded': bool(cc_bonded),
            'C_OnTop': bool(c_on_top),
            'Min_C_Metal_Dist_Ang': min_c_m
        })

        # record first two carbons' nearest metal info
        for i, cm_info in enumerate(cm[:2]):
            prefix = f'C{i+1}'
            result[f'{prefix}_Nearest_Metal_Dist_Ang'] = cm_info['distance']
            result[f'{prefix}_Nearest_Metal_Elem'] = cm_info['metal_element']

        if len(cm) > 2:
            all_d = [x['distance'] for x in cm if x['distance'] is not None]
            if all_d:
                result['All_C_Metal_Avg_Ang'] = float(np.mean(all_d))
                result['All_C_Metal_Min_Ang'] = float(np.min(all_d))

        return result

    except Exception as e:
        return {
            'SlabFolder': coco_dir.name,
            'SlabPath': 'unknown',
            'Site': 'unknown',
            'POSCAR': getattr(poscar_file, 'name', 'unknown'),
            'Energy_eV': None,
            'FullPath': str(poscar_file),
            'Error': f'Exception: {e}',
            'C_Count': None,
            'Metal_Count': None,
            'CC_Bonded': False,
            'C_OnTop': False,
            'Min_C_Metal_Dist_Ang': None
        }

# -------------------- scan & main --------------------

def gather_tasks(root_path: Path):
    coco_dirs = [p for p in root_path.iterdir()
                 if p.is_dir() and 'COCO' in p.name.upper() and p.name.startswith('Slab_')]
    tasks = []
    for coco_dir in coco_dirs:
        for poscar_file in coco_dir.rglob('POSCAR_opt_*.vasp'):
            tasks.append((coco_dir, poscar_file))
    return coco_dirs, tasks


def main():
    parser = argparse.ArgumentParser(description="COCO parallel analyzer (English headers)")
    parser.add_argument('--root', type=str, default='PDS_OPT_GPU', help='Root directory containing Slab_*COCO* folders')
    parser.add_argument('--out', type=str, default='coco_structure_analysis.csv', help='Output CSV filename')
    parser.add_argument('--filtered_out', type=str, default='coco_structure_filtered.csv', help='Filtered CSV filename')
    parser.add_argument('--workers', type=int, default=0, help='Worker processes (0 = auto from SLURM_CPUS_PER_TASK)')
    parser.add_argument('--cc_bond_cutoff', type=float, default=1.70, help='Cutoff (Å) for C–C bond existence')
    parser.add_argument('--c_m_top_cutoff', type=float, default=2.20, help='Cutoff (Å) for C on-top adsorption by min C–Metal distance')
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"[ERROR] Root path not found: {root_path}")
        return

    coco_dirs, files = gather_tasks(root_path)
    if not coco_dirs:
        print("[ERROR] No Slab_*COCO* folders found.")
        return
    if not files:
        print("[ERROR] No POSCAR_opt_*.vasp files found.")
        return

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    auto_workers = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    workers = args.workers if args.workers > 0 else auto_workers
    workers = max(1, min(workers, len(files)))

    print("=" * 60)
    print(f"Root: {root_path}")
    print(f"COCO folders: {len(coco_dirs)}")
    print(f"Files to process: {len(files)}")
    print(f"Workers: {workers}")
    print(f"C–C bond cutoff: {args.cc_bond_cutoff} Å")
    print(f"C on-top cutoff (min C–M): {args.c_m_top_cutoff} Å")
    print("=" * 60)

    tasks = [(cdir, f, args.cc_bond_cutoff, args.c_m_top_cutoff) for (cdir, f) in files]

    results = []
    chunksize = max(1, len(files) // (workers * 4))
    with Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(process_one_file, tasks, chunksize=chunksize), start=1):
            results.append(row)
            if i % 200 == 0 or i == len(tasks):
                print(f"Progress: {i}/{len(tasks)}")

    if not results:
        print("[ERROR] No results to save.")
        return

    df = pd.DataFrame(results)
    # Save main table (unchanged)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Saved main table: {args.out}  (rows={len(df)})")

    # ---- New: filtered table (both distances ≤ thresholds) ----
    mask = (
        df['Min_CC_Ang'].notna() &
        df['Min_C_Metal_Dist_Ang'].notna() &
        (df['Min_CC_Ang'] <= args.cc_bond_cutoff) &
        (df['Min_C_Metal_Dist_Ang'] <= args.c_m_top_cutoff)
    )
    filtered_df = df[mask].copy()
    filtered_df.to_csv(args.filtered_out, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved filtered table: {args.filtered_out}  (rows={len(filtered_df)})")

    # quick stats
    if 'Min_CC_Ang' in df.columns:
        s = df['Min_CC_Ang'].dropna()
        if len(s) > 0:
            print("\n[Stats] C–C distances:")
            print(f"  min={s.min():.3f} Å, max={s.max():.3f} Å, mean={s.mean():.3f} Å")
            coupled = (s <= args.cc_bond_cutoff).sum()
            print(f"  CC_Bonded (≤ {args.cc_bond_cutoff} Å): {coupled}/{len(s)}")

    if 'Min_C_Metal_Dist_Ang' in df.columns:
        s = df['Min_C_Metal_Dist_Ang'].dropna()
        if len(s) > 0:
            print("\n[Stats] Min C–Metal distance (across carbons):")
            print(f"  min={s.min():.3f} Å, max={s.max():.3f} Å, mean={s.mean():.3f} Å")
            ontop = (s <= args.c_m_top_cutoff).sum()
            print(f"  C_OnTop (≤ {args.c_m_top_cutoff} Å): {ontop}/{len(s)}")

    if 'Error' in df.columns:
        err = df['Error'].notna().sum()
        if err > 0:
            print(f"\n[Warn] Error/exception rows: {err}")

    print("\n[Done] COCO parallel analysis complete.")

if __name__ == "__main__":
    main()
