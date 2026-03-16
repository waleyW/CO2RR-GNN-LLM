#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from pathlib import Path

def extract_energy_from_filename(filename):
    """
    Extract energy value from the filename
    Example: POSCAR_opt_-0.23305eV.vasp -> -0.23305
    """
    pattern = r'POSCAR_opt_([+-]?\d+\.?\d*)eV\.vasp'
    m = re.search(pattern, filename)
    return float(m.group(1)) if m else None


def scan_slab_directory(base_path):
    """
    Scan the structure where each Slab directory directly contains POSCAR_opt_*eV.vasp
    Example directory layout:

      base_path/
        Ag3Pt_mp-12065_proc1/
          POSCAR_opt_-0.76018eV.vasp
        Ag3Sn_mp-611_proc1/
          POSCAR_opt_-0.12345eV.vasp
        ...

    Return list[dict]: [{'Slab': slab_name, 'slab_eV': energy}, ...]
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    data = []

    # Only check first-level subdirectories, assuming each is a slab
    slab_dirs = [d for d in base.iterdir() if d.is_dir()]

    if not slab_dirs:
        print("No subdirectories (slabs) found under the base path.")
        return data

    print(f"Found {len(slab_dirs)} slab directories. Extracting energies...")
    
    for sd in sorted(slab_dirs):
        slab_name = sd.name

        # Match all POSCAR files following the naming pattern
        matches = list(sd.glob("POSCAR_opt_*eV.vasp"))

        if not matches:
            # Some directories may not be finished yet
            print(f"  [Skip] {slab_name}: no POSCAR_opt_*eV.vasp found")
            continue

        # If multiple exist, choose the lowest energy
        energies = []

        for f in matches:
            e = extract_energy_from_filename(f.name)
            if e is not None:
                energies.append(e)

        if not energies:
            print(f"  [Warning] {slab_name}: files found but energy could not be parsed")
            continue

        e_min = min(energies)

        data.append({
            "Slab": slab_name,
            "slab_eV": e_min
        })

        print(f"  {slab_name}: {e_min:.6f} eV  (minimum of {len(energies)} candidates)")

    return data


def create_slab_energy_table(base_path, output_file="slab_energy_table.csv"):
    """
    Generate slab energy table CSV with two columns [Slab, slab_eV]
    """
    print("=" * 60)
    print("Scanning directory structure and extracting slab energies...")
    print("=" * 60)

    rows = scan_slab_directory(base_path)

    if not rows:
        print(" No slab energy data extracted. Please check directory structure and filenames.")
        return None

    df = pd.DataFrame(rows).sort_values("Slab").reset_index(drop=True)

    df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n" + "=" * 60)
    print(f" Saved: {output_file}")
    print(f" Number of records: {len(df)}")
    print("=" * 60)

    print("\nPreview:")
    print(df.head(10).to_string(index=False))

    return df


def main():
    print("=" * 60)
    print("   Slab Energy Extraction Tool (no site hierarchy)")
    print("=" * 60)

    base_path = input("\nEnter base path (press Enter to use current directory): ").strip() or "."

    if not os.path.exists(base_path):
        print(f" Path does not exist: {base_path}")
        return

    try:
        df = create_slab_energy_table(base_path)

        if df is None:
            return

        # Summary statistics
        print("\n📈 Statistics:")
        print(f"  Number of slabs: {df['Slab'].nunique()}")
        print(f"  Energy range: {df['slab_eV'].min():.6f} ~ {df['slab_eV'].max():.6f} eV")
        print(f"  Average energy: {df['slab_eV'].mean():.6f} eV")

        # Most stable slab
        idx_min = df['slab_eV'].idxmin()

        print("\n Most stable slab:")
        print(f"  Slab: {df.loc[idx_min, 'Slab']}")
        print(f"  Energy: {df.loc[idx_min, 'slab_eV']:.6f} eV")

    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()

        print("\nPlease check:")
        print("  1) Whether the base path is correct")
        print("  2) Whether subdirectories correspond to slabs")
        print("  3) Whether filenames follow POSCAR_opt_XXXeV.vasp format")


if __name__ == "__main__":
    main()
