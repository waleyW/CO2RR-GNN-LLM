#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export adsorption site tables from POSCARs using pymatgen's AdsorbateSiteFinder
without adding adsorbates.

For each slab folder containing a POSCAR:
  - Detect adsorption sites via AdsorbateSiteFinder
  - Classify into top / bridge / hollow
  - Save per-slab CSV and append to a master CSV

Columns:
  Slab, site_global_id, site_type, coordination, x, y, z, fx, fy, fz,
  a, b, c, alpha, beta, gamma, n_sites_top, n_sites_bridge, n_sites_hollow

Usage:
  python export_adsorption_sites.py \
    --parent "/path/to/BinaryAlloys_Slab_Fixed" \
    --outdir "/path/to/output/PDS/SiteTables" \
    --distance 1.5 \
    --near-reduce 0.25 \
    --no-obtuse-hollow \
    --max-per-type 10
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def find_adsorption_sites_table(
    slab,
    distance: float = 1.5,
    near_reduce: float = 0.25,
    no_obtuse_hollow: bool = True,
    max_per_type: int = 10,
) -> pd.DataFrame:
    """
    Run AdsorbateSiteFinder and format results as a DataFrame.

    Returns DataFrame with columns:
      Slab (placeholder), site_global_id, site_type, coordination,
      x,y,z (Cartesian, Å), fx,fy,fz (fractional),
      a,b,c,alpha,beta,gamma, n_sites_top, n_sites_bridge, n_sites_hollow
    """
    asf = AdsorbateSiteFinder(slab)
    sites: Dict[str, List[np.ndarray]] = asf.find_adsorption_sites(
        distance=distance, near_reduce=near_reduce, no_obtuse_hollow=no_obtuse_hollow
    )

    # respect cap per type
    def cap(lst): return lst[:max_per_type] if max_per_type and max_per_type > 0 else lst

    top = cap(sites.get("top", []))
    bridge = cap(sites.get("bridge", []))
    hollow = cap(sites.get("hollow", []))

    # lattice info
    lat = slab.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

    rows = []
    gid = 0
    order = [("top", 1, top), ("bridge", 2, bridge), ("hollow", 3, hollow)]
    for s_type, coord, coords in order:
        for cart in coords:
            gid += 1
            frac = lat.get_fractional_coords(cart)
            rows.append({
                "Slab": "",  # fill later
                "site_global_id": gid,
                "site_type": s_type,
                "coordination": coord,      # 1=top, 2=bridge, 3=hollow
                "x": float(cart[0]), "y": float(cart[1]), "z": float(cart[2]),
                "fx": float(frac[0]), "fy": float(frac[1]), "fz": float(frac[2]),
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "n_sites_top": len(top),
                "n_sites_bridge": len(bridge),
                "n_sites_hollow": len(hollow),
            })

    df = pd.DataFrame(rows, columns=[
        "Slab", "site_global_id", "site_type", "coordination",
        "x","y","z","fx","fy","fz",
        "a","b","c","alpha","beta","gamma",
        "n_sites_top","n_sites_bridge","n_sites_hollow"
    ])
    return df


def process_parent(
    parent: Path,
    outdir: Path,
    distance: float,
    near_reduce: float,
    no_obtuse_hollow: bool,
    max_per_type: int,
    save_master: bool = True,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted([f for f in os.listdir(parent) if (parent / f / "POSCAR").exists()])
    if not subfolders:
        print(f"[WARN] No POSCAR found under: {parent}")
        return

    print(f"[INFO] Found {len(subfolders)} slab folders with POSCAR")

    master_rows = []
    ok, fail = 0, 0

    for idx, name in enumerate(subfolders, 1):
        poscar_path = parent / name / "POSCAR"
        print(f"[{idx}/{len(subfolders)}] {name} ... ", end="", flush=True)

        try:
            poscar = Poscar.from_file(str(poscar_path))
            slab = poscar.structure

            df = find_adsorption_sites_table(
                slab,
                distance=distance,
                near_reduce=near_reduce,
                no_obtuse_hollow=no_obtuse_hollow,
                max_per_type=max_per_type,
            )
            if df.empty:
                print("no sites")
                fail += 1
                continue

            df["Slab"] = name  # fill slab id
            csv_path = outdir / f"{name}__ads_sites.csv"
            df.to_csv(csv_path, index=False)

            master_rows.append(df)
            ok += 1
            print(f"{len(df)} rows -> {csv_path.name}")
        except Exception as e:
            fail += 1
            print(f"FAILED ({e})")

    if save_master and master_rows:
        master = pd.concat(master_rows, ignore_index=True)
        master_csv = outdir / "all_ads_sites_master.csv"
        master.to_csv(master_csv, index=False)
        print(f"[OK] Master table saved -> {master_csv} ({len(master)} rows)")

    print(f"\n[SUMMARY] success: {ok}, failed: {fail}, total: {len(subfolders)}")


def main():
    ap = argparse.ArgumentParser(description="Export adsorption site tables (top/bridge/hollow) from POSCARs.")
    ap.add_argument("--parent", required=True, help="Parent folder containing slab subfolders with POSCAR.")
    ap.add_argument("--outdir", required=True, help="Output directory for site tables.")
    ap.add_argument("--distance", type=float, default=1.5, help="ASF adsorption height (Å).")
    ap.add_argument("--near-reduce", type=float, default=0.25, help="ASF near_reduce.")
    ap.add_argument("--no-obtuse-hollow", action="store_true", help="ASF no_obtuse_hollow flag.")
    ap.add_argument("--max-per-type", type=int, default=10, help="Cap number of sites per type (0 = no cap).")
    args = ap.parse_args()

    parent = Path(args.parent).expanduser()
    outdir = Path(args.outdir).expanduser()

    process_parent(
        parent=parent,
        outdir=outdir,
        distance=args.distance,
        near_reduce=args.near_reduce,
        no_obtuse_hollow=args.no_obtuse_hollow,
        max_per_type=args.max_per_type,
    )


if __name__ == "__main__":
    main()
