#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from pathlib import Path

def extract_energy_from_filename(filename):
    """
    从文件名中提取能量值
    例如：POSCAR_opt_-0.23305eV.vasp -> -0.23305
    """
    pattern = r'POSCAR_opt_([+-]?\d+\.?\d*)eV\.vasp'
    m = re.search(pattern, filename)
    return float(m.group(1)) if m else None

def scan_slab_directory(base_path):
    """
    扫描“每个 Slab 目录下直接有 POSCAR_opt_*eV.vasp”的结构，提取 slab 能量。
    目录形如：
      base_path/
        Ag3Pt_mp-12065_proc1/
          POSCAR_opt_-0.76018eV.vasp
        Ag3Sn_mp-611_proc1/
          POSCAR_opt_-0.12345eV.vasp
        ...
    返回 list[dict]: [{'Slab': slab_name, 'slab_eV': energy}, ...]
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"路径不存在：{base}")

    data = []
    # 只看第一层子目录，认为每个子目录是一个 slab
    slab_dirs = [d for d in base.iterdir() if d.is_dir()]
    if not slab_dirs:
        print("未在基础路径下找到任何子目录（slab）。")
        return data

    print(f"发现 {len(slab_dirs)} 个 slab 目录。开始提取能量…")
    for sd in sorted(slab_dirs):
        slab_name = sd.name
        # 匹配该目录下所有符合命名的 POSCAR 文件
        matches = list(sd.glob("POSCAR_opt_*eV.vasp"))
        if not matches:
            # 有的目录可能还没算完；跳过但提示
            print(f"  [跳过] {slab_name}: 未找到 POSCAR_opt_*eV.vasp")
            continue

        # 若存在多个，取能量最低者
        energies = []
        for f in matches:
            e = extract_energy_from_filename(f.name)
            if e is not None:
                energies.append(e)

        if not energies:
            print(f"  [警告] {slab_name}: 找到文件但无法解析能量（命名不合规）")
            continue

        e_min = min(energies)
        data.append({"Slab": slab_name, "slab_eV": e_min})
        print(f"  {slab_name}: {e_min:.6f} eV  (从 {len(energies)} 个候选中取最小)")

    return data

def create_slab_energy_table(base_path, output_file="slab_energy_table.csv"):
    """
    生成 slab 能量表 CSV：两列 [Slab, slab_eV]
    """
    print("=" * 60)
    print("开始扫描目录结构，提取 slab 能量…")
    print("=" * 60)

    rows = scan_slab_directory(base_path)
    if not rows:
        print("❌ 未得到任何 slab 能量数据。请检查目录与文件命名。")
        return None

    df = pd.DataFrame(rows).sort_values("Slab").reset_index(drop=True)
    df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n" + "=" * 60)
    print(f"✅ 已保存：{output_file}")
    print(f"✅ 记录数：{len(df)}")
    print("=" * 60)
    print("\n📊 预览：")
    print(df.head(10).to_string(index=False))
    return df

def main():
    print("=" * 60)
    print("   🧱 Slab 能量提取工具（无位点层级）")
    print("=" * 60)

    base_path = input("\n请输入基础路径 (直接回车使用当前目录): ").strip() or "."
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return

    try:
        df = create_slab_energy_table(base_path)
        if df is None:
            return

        # 一些汇总信息
        print("\n📈 统计：")
        print(f"  Slab 数量：{df['Slab'].nunique()}")
        print(f"  能量范围：{df['slab_eV'].min():.6f} ~ {df['slab_eV'].max():.6f} eV")
        print(f"  能量均值：{df['slab_eV'].mean():.6f} eV")

        # 最稳定 slab
        idx_min = df['slab_eV'].idxmin()
        print("\n⭐ 最稳定 slab：")
        print(f"  Slab: {df.loc[idx_min, 'Slab']}")
        print(f"  能量: {df.loc[idx_min, 'slab_eV']:.6f} eV")

    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()
        print("\n请检查：")
        print("  1) 基础路径是否正确")
        print("  2) 子目录是否为各个 slab")
        print("  3) 文件名是否形如 POSCAR_opt_XXXeV.vasp")

if __name__ == "__main__":
    main()
