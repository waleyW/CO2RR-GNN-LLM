# %% [markdown]
# # 二元合金Slab吸附结构生成工具
#
# 本工具用于批量生成吸附结构，保持原始POSCAR的选择性动力学约束，只对新添加的吸附分子原子设置为可移动。
#
# ## 功能特点
# - 保持原始slab的选择性动力学约束不变
# - 新添加的吸附分子原子设置为 `T T T` (完全可移动)
# - 支持多种吸附分子：H
# - 自动限制吸附位点数量以控制输出规模
# - 实时进度显示

# %% [markdown]
# ## 1. 导入必要的库

# %%
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
import os
import numpy as np
from pathlib import Path
import sys
import time

# %% [markdown]
# ## 2. 设置参数

# %%
# 设置路径参数
parent_folder = "/nesi/nobackup/uoa04335/WXY/GNN/Slab/BinaryAlloys_Slab_Fixed"
output_base = "/nesi/nobackup/uoa04335/WXY/CO2RRChat/GNN_OPT/PDS_H_2"

# 最大吸附位点数量限制
MAX_ADSORPTION_SITES = 10

print(f"输入文件夹: {parent_folder}", flush=True)
print(f"输出文件夹: {output_base}", flush=True)
print(f"最大吸附位点数: {MAX_ADSORPTION_SITES}", flush=True)

# 检查路径
print(f"\n检查输入路径是否存在: {os.path.exists(parent_folder)}", flush=True)
if not os.path.exists(parent_folder):
    print(f"错误: 输入路径不存在!", flush=True)
    sys.exit(1)

# %% [markdown]
# ## 3. 定义吸附分子

# %%
# 定义吸附分子
adsorbates = {
    "H": Molecule("H", [[0.0, 0.0, 0.0]])
}
# 选择要生成的分子
molecules_to_generate = ["H"]

print("\n定义的吸附分子:", flush=True)
for name, mol in adsorbates.items():
    print(f"  {name}: {len(mol)} 个原子 ({mol.formula})", flush=True)
print(f"\n将要生成的分子: {', '.join(molecules_to_generate)}", flush=True)

# %% [markdown]
# ## 4. 定义核心函数

# %%
def create_selective_dynamics_with_adsorbate(original_poscar, ads_structure):
    """
    创建保持原始约束的选择性动力学标记
    - 保持原始slab的选择性动力学约束不变
    - 新添加的吸附分子原子设置为可移动 (True, True, True)
    """
    # 获取原始结构的原子数
    original_natoms = len(original_poscar.structure)
    ads_natoms = len(ads_structure)

    selective_dynamics = []

    # 1. 处理原始slab的原子（保持原来的约束）
    if hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None:
        for i in range(original_natoms):
            selective_dynamics.append(original_poscar.selective_dynamics[i])
    else:
        for i in range(original_natoms):
            selective_dynamics.append([False, False, False])

    # 2. 新添加的吸附分子原子（设置为可移动）
    adsorbate_atoms = ads_natoms - original_natoms
    for i in range(adsorbate_atoms):
        selective_dynamics.append([True, True, True])

    return selective_dynamics

# %%
def process_single_folder(subfolder_path, subfolder_name):
    """
    处理单个文件夹
    """
    # 检查POSCAR是否存在
    poscar_path = os.path.join(subfolder_path, "POSCAR")
    if not os.path.exists(poscar_path):
        return False, {'folder_name': subfolder_name, 'status': 'no_poscar'}

    folder_stats = {
        'folder_name': subfolder_name,
        'status': 'failed',
        'slab_atoms': 0,
        'composition': '',
        'has_selective': False,
        'adsorption_sites': 0,
        'structures_generated': {}
    }

    try:
        # 读取原始POSCAR对象（保留选择性动力学信息）
        original_poscar = Poscar.from_file(poscar_path)
        slab = original_poscar.structure

        # 更新统计信息
        folder_stats['slab_atoms'] = len(slab)
        folder_stats['composition'] = str(slab.composition)
        folder_stats['has_selective'] = hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None

        # 创建吸附位点查找器
        asf = AdsorbateSiteFinder(slab)

        # 查找吸附位点
        ads_sites = asf.find_adsorption_sites(
            distance=1.5,
            near_reduce=0.25,
            no_obtuse_hollow=True
        )

        total_sites = len(ads_sites['all'])
        folder_stats['adsorption_sites'] = total_sites

        # 为每个分子生成吸附结构
        for mol_name in molecules_to_generate:
            try:
                ads_structs = asf.generate_adsorption_structures(
                    adsorbates[mol_name],
                    repeat=[1, 1, 1],
                    min_lw=10.0,
                    find_args={"distance": 1.5, "near_reduce": 0.25}
                )

                if len(ads_structs) > MAX_ADSORPTION_SITES:
                    ads_structs = ads_structs[:MAX_ADSORPTION_SITES]

                folder_stats['structures_generated'][mol_name] = len(ads_structs)

                # 保存结构
                for i, ads_struct in enumerate(ads_structs):
                    folder_path = os.path.join(
                        output_base, f"Slab_{mol_name}", subfolder_name, f"{mol_name}_{i+1:02d}"
                    )
                    os.makedirs(folder_path, exist_ok=True)

                    # 创建保持原始约束的选择性动力学标记
                    selective_dynamics = create_selective_dynamics_with_adsorbate(original_poscar, ads_struct)

                    # 保存POSCAR（带选择性动力学）
                    poscar = Poscar(ads_struct, selective_dynamics=selective_dynamics)
                    poscar.write_file(os.path.join(folder_path, "POSCAR"))

            except Exception as e:
                folder_stats['structures_generated'][mol_name] = 0
                continue

        folder_stats['status'] = 'success'
        return True, folder_stats

    except Exception as e:
        folder_stats['error'] = str(e)
        return False, folder_stats

# %% [markdown]
# ## 5. 主处理函数（带进度显示）

# %%
def batch_process_folders():
    """
    批量处理所有文件夹
    """
    print("\n" + "=" * 70, flush=True)
    print("二元合金Slab吸附结构生成工具 - 保持原始约束版本", flush=True)
    print("=" * 70, flush=True)
    print(f"输入文件夹: {parent_folder}", flush=True)
    print(f"输出文件夹: {output_base}", flush=True)
    print(f"处理分子: {', '.join(molecules_to_generate)}", flush=True)
    print(f"最大吸附位点数: {MAX_ADSORPTION_SITES}", flush=True)
    print("约束策略: 保持原始slab约束，新添加的吸附分子原子可移动", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(parent_folder):
        print(f"错误: 输入文件夹不存在: {parent_folder}", flush=True)
        return None

    os.makedirs(output_base, exist_ok=True)

    total_folders = 0
    processed_folders = 0
    failed_folders = 0
    no_poscar_folders = 0
    total_structures = 0
    all_stats = []

    print("\n正在获取文件夹列表...", flush=True)
    subfolders = sorted([f for f in os.listdir(parent_folder)
                        if os.path.isdir(os.path.join(parent_folder, f))])

    total = len(subfolders)
    print(f"找到 {total} 个子文件夹", flush=True)
    print("-" * 70, flush=True)
    
    start_time = time.time()

    for idx, subfolder in enumerate(subfolders, 1):
        total_folders += 1
        subfolder_path = os.path.join(parent_folder, subfolder)

        success, folder_stats = process_single_folder(subfolder_path, subfolder)
        all_stats.append(folder_stats)

        if success:
            processed_folders += 1
            structures_count = sum(folder_stats['structures_generated'].values())
            total_structures += structures_count
        elif folder_stats['status'] == 'no_poscar':
            no_poscar_folders += 1
        else:
            failed_folders += 1

        # 每10个文件夹显示一次进度
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total - idx)
            
            print(f"进度: {idx}/{total} ({idx*100//total}%) | "
                  f"成功: {processed_folders} | 失败: {failed_folders} | 无POSCAR: {no_poscar_folders} | "
                  f"已生成结构: {total_structures} | "
                  f"用时: {elapsed/60:.1f}分 | 预计剩余: {eta/60:.1f}分", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("处理完成！", flush=True)
    print(f"总文件夹数: {total_folders}", flush=True)
    print(f"成功处理: {processed_folders}", flush=True)
    print(f"处理失败: {failed_folders}", flush=True)
    print(f"无POSCAR: {no_poscar_folders}", flush=True)
    print(f"总生成结构数: {total_structures}", flush=True)
    print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟", flush=True)
    print(f"处理的分子: {', '.join(molecules_to_generate)}", flush=True)

    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'failed_folders': failed_folders,
        'no_poscar_folders': no_poscar_folders,
        'total_structures': total_structures,
        'folder_stats': all_stats
    }

# %% [markdown]
# ## 6. 直接执行处理（跳过统计阶段）

# %%
# 直接运行批处理（跳过统计以加快速度）
print("\n开始批量处理（跳过统计阶段以加快速度）...", flush=True)
results = batch_process_folders()

# %% [markdown]
# ## 7. 结果分析

# %%
if results is not None:
    print("\n" + "=" * 70, flush=True)
    print("处理结果详细统计:", flush=True)
    print("=" * 70, flush=True)
    print(f"成功处理的文件夹: {results['processed_folders']}", flush=True)
    print(f"失败的文件夹: {results['failed_folders']}", flush=True)
    print(f"无POSCAR的文件夹: {results['no_poscar_folders']}", flush=True)
    print(f"总生成结构数: {results['total_structures']}", flush=True)

    # 统计每个分子生成的结构数量
    print("\n各分子生成的结构统计:", flush=True)
    for mol_name in molecules_to_generate:
        total_struct = sum([stats['structures_generated'].get(mol_name, 0)
                           for stats in results['folder_stats']
                           if stats['status'] == 'success'])
        print(f"  {mol_name}: {total_struct} 个结构", flush=True)

    # 显示前几个成功处理的文件夹
    successful_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'success']
    if successful_folders:
        print(f"\n前5个成功处理的文件夹:", flush=True)
        for i, stats in enumerate(successful_folders[:5]):
            struct_count = sum(stats['structures_generated'].values())
            print(f"  {i+1}. {stats['folder_name']}: {stats['slab_atoms']}原子, "
                  f"{stats['adsorption_sites']}吸附位点, 生成{struct_count}个结构", flush=True)

    # 显示失败的文件夹
    failed_folders = [stats for stats in results['folder_stats'] 
                     if stats['status'] == 'failed']
    if failed_folders:
        print(f"\n处理失败的文件夹 ({len(failed_folders)}个):", flush=True)
        for stats in failed_folders[:5]:
            error_msg = stats.get('error', '未知错误')
            print(f"  - {stats['folder_name']}: {error_msg}", flush=True)
        if len(failed_folders) > 5:
            print(f"  ... 还有 {len(failed_folders)-5} 个失败文件夹", flush=True)
else:
    print("没有处理结果可显示", flush=True)

# %% [markdown]
# ## 8. 输出目录结构示意

# %%
if results is not None:
    print(f"\n输出目录结构:", flush=True)
    print(f"{output_base}/", flush=True)
    for mol_name in molecules_to_generate:
        print(f"├── Slab_{mol_name}/", flush=True)
        print(f"│   ├── [子文件夹名]/", flush=True)
        print(f"│   │   ├── {mol_name}_01/POSCAR", flush=True)
        print(f"│   │   ├── {mol_name}_02/POSCAR", flush=True)
        print(f"│   │   └── ... (最多{MAX_ADSORPTION_SITES}个)", flush=True)
        print(f"│   └── ...", flush=True)

    print(f"\n约束规则:", flush=True)
    print(f"• 原始slab原子: 保持原POSCAR中的选择性动力学约束", flush=True)
    print(f"• 新添加的吸附分子原子: T T T (完全可移动)", flush=True)

# %% [markdown]
# ## 9. 验证生成的结构

# %%
def verify_generated_structure():
    """
    验证生成的结构是否正确
    """
    for mol_name in molecules_to_generate:
        mol_folder = os.path.join(output_base, f"Slab_{mol_name}")
        if os.path.exists(mol_folder):
            subfolders = [f for f in os.listdir(mol_folder) if os.path.isdir(os.path.join(mol_folder, f))]
            if subfolders:
                first_subfolder = subfolders[0]
                struct_folders = [f for f in os.listdir(os.path.join(mol_folder, first_subfolder))
                                 if os.path.isdir(os.path.join(mol_folder, first_subfolder, f))]
                if struct_folders:
                    sample_path = os.path.join(mol_folder, first_subfolder, struct_folders[0], "POSCAR")

                    if os.path.exists(sample_path):
                        try:
                            poscar = Poscar.from_file(sample_path)
                            structure = poscar.structure

                            print(f"\n验证结构: {sample_path}", flush=True)
                            print(f"  总原子数: {len(structure)}", flush=True)
                            print(f"  化学式: {structure.composition}", flush=True)

                            if hasattr(poscar, 'selective_dynamics') and poscar.selective_dynamics is not None:
                                moveable = sum(1 for sd in poscar.selective_dynamics if any(sd))
                                fixed = len(structure) - moveable
                                print(f"  选择性动力学: 可移动 {moveable} 个，固定 {fixed} 个", flush=True)

                                # 显示最后几个原子的约束（应该是吸附分子，应该是T T T）
                                print(f"  最后3个原子的约束 (应为吸附分子原子):", flush=True)
                                for i, sd in enumerate(poscar.selective_dynamics[-3:], len(poscar.selective_dynamics)-2):
                                    print(f"    原子 {i}: {sd}", flush=True)
                            else:
                                print(f"  无选择性动力学信息", flush=True)

                            return True
                        except Exception as e:
                            print(f"  验证失败: {e}", flush=True)
                            return False

    print("未找到可验证的结构", flush=True)
    return False

# 验证结构
if results is not None and results['processed_folders'] > 0:
    print("\n" + "=" * 70, flush=True)
    print("结构验证", flush=True)
    print("=" * 70, flush=True)
    verify_generated_structure()

print("\n脚本执行完毕！", flush=True)