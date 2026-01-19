# %% [markdown]
# # 二元合金Slab吸附结构生成工具
#
# 本工具用于批量生成吸附结构，保持原始POSCAR的选择性动力学约束，只对新添加的吸附分子原子设置为可移动。
#
# ## 功能特点
# - 保持原始slab的选择性动力学约束不变
# - 新添加的吸附分子原子设置为 `T T T` (完全可移动)
# - 支持多种吸附分子：CHC, CHCOHH, CHCOH
# - 自动限制吸附位点数量以控制输出规模

# %% [markdown]
# ## 1. 导入必要的库

# %%
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
import os
import numpy as np
from pathlib import Path

# %% [markdown]
# ## 2. 设置参数

# %%
# 设置路径参数
parent_folder = "/nesi/nobackup/uoa04335/WXY/GNN/Slab/BinaryAlloys_Slab_Fixed"
output_base = "/nesi/nobackup/uoa04335/WXY/CO2RRChat/GNN_OPT/PDS"

# 最大吸附位点数量限制
MAX_ADSORPTION_SITES = 10

print(f"输入文件夹: {parent_folder}")
print(f"输出文件夹: {output_base}")
print(f"最大吸附位点数: {MAX_ADSORPTION_SITES}")

# %% [markdown]
# ## 3. 定义吸附分子

# %%
# 定义吸附分子（使用验证过的坐标）
adsorbates = {
    "H": Molecule("H", [[0.0, 0.0, 0.0]])
}
# 选择要生成的分子
molecules_to_generate = ["H"]

print("定义的吸附分子:")
for name, mol in adsorbates.items():
    print(f"  {name}: {len(mol)} 个原子 ({mol.formula})")
print(f"\n将要生成的分子: {', '.join(molecules_to_generate)}")

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

    print(f"        原始结构原子数: {original_natoms}")
    print(f"        吸附后原子数: {ads_natoms}")
    print(f"        新增原子数: {ads_natoms - original_natoms}")

    selective_dynamics = []

    # 1. 处理原始slab的原子（保持原来的约束）
    if hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None:
        print(f"        使用原始POSCAR的选择性动力学约束")
        for i in range(original_natoms):
            selective_dynamics.append(original_poscar.selective_dynamics[i])
    else:
        print(f"        原始POSCAR无选择性动力学信息，默认固定所有slab原子")
        for i in range(original_natoms):
            selective_dynamics.append([False, False, False])

    # 2. 新添加的吸附分子原子（设置为可移动）
    adsorbate_atoms = ads_natoms - original_natoms
    for i in range(adsorbate_atoms):
        selective_dynamics.append([True, True, True])

    print(f"        最终约束设置: slab原子保持原约束，{adsorbate_atoms}个吸附原子可移动")
    return selective_dynamics

# %%
def process_single_folder(subfolder_path, subfolder_name):
    """
    处理单个文件夹
    """
    # 检查POSCAR是否存在
    poscar_path = os.path.join(subfolder_path, "POSCAR")
    if not os.path.exists(poscar_path):
        print(f"跳过 {subfolder_name}: 没有POSCAR文件")
        return False, {}

    print(f"\n处理文件夹: {subfolder_name}")

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

        # 检查是否包含选择性动力学信息
        has_selective = hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None
        folder_stats['has_selective'] = has_selective

        print(f"  Slab原子数: {len(slab)}")
        print(f"  化学式: {slab.composition}")
        print(f"  包含选择性动力学: {'是' if has_selective else '否'}")

        if has_selective:
            original_moveable = sum(1 for sd in original_poscar.selective_dynamics if any(sd))
            original_fixed = len(slab) - original_moveable
            print(f"  原始约束: 可移动 {original_moveable} 个，固定 {original_fixed} 个")

        # 创建吸附位点查找器
        asf = AdsorbateSiteFinder(slab)

        # 查找吸附位点
        ads_sites = asf.find_adsorption_sites(
            distance=1.5,        # 吸附高度
            near_reduce=0.25,    # 减少相近位点
            no_obtuse_hollow=True
        )

        total_sites = len(ads_sites['all'])
        folder_stats['adsorption_sites'] = total_sites
        print(f"  找到吸附位点: {total_sites} 个")

        if total_sites > MAX_ADSORPTION_SITES:
            print(f"  限制为前 {MAX_ADSORPTION_SITES} 个位点")

        # 为每个分子生成吸附结构
        for mol_name in molecules_to_generate:
            print(f"  生成 {mol_name} 吸附结构...")

            try:
                ads_structs = asf.generate_adsorption_structures(
                    adsorbates[mol_name],
                    repeat=[1, 1, 1],
                    min_lw=10.0,
                    find_args={"distance": 1.5, "near_reduce": 0.25}
                )

                if len(ads_structs) > MAX_ADSORPTION_SITES:
                    ads_structs = ads_structs[:MAX_ADSORPTION_SITES]

                print(f"    生成了 {len(ads_structs)} 个结构")
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

                    # 统计信息
                    total_atoms = len(ads_struct)
                    moveable_atoms = sum(1 for sd in selective_dynamics if any(sd))
                    fixed_atoms = total_atoms - moveable_atoms

                    print(f"      保存: {folder_path}/POSCAR")
                    print(f"        总原子数: {total_atoms}, 可移动: {moveable_atoms}, 固定: {fixed_atoms}")

            except Exception as e:
                print(f"    生成 {mol_name} 时出错: {str(e)}")
                folder_stats['structures_generated'][mol_name] = 0
                continue

        folder_stats['status'] = 'success'
        return True, folder_stats

    except Exception as e:
        print(f"  处理 {subfolder_name} 时出错: {str(e)}")
        return False, folder_stats

# %% [markdown]
# ## 5. 获取文件夹统计信息

# %%
def get_folder_statistics():
    """
    获取文件夹统计信息
    """
    if not os.path.exists(parent_folder):
        print(f"输入文件夹不存在: {parent_folder}")
        return None

    subfolders = [f for f in os.listdir(parent_folder)
                  if os.path.isdir(os.path.join(parent_folder, f))]

    print(f"输入文件夹统计:")
    print(f"  总子文件夹数: {len(subfolders)}")

    poscar_count = 0
    selective_count = 0
    for subfolder in subfolders:
        poscar_path = os.path.join(parent_folder, subfolder, "POSCAR")
        if os.path.exists(poscar_path):
            poscar_count += 1
            try:
                poscar = Poscar.from_file(poscar_path)
                if hasattr(poscar, 'selective_dynamics') and poscar.selective_dynamics is not None:
                    selective_count += 1
            except:
                pass

    print(f"  包含POSCAR的文件夹: {poscar_count}")
    print(f"  包含选择性动力学的POSCAR: {selective_count}")
    print(f"  预计生成结构总数: {poscar_count * len(molecules_to_generate) * MAX_ADSORPTION_SITES} (最大)")

    return {
        'total_folders': len(subfolders),
        'poscar_folders': poscar_count,
        'selective_folders': selective_count
    }

# 获取统计信息
stats = get_folder_statistics()

# %% [markdown]
# ## 6. 主处理函数

# %%
def batch_process_folders():
    """
    批量处理所有文件夹
    """
    print("二元合金Slab吸附结构生成工具 - 保持原始约束版本")
    print("=" * 70)
    print(f"输入文件夹: {parent_folder}")
    print(f"输出文件夹: {output_base}")
    print(f"处理分子: {', '.join(molecules_to_generate)}")
    print(f"最大吸附位点数: {MAX_ADSORPTION_SITES}")
    print("约束策略: 保持原始slab约束，新添加的吸附分子原子可移动")
    print("=" * 70)

    if not os.path.exists(parent_folder):
        print(f"错误: 输入文件夹不存在: {parent_folder}")
        return None

    os.makedirs(output_base, exist_ok=True)

    total_folders = 0
    processed_folders = 0
    failed_folders = 0
    all_stats = []

    subfolders = sorted([f for f in os.listdir(parent_folder)
                        if os.path.isdir(os.path.join(parent_folder, f))])

    print(f"找到 {len(subfolders)} 个子文件夹")
    print("-" * 70)

    for subfolder in subfolders:
        total_folders += 1
        subfolder_path = os.path.join(parent_folder, subfolder)

        success, folder_stats = process_single_folder(subfolder_path, subfolder)
        all_stats.append(folder_stats)

        if success:
            processed_folders += 1
        else:
            failed_folders += 1

    print("\n" + "=" * 70)
    print("处理完成！")
    print(f"总文件夹数: {total_folders}")
    print(f"成功处理: {processed_folders}")
    print(f"处理失败: {failed_folders}")
    print(f"处理的分子: {', '.join(molecules_to_generate)}")

    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'failed_folders': failed_folders,
        'folder_stats': all_stats
    }

# %% [markdown]
# ## 7. 直接执行处理（无交互）

# %%
# 直接运行批处理
results = batch_process_folders()

# %% [markdown]
# ## 8. 结果分析

# %%
if results is not None:
    print("\n处理结果详细统计:")
    print(f"成功处理的文件夹: {results['processed_folders']}")
    print(f"失败的文件夹: {results['failed_folders']}")

    # 统计每个分子生成的结构数量
    print("\n各分子生成的结构统计:")
    for mol_name in molecules_to_generate:
        total_structures = sum([stats['structures_generated'].get(mol_name, 0)
                               for stats in results['folder_stats']
                               if stats['status'] == 'success'])
        print(f"  {mol_name}: {total_structures} 个结构")

    # 显示前几个成功处理的文件夹
    successful_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'success']
    if successful_folders:
        print(f"\n前5个成功处理的文件夹:")
        for i, stats in enumerate(successful_folders[:5]):
            print(f"  {i+1}. {stats['folder_name']}: {stats['slab_atoms']}原子, {stats['adsorption_sites']}吸附位点")

    # 显示失败的文件夹
    failed_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'failed']
    if failed_folders:
        print(f"\n处理失败的文件夹 ({len(failed_folders)}个):")
        for stats in failed_folders[:5]:  # 只显示前5个
            print(f"  - {stats['folder_name']}")
else:
    print("没有处理结果可显示")

# %% [markdown]
# ## 9. 输出目录结构示意

# %%
if results is not None:
    print(f"\n输出目录结构:")
    print(f"{output_base}/")
    for mol_name in molecules_to_generate:
        print(f"├── Slab_{mol_name}/")
        print(f"│   ├── [子文件夹名]/")
        print(f"│   │   ├── {mol_name}_01/POSCAR")
        print(f"│   │   ├── {mol_name}_02/POSCAR")
        print(f"│   │   └── ... (最多{MAX_ADSORPTION_SITES}个)")
        print(f"│   └── ...")

    print(f"\n约束规则:")
    print(f"• 原始slab原子: 保持原POSCAR中的选择性动力学约束")
    print(f"• 新添加的吸附分子原子: T T T (完全可移动)")

# %% [markdown]
# ## 10. 验证生成的结构（可选）

# %%
def verify_generated_structure(sample_folder):
    """
    验证生成的结构是否正确
    """
    # 找到第一个生成的结构进行验证
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

                            print(f"\n验证结构: {sample_path}")
                            print(f"  总原子数: {len(structure)}")
                            print(f"  化学式: {structure.composition}")

                            if hasattr(poscar, 'selective_dynamics') and poscar.selective_dynamics is not None:
                                moveable = sum(1 for sd in poscar.selective_dynamics if any(sd))
                                fixed = len(structure) - moveable
                                print(f"  选择性动力学: 可移动 {moveable} 个，固定 {fixed} 个")

                                # 显示最后几个原子的约束（应该是吸附分子，应该是T T T）
                                print(f"  最后5个原子的约束:")
                                for i, sd in enumerate(poscar.selective_dynamics[-5:], len(poscar.selective_dynamics)-4):
                                    print(f"    原子 {i}: {sd}")
                            else:
                                print(f"  无选择性动力学信息")

                            return True
                        except Exception as e:
                            print(f"  验证失败: {e}")
                            return False

    print("未找到可验证的结构")
    return False

# 有结果则做一次示例验证
if results is not None and results['processed_folders'] > 0:
    print("\n=== 结构验证 ===")
    verify_generated_structure(output_base)
