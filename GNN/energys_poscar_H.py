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
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None

def scan_directory_structure(base_path):
    """
    扫描目录结构并提取H的能量数据
    """
    data = []
    base_path = Path(base_path)
    
    # 找到所有Slab_H开头的文件夹（或者包含H的文件夹）
    slab_folders = [f for f in base_path.iterdir() if f.is_dir() and 'H' in f.name]
    
    if not slab_folders:
        print("未找到包含 'H' 的文件夹，尝试查找所有Slab_开头的文件夹...")
        slab_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('Slab_')]
    
    for slab_folder in slab_folders:
        slab_name = slab_folder.name
        
        print(f"处理 {slab_name}...")
        
        # 在每个Slab文件夹下找到金属表面文件夹（Ag3Pt, Cu等）
        metal_folders = [f for f in slab_folder.iterdir() if f.is_dir()]
        
        for metal_folder in metal_folders:
            metal_name = metal_folder.name
            
            # 在金属文件夹下找到所有位点文件夹（如H_01, H_02等）
            site_folders = [f for f in metal_folder.iterdir() if f.is_dir()]
            
            for site_folder in site_folders:
                site_name = site_folder.name.split('_')[-1]  # 获取位点编号，如01, 02, 03
                
                # 查找POSCAR文件
                poscar_files = list(site_folder.glob('POSCAR_opt_*eV.vasp'))
                
                if poscar_files:
                    poscar_file = poscar_files[0]  # 取第一个匹配的文件
                    energy = extract_energy_from_filename(poscar_file.name)
                    
                    if energy is not None:
                        data.append({
                            'Slab': metal_name,
                            'Site': site_name,
                            'H_eV': energy
                        })
                        
                        print(f"  位点 {site_name}: {energy:.5f} eV")
    
    return data

def create_energy_table(base_path, output_file='H_energy_table.csv'):
    """
    创建H能量表格
    """
    print("=" * 60)
    print("开始扫描目录结构，提取 H 吸附能量...")
    print("=" * 60)
    
    data = scan_directory_structure(base_path)
    
    if not data:
        print("\n❌ 未找到任何数据！")
        print("请检查:")
        print("  1. 目录结构是否正确")
        print("  2. 是否存在包含 'H' 的文件夹")
        print("  3. 是否存在 POSCAR_opt_XXXeV.vasp 格式的文件")
        return None
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(data)
    df = df.sort_values(['Slab', 'Site']).reset_index(drop=True)
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False, float_format='%.5f')
    
    print(f"\n{'='*60}")
    print(f"✅ 数据已保存到 {output_file}")
    print(f"✅ 共处理 {len(df)} 条记录")
    print(f"{'='*60}")
    
    print("\n📊 数据预览:")
    print(df.to_string(index=False))
    
    return df

def main():
    print("=" * 60)
    print("   🔋 H 吸附能量提取工具")
    print("=" * 60)
    
    # 设置基础路径
    base_path = input("\n请输入基础路径 (直接回车使用当前目录): ").strip()
    if not base_path:
        base_path = "."
    
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return
    
    try:
        df = create_energy_table(base_path)
        
        if df is not None:
            # 显示统计信息
            print(f"\n📈 统计信息:")
            print(f"  不同 Slab 数量: {df['Slab'].nunique()}")
            print(f"  不同位点数量: {df['Site'].nunique()}")
            print(f"  H 吸附能量范围: {df['H_eV'].min():.5f} ~ {df['H_eV'].max():.5f} eV")
            print(f"  H 吸附能量平均: {df['H_eV'].mean():.5f} eV")
            
            # 找到最稳定的吸附位点
            min_idx = df['H_eV'].idxmin()
            print(f"\n⭐ 最稳定的吸附位点:")
            print(f"  Slab: {df.loc[min_idx, 'Slab']}")
            print(f"  Site: {df.loc[min_idx, 'Site']}")
            print(f"  能量: {df.loc[min_idx, 'H_eV']:.5f} eV")
            
            # 检查缺失数据
            missing_count = df['H_eV'].isnull().sum()
            if missing_count > 0:
                print(f"\n⚠️ 发现 {missing_count} 个缺失数据")
    
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("  1. 文件路径是否正确")
        print("  2. 目录结构是否符合预期")
        print("  3. 文件名格式是否为 POSCAR_opt_XXXeV.vasp")

if __name__ == "__main__":
    main()