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
    扫描目录结构并提取能量数据
    """
    data = []
    base_path = Path(base_path)
    
    # 找到所有Slab_开头的文件夹
    slab_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('Slab_')]
    
    for slab_folder in slab_folders:
        slab_name = slab_folder.name
        adsorbate = slab_name.replace('Slab_', '')  # 获取吸附物种类型
        
        print(f"处理 {slab_name}...")
        
        # 在每个Slab文件夹下找到Ag3Pt_mp开头的文件夹
        ag3pt_folders = [f for f in slab_folder.iterdir() if f.is_dir() and f.name.startswith('Ag3')]
        
        for ag3pt_folder in ag3pt_folders:
            # 在Ag3Pt文件夹下找到所有位点文件夹（如CO_01, CO_02等）
            site_folders = [f for f in ag3pt_folder.iterdir() if f.is_dir() and '_' in f.name]
            
            for site_folder in site_folders:
                site_name = site_folder.name.split('_')[-1]  # 获取位点编号，如01, 02, 03
                
                # 查找POSCAR文件
                poscar_files = list(site_folder.glob('POSCAR_opt_*eV.vasp'))
                
                if poscar_files:
                    poscar_file = poscar_files[0]  # 取第一个匹配的文件
                    energy = extract_energy_from_filename(poscar_file.name)
                    
                    if energy is not None:
                        # 检查是否已经存在相同slab和位点的记录
                        existing_row = None
                        for row in data:
                            if row['Slab'] == ag3pt_folder.name and row['Site'] == site_name:
                                existing_row = row
                                break
                        
                        if existing_row is None:
                            # 创建新记录
                            new_row = {
                                'Slab': ag3pt_folder.name,
                                'Site': site_name,
                                'OCHO_eV': None,
                                'COOH_eV': None,
                                'CO_eV': None,
                                'COCO_eV': None
                            }
                            data.append(new_row)
                            existing_row = new_row
                        
                        # 根据吸附物类型填入相应列
                        if adsorbate == 'OCHO':
                            existing_row['OCHO_eV'] = energy
                        elif adsorbate == 'COOH':
                            existing_row['COOH_eV'] = energy
                        elif adsorbate == 'CO':
                            existing_row['CO_eV'] = energy
                        elif adsorbate == 'COCO':
                            existing_row['COCO_eV'] = energy
                
                print(f"  处理位点: {site_folder.name}, 吸附物: {adsorbate}")
    
    return data

def create_energy_table(base_path, output_file='energy_table.csv'):
    """
    创建能量表格
    """
    print("开始扫描目录结构...")
    data = scan_directory_structure(base_path)
    
    if not data:
        print("未找到任何数据！")
        return None
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(data)
    df = df.sort_values(['Slab', 'Site']).reset_index(drop=True)
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False, float_format='%.5f')
    
    print(f"\n数据已保存到 {output_file}")
    print(f"共处理 {len(df)} 条记录")
    print("\n数据预览:")
    print(df.head(10))
    
    return df

def main():
    # 设置基础路径（请根据实际情况修改）
    base_path = "."  # 当前目录，您可以修改为实际路径
    
    # 如果您想指定特定路径，请取消注释并修改下面的行：
    # base_path = "/path/to/your/PDS_OPT_GPU"
    
    try:
        df = create_energy_table(base_path)
        
        if df is not None:
            # 显示统计信息
            print(f"\n统计信息:")
            print(f"不同Slab数量: {df['Slab'].nunique()}")
            print(f"不同位点数量: {df['Site'].nunique()}")
            print(f"OCHO数据点: {df['OCHO_eV'].notna().sum()}")
            print(f"COOH数据点: {df['COOH_eV'].notna().sum()}")
            print(f"CO数据点: {df['CO_eV'].notna().sum()}")
            print(f"COCO数据点: {df['COCO_eV'].notna().sum()}")
            
            # 检查缺失数据
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                print(f"\n缺失数据统计:")
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        print(f"{col}: {missing_count} 个缺失值")
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print("请检查文件路径和目录结构是否正确")

if __name__ == "__main__":
    main()