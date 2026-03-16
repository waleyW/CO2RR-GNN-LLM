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
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None


def scan_directory_structure(base_path):
    """
    Scan the directory structure and extract energy data
    """
    data = []
    base_path = Path(base_path)
    
    # Find all folders starting with 'Slab_'
    slab_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('Slab_')]
    
    for slab_folder in slab_folders:
        slab_name = slab_folder.name
        adsorbate = slab_name.replace('Slab_', '')  # Get adsorbate species
        
        print(f"Processing {slab_name}...")
        
        # Find folders starting with Ag3Pt_mp under each Slab folder
        ag3pt_folders = [f for f in slab_folder.iterdir() if f.is_dir() and f.name.startswith('Ag3')]
        
        for ag3pt_folder in ag3pt_folders:
            # Find all site folders under Ag3Pt (e.g., CO_01, CO_02)
            site_folders = [f for f in ag3pt_folder.iterdir() if f.is_dir() and '_' in f.name]
            
            for site_folder in site_folders:
                site_name = site_folder.name.split('_')[-1]  # Extract site index (e.g., 01, 02, 03)
                
                # Search for POSCAR files
                poscar_files = list(site_folder.glob('POSCAR_opt_*eV.vasp'))
                
                if poscar_files:
                    poscar_file = poscar_files[0]  # Take the first matching file
                    energy = extract_energy_from_filename(poscar_file.name)
                    
                    if energy is not None:
                        # Check whether a record with the same slab and site already exists
                        existing_row = None
                        for row in data:
                            if row['Slab'] == ag3pt_folder.name and row['Site'] == site_name:
                                existing_row = row
                                break
                        
                        if existing_row is None:
                            # Create a new record
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
                        
                        # Fill the corresponding column according to adsorbate type
                        if adsorbate == 'OCHO':
                            existing_row['OCHO_eV'] = energy
                        elif adsorbate == 'COOH':
                            existing_row['COOH_eV'] = energy
                        elif adsorbate == 'CO':
                            existing_row['CO_eV'] = energy
                        elif adsorbate == 'COCO':
                            existing_row['COCO_eV'] = energy
                
                print(f"  Processing site: {site_folder.name}, adsorbate: {adsorbate}")
    
    return data


def create_energy_table(base_path, output_file='energy_table.csv'):
    """
    Create the energy table
    """
    print("Starting directory scan...")
    data = scan_directory_structure(base_path)
    
    if not data:
        print("No data found!")
        return None
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(data)
    df = df.sort_values(['Slab', 'Site']).reset_index(drop=True)
    
    # Save to CSV file
    df.to_csv(output_file, index=False, float_format='%.5f')
    
    print(f"\nData saved to {output_file}")
    print(f"Total records processed: {len(df)}")
    print("\nData preview:")
    print(df.head(10))
    
    return df


def main():
    # Set base path (modify according to your actual situation)
    base_path = "."  # Current directory; you may change this to the actual path
    
    # If you want to specify a specific path, uncomment and modify the line below:
    # base_path = "/path/to/your/PDS_OPT_GPU"
    
    try:
        df = create_energy_table(base_path)
        
        if df is not None:
            # Display statistics
            print(f"\nStatistics:")
            print(f"Number of different slabs: {df['Slab'].nunique()}")
            print(f"Number of different sites: {df['Site'].nunique()}")
            print(f"OCHO data points: {df['OCHO_eV'].notna().sum()}")
            print(f"COOH data points: {df['COOH_eV'].notna().sum()}")
            print(f"CO data points: {df['CO_eV'].notna().sum()}")
            print(f"COCO data points: {df['COCO_eV'].notna().sum()}")
            
            # Check missing data
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                print(f"\nMissing data statistics:")
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        print(f"{col}: {missing_count} missing values")
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("Please check whether the file path and directory structure are correct")


if __name__ == "__main__":
    main()
