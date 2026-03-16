import os
import re
import pandas as pd
from pathlib import Path

def extract_energy_from_filename(filename):
    """
    Extract energy value from filename
    Example:
    POSCAR_opt_-0.23305eV.vasp -> -0.23305
    """
    pattern = r'POSCAR_opt_([+-]?\d+\.?\d*)eV\.vasp'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None


def scan_directory_structure(base_path):
    """
    Scan directory structure and extract H adsorption energies
    """
    data = []
    base_path = Path(base_path)
    
    # Find folders containing "H"
    slab_folders = [f for f in base_path.iterdir() if f.is_dir() and 'H' in f.name]

    if not slab_folders:
        print("No folders containing 'H' found. Trying folders starting with 'Slab_'...")
        slab_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('Slab_')]

    for slab_folder in slab_folders:
        slab_name = slab_folder.name
        
        print(f"Processing {slab_name}...")

        # Find metal surface folders (e.g., Ag3Pt, Cu)
        metal_folders = [f for f in slab_folder.iterdir() if f.is_dir()]

        for metal_folder in metal_folders:
            metal_name = metal_folder.name

            # Find adsorption site folders (e.g., H_01, H_02)
            site_folders = [f for f in metal_folder.iterdir() if f.is_dir()]

            for site_folder in site_folders:
                site_name = site_folder.name.split('_')[-1]  # site index

                # Search for POSCAR files
                poscar_files = list(site_folder.glob('POSCAR_opt_*eV.vasp'))

                if poscar_files:
                    poscar_file = poscar_files[0]
                    energy = extract_energy_from_filename(poscar_file.name)

                    if energy is not None:
                        data.append({
                            'Slab': metal_name,
                            'Site': site_name,
                            'H_eV': energy
                        })

                        print(f"  Site {site_name}: {energy:.5f} eV")

    return data


def create_energy_table(base_path, output_file='H_energy_table.csv'):
    """
    Create H adsorption energy table
    """
    print("=" * 60)
    print("Scanning directory structure and extracting H adsorption energies...")
    print("=" * 60)

    data = scan_directory_structure(base_path)

    if not data:
        print("\n❌ No data found!")
        print("Please check:")
        print("  1. Whether the directory structure is correct")
        print("  2. Whether folders containing 'H' exist")
        print("  3. Whether files follow the format POSCAR_opt_XXXeV.vasp")
        return None

    # Convert to DataFrame and sort
    df = pd.DataFrame(data)
    df = df.sort_values(['Slab', 'Site']).reset_index(drop=True)

    # Save CSV
    df.to_csv(output_file, index=False, float_format='%.5f')

    print(f"\n{'='*60}")
    print(f"✅ Data saved to {output_file}")
    print(f"✅ Total records: {len(df)}")
    print(f"{'='*60}")

    print("\nData preview:")
    print(df.to_string(index=False))

    return df


def main():
    print("=" * 60)
    print("   🔋 H Adsorption Energy Extraction Tool")
    print("=" * 60)

    # Set base path
    base_path = input("\nEnter base path (press Enter to use current directory): ").strip()

    if not base_path:
        base_path = "."

    if not os.path.exists(base_path):
        print(f"❌ Path does not exist: {base_path}")
        return

    try:
        df = create_energy_table(base_path)

        if df is not None:

            # Statistics
            print(f"\nStatistics:")
            print(f"  Number of slabs: {df['Slab'].nunique()}")
            print(f"  Number of sites: {df['Site'].nunique()}")
            print(f"  H adsorption energy range: {df['H_eV'].min():.5f} ~ {df['H_eV'].max():.5f} eV")
            print(f"  Average H adsorption energy: {df['H_eV'].mean():.5f} eV")

            # Most stable adsorption site
            min_idx = df['H_eV'].idxmin()

            print(f"\n Most stable adsorption site:")
            print(f"  Slab: {df.loc[min_idx, 'Slab']}")
            print(f"  Site: {df.loc[min_idx, 'Site']}")
            print(f"  Energy: {df.loc[min_idx, 'H_eV']:.5f} eV")

            # Missing data check
            missing_count = df['H_eV'].isnull().sum()
            if missing_count > 0:
                print(f"\n Found {missing_count} missing values")

    except Exception as e:

        print(f"\n Error occurred during processing: {e}")

        import traceback
        traceback.print_exc()

        print("\nPlease check:")
        print("  1. Whether file paths are correct")
        print("  2. Whether directory structure matches expectation")
        print("  3. Whether filenames follow POSCAR_opt_XXXeV.vasp format")


if __name__ == "__main__":
    main()
