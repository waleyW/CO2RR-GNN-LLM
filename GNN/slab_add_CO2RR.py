# %% [markdown]
# # Binary Alloy Slab Adsorption Structure Generator
#
# This tool batch-generates adsorption structures while preserving the
# original selective dynamics constraints from the POSCAR file.
# Only the newly added adsorbate atoms are set to movable.
#
# ## Features
# - Preserve the original selective dynamics constraints of the slab
# - Newly added adsorbate atoms are set to `T T T` (fully movable)
# - Supports multiple adsorbates: CHC, CHCOHH, CHCOH
# - Automatically limits the number of adsorption sites to control output size

# %% [markdown]
# ## 1. Import Required Libraries

# %%
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
import os
import numpy as np
from pathlib import Path

# %% [markdown]
# ## 2. Set Parameters

# %%
# Path configuration
parent_folder = "/Slab/BinaryAlloys_Slab_Fixed"
output_base = "/GNN_OPT/PDS"

# Maximum number of adsorption sites
MAX_ADSORPTION_SITES = 10

print(f"Input folder: {parent_folder}")
print(f"Output folder: {output_base}")
print(f"Maximum adsorption sites: {MAX_ADSORPTION_SITES}")

# %% [markdown]
# ## 3. Define Adsorbate Molecules

# %%
# Define adsorbate molecules (validated coordinates)
adsorbates = {
    "H": Molecule("H", [[0.0, 0.0, 0.0]])
}

# Select molecules to generate
molecules_to_generate = ["H"]

print("Defined adsorbate molecules:")
for name, mol in adsorbates.items():
    print(f"  {name}: {len(mol)} atoms ({mol.formula})")

print(f"\nMolecules to generate: {', '.join(molecules_to_generate)}")

# %% [markdown]
# ## 4. Define Core Functions

# %%
def create_selective_dynamics_with_adsorbate(original_poscar, ads_structure):
    """
    Create selective dynamics flags while preserving original constraints.

    - Original slab atoms keep their original constraints
    - Newly added adsorbate atoms are set to movable (True, True, True)
    """

    original_natoms = len(original_poscar.structure)
    ads_natoms = len(ads_structure)

    print(f"        Original atom count: {original_natoms}")
    print(f"        Atom count after adsorption: {ads_natoms}")
    print(f"        Newly added atoms: {ads_natoms - original_natoms}")

    selective_dynamics = []

    # Handle original slab atoms
    if hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None:
        print("        Using original POSCAR selective dynamics constraints")
        for i in range(original_natoms):
            selective_dynamics.append(original_poscar.selective_dynamics[i])
    else:
        print("        No selective dynamics in original POSCAR, fixing all slab atoms")
        for i in range(original_natoms):
            selective_dynamics.append([False, False, False])

    # Handle newly added adsorbate atoms
    adsorbate_atoms = ads_natoms - original_natoms
    for i in range(adsorbate_atoms):
        selective_dynamics.append([True, True, True])

    print(f"        Final constraint setup: slab constraints preserved, {adsorbate_atoms} adsorbate atoms movable")

    return selective_dynamics


# %%
def process_single_folder(subfolder_path, subfolder_name):
    """
    Process a single slab folder
    """

    poscar_path = os.path.join(subfolder_path, "POSCAR")
    if not os.path.exists(poscar_path):
        print(f"Skipping {subfolder_name}: POSCAR not found")
        return False, {}

    print(f"\nProcessing folder: {subfolder_name}")

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

        original_poscar = Poscar.from_file(poscar_path)
        slab = original_poscar.structure

        folder_stats['slab_atoms'] = len(slab)
        folder_stats['composition'] = str(slab.composition)

        has_selective = hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None
        folder_stats['has_selective'] = has_selective

        print(f"  Slab atom count: {len(slab)}")
        print(f"  Composition: {slab.composition}")
        print(f"  Contains selective dynamics: {'Yes' if has_selective else 'No'}")

        if has_selective:
            original_moveable = sum(1 for sd in original_poscar.selective_dynamics if any(sd))
            original_fixed = len(slab) - original_moveable
            print(f"  Original constraints: movable {original_moveable}, fixed {original_fixed}")

        asf = AdsorbateSiteFinder(slab)

        ads_sites = asf.find_adsorption_sites(
            distance=1.5,
            near_reduce=0.25,
            no_obtuse_hollow=True
        )

        total_sites = len(ads_sites['all'])
        folder_stats['adsorption_sites'] = total_sites

        print(f"  Found adsorption sites: {total_sites}")

        if total_sites > MAX_ADSORPTION_SITES:
            print(f"  Limiting to first {MAX_ADSORPTION_SITES} sites")

        for mol_name in molecules_to_generate:

            print(f"  Generating adsorption structures for {mol_name}...")

            try:

                ads_structs = asf.generate_adsorption_structures(
                    adsorbates[mol_name],
                    repeat=[1, 1, 1],
                    min_lw=10.0,
                    find_args={"distance": 1.5, "near_reduce": 0.25}
                )

                if len(ads_structs) > MAX_ADSORPTION_SITES:
                    ads_structs = ads_structs[:MAX_ADSORPTION_SITES]

                print(f"    Generated {len(ads_structs)} structures")

                folder_stats['structures_generated'][mol_name] = len(ads_structs)

                for i, ads_struct in enumerate(ads_structs):

                    folder_path = os.path.join(
                        output_base, f"Slab_{mol_name}", subfolder_name, f"{mol_name}_{i+1:02d}"
                    )

                    os.makedirs(folder_path, exist_ok=True)

                    selective_dynamics = create_selective_dynamics_with_adsorbate(original_poscar, ads_struct)

                    poscar = Poscar(ads_struct, selective_dynamics=selective_dynamics)

                    poscar.write_file(os.path.join(folder_path, "POSCAR"))

                    total_atoms = len(ads_struct)
                    moveable_atoms = sum(1 for sd in selective_dynamics if any(sd))
                    fixed_atoms = total_atoms - moveable_atoms

                    print(f"      Saved: {folder_path}/POSCAR")
                    print(f"        Total atoms: {total_atoms}, Movable: {moveable_atoms}, Fixed: {fixed_atoms}")

            except Exception as e:
                print(f"    Error generating {mol_name}: {str(e)}")
                folder_stats['structures_generated'][mol_name] = 0
                continue

        folder_stats['status'] = 'success'

        return True, folder_stats

    except Exception as e:
        print(f"  Error processing {subfolder_name}: {str(e)}")
        return False, folder_stats


# %% [markdown]
# ## 5. Folder Statistics

# %%
def get_folder_statistics():
    """
    Get statistics of input folders
    """

    if not os.path.exists(parent_folder):
        print(f"Input folder does not exist: {parent_folder}")
        return None

    subfolders = [f for f in os.listdir(parent_folder)
                  if os.path.isdir(os.path.join(parent_folder, f))]

    print("Input folder statistics:")
    print(f"  Total subfolders: {len(subfolders)}")

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

    print(f"  Folders containing POSCAR: {poscar_count}")
    print(f"  POSCAR with selective dynamics: {selective_count}")

    print(f"  Estimated max structures generated: {poscar_count * len(molecules_to_generate) * MAX_ADSORPTION_SITES}")

    return {
        'total_folders': len(subfolders),
        'poscar_folders': poscar_count,
        'selective_folders': selective_count
    }


stats = get_folder_statistics()


# %% [markdown]
# ## 6. Batch Processing

# %%
def batch_process_folders():
    """
    Batch process all folders
    """

    print("Binary Alloy Slab Adsorption Structure Generator - Preserve Constraints Version")

    print("=" * 70)

    print(f"Input folder: {parent_folder}")
    print(f"Output folder: {output_base}")
    print(f"Molecules: {', '.join(molecules_to_generate)}")

    print(f"Maximum adsorption sites: {MAX_ADSORPTION_SITES}")

    print("Constraint strategy: preserve original slab constraints, adsorbate atoms movable")

    print("=" * 70)

    if not os.path.exists(parent_folder):
        print(f"Error: input folder does not exist: {parent_folder}")
        return None

    os.makedirs(output_base, exist_ok=True)

    total_folders = 0
    processed_folders = 0
    failed_folders = 0
    all_stats = []

    subfolders = sorted([f for f in os.listdir(parent_folder)
                        if os.path.isdir(os.path.join(parent_folder, f))])

    print(f"Found {len(subfolders)} subfolders")

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

    print("Processing finished!")

    print(f"Total folders: {total_folders}")
    print(f"Successfully processed: {processed_folders}")
    print(f"Failed: {failed_folders}")

    print(f"Molecules processed: {', '.join(molecules_to_generate)}")

    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'failed_folders': failed_folders,
        'folder_stats': all_stats
    }


# %% [markdown]
# ## 7. Run Batch Processing

# %%
results = batch_process_folders()


# %% [markdown]
# ## 8. Result Analysis

# %%
if results is not None:

    print("\nDetailed processing statistics:")

    print(f"Successful folders: {results['processed_folders']}")
    print(f"Failed folders: {results['failed_folders']}")

    print("\nGenerated structures per molecule:")

    for mol_name in molecules_to_generate:

        total_structures = sum([stats['structures_generated'].get(mol_name, 0)
                               for stats in results['folder_stats']
                               if stats['status'] == 'success'])

        print(f"  {mol_name}: {total_structures} structures")

    successful_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'success']

    if successful_folders:

        print("\nFirst 5 successfully processed folders:")

        for i, stats in enumerate(successful_folders[:5]):

            print(f"  {i+1}. {stats['folder_name']}: {stats['slab_atoms']} atoms, {stats['adsorption_sites']} sites")

    failed_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'failed']

    if failed_folders:

        print(f"\nFailed folders ({len(failed_folders)}):")

        for stats in failed_folders[:5]:
            print(f"  - {stats['folder_name']}")

else:

    print("No results to display")


# %% [markdown]
# ## 9. Output Directory Structure

# %%
if results is not None:

    print("\nOutput directory structure:")

    print(f"{output_base}/")

    for mol_name in molecules_to_generate:

        print(f"├── Slab_{mol_name}/")
        print(f"│   ├── [slab folder]/")
        print(f"│   │   ├── {mol_name}_01/POSCAR")
        print(f"│   │   ├── {mol_name}_02/POSCAR")
        print(f"│   │   └── ... (max {MAX_ADSORPTION_SITES})")
        print(f"│   └── ...")

    print("\nConstraint rules:")

    print("• Original slab atoms: keep selective dynamics from POSCAR")

    print("• Newly added adsorbate atoms: T T T (fully movable)")
