# %% [markdown]
# # Binary Alloy Slab Adsorption Structure Generator
#
# This tool is used to batch-generate adsorption structures while preserving the
# selective dynamics constraints of the original POSCAR. Only the newly added
# adsorbate atoms are set to be movable.
#
# ## Features
# - Preserve the original slab selective dynamics constraints
# - Newly added adsorbate atoms are set to `T T T` (fully movable)
# - Support multiple adsorbates: H
# - Automatically limit the number of adsorption sites to control output size
# - Real-time progress display

# %% [markdown]
# ## 1. Import required libraries

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
# ## 2. Set parameters

# %%
# Set path parameters
parent_folder = "GNN/Slab/BinaryAlloys_Slab_Fixed"
output_base = "GNN_OPT/PDS_H_2"

# Maximum number of adsorption sites
MAX_ADSORPTION_SITES = 10

print(f"Input folder: {parent_folder}", flush=True)
print(f"Output folder: {output_base}", flush=True)
print(f"Maximum adsorption sites: {MAX_ADSORPTION_SITES}", flush=True)

# Check path
print(f"\nChecking whether the input path exists: {os.path.exists(parent_folder)}", flush=True)
if not os.path.exists(parent_folder):
    print("Error: Input path does not exist!", flush=True)
    sys.exit(1)

# %% [markdown]
# ## 3. Define adsorbates

# %%
# Define adsorbate molecules
adsorbates = {
    "H": Molecule("H", [[0.0, 0.0, 0.0]])
}

# Select molecules to generate
molecules_to_generate = ["H"]

print("\nDefined adsorbates:", flush=True)
for name, mol in adsorbates.items():
    print(f"  {name}: {len(mol)} atoms ({mol.formula})", flush=True)
print(f"\nMolecules to generate: {', '.join(molecules_to_generate)}", flush=True)

# %% [markdown]
# ## 4. Define core functions

# %%
def create_selective_dynamics_with_adsorbate(original_poscar, ads_structure):
    """
    Create selective dynamics flags while preserving the original constraints
    - Preserve the original slab selective dynamics constraints
    - Set newly added adsorbate atoms to movable (True, True, True)
    """
    # Number of atoms in the original structure
    original_natoms = len(original_poscar.structure)
    ads_natoms = len(ads_structure)

    selective_dynamics = []

    # 1. Process original slab atoms (preserve original constraints)
    if hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None:
        for i in range(original_natoms):
            selective_dynamics.append(original_poscar.selective_dynamics[i])
    else:
        for i in range(original_natoms):
            selective_dynamics.append([False, False, False])

    # 2. Newly added adsorbate atoms (set to movable)
    adsorbate_atoms = ads_natoms - original_natoms
    for i in range(adsorbate_atoms):
        selective_dynamics.append([True, True, True])

    return selective_dynamics

# %%
def process_single_folder(subfolder_path, subfolder_name):
    """
    Process a single folder
    """
    # Check whether POSCAR exists
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
        # Read the original POSCAR object (preserve selective dynamics information)
        original_poscar = Poscar.from_file(poscar_path)
        slab = original_poscar.structure

        # Update statistics
        folder_stats['slab_atoms'] = len(slab)
        folder_stats['composition'] = str(slab.composition)
        folder_stats['has_selective'] = hasattr(original_poscar, 'selective_dynamics') and original_poscar.selective_dynamics is not None

        # Create adsorption site finder
        asf = AdsorbateSiteFinder(slab)

        # Find adsorption sites
        ads_sites = asf.find_adsorption_sites(
            distance=1.5,
            near_reduce=0.25,
            no_obtuse_hollow=True
        )

        total_sites = len(ads_sites['all'])
        folder_stats['adsorption_sites'] = total_sites

        # Generate adsorption structures for each molecule
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

                # Save structures
                for i, ads_struct in enumerate(ads_structs):
                    folder_path = os.path.join(
                        output_base, f"Slab_{mol_name}", subfolder_name, f"{mol_name}_{i+1:02d}"
                    )
                    os.makedirs(folder_path, exist_ok=True)

                    # Create selective dynamics flags while preserving original constraints
                    selective_dynamics = create_selective_dynamics_with_adsorbate(original_poscar, ads_struct)

                    # Save POSCAR (with selective dynamics)
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
# ## 5. Main processing function (with progress display)

# %%
def batch_process_folders():
    """
    Batch process all folders
    """
    print("\n" + "=" * 70, flush=True)
    print("Binary Alloy Slab Adsorption Structure Generator - Preserve Original Constraints Version", flush=True)
    print("=" * 70, flush=True)
    print(f"Input folder: {parent_folder}", flush=True)
    print(f"Output folder: {output_base}", flush=True)
    print(f"Molecules to generate: {', '.join(molecules_to_generate)}", flush=True)
    print(f"Maximum adsorption sites: {MAX_ADSORPTION_SITES}", flush=True)
    print("Constraint rule: preserve original slab constraints, newly added adsorbate atoms are movable", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(parent_folder):
        print(f"Error: Input folder does not exist: {parent_folder}", flush=True)
        return None

    os.makedirs(output_base, exist_ok=True)

    total_folders = 0
    processed_folders = 0
    failed_folders = 0
    no_poscar_folders = 0
    total_structures = 0
    all_stats = []

    print("\nCollecting folder list...", flush=True)
    subfolders = sorted([f for f in os.listdir(parent_folder)
                        if os.path.isdir(os.path.join(parent_folder, f))])

    total = len(subfolders)
    print(f"Found {total} subfolders", flush=True)
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

        # Display progress every 10 folders
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total - idx)

            print(f"Progress: {idx}/{total} ({idx*100//total}%) | "
                  f"Success: {processed_folders} | Failed: {failed_folders} | No POSCAR: {no_poscar_folders} | "
                  f"Generated structures: {total_structures} | "
                  f"Elapsed: {elapsed/60:.1f} min | Estimated remaining: {eta/60:.1f} min", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Processing completed!", flush=True)
    print(f"Total folders: {total_folders}", flush=True)
    print(f"Successfully processed: {processed_folders}", flush=True)
    print(f"Failed: {failed_folders}", flush=True)
    print(f"Folders without POSCAR: {no_poscar_folders}", flush=True)
    print(f"Total generated structures: {total_structures}", flush=True)
    print(f"Total elapsed time: {(time.time() - start_time)/60:.1f} minutes", flush=True)
    print(f"Processed molecules: {', '.join(molecules_to_generate)}", flush=True)

    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'failed_folders': failed_folders,
        'no_poscar_folders': no_poscar_folders,
        'total_structures': total_structures,
        'folder_stats': all_stats
    }

# %% [markdown]
# ## 6. Run processing directly (skip statistics stage)

# %%
# Run batch processing directly (skip statistics stage to improve speed)
print("\nStarting batch processing (skipping the statistics stage for faster execution)...", flush=True)
results = batch_process_folders()

# %% [markdown]
# ## 7. Result analysis

# %%
if results is not None:
    print("\n" + "=" * 70, flush=True)
    print("Detailed processing statistics:", flush=True)
    print("=" * 70, flush=True)
    print(f"Successfully processed folders: {results['processed_folders']}", flush=True)
    print(f"Failed folders: {results['failed_folders']}", flush=True)
    print(f"Folders without POSCAR: {results['no_poscar_folders']}", flush=True)
    print(f"Total generated structures: {results['total_structures']}", flush=True)

    # Count the number of generated structures for each molecule
    print("\nGenerated structure statistics by molecule:", flush=True)
    for mol_name in molecules_to_generate:
        total_struct = sum([stats['structures_generated'].get(mol_name, 0)
                           for stats in results['folder_stats']
                           if stats['status'] == 'success'])
        print(f"  {mol_name}: {total_struct} structures", flush=True)

    # Display the first few successfully processed folders
    successful_folders = [stats for stats in results['folder_stats'] if stats['status'] == 'success']
    if successful_folders:
        print(f"\nFirst 5 successfully processed folders:", flush=True)
        for i, stats in enumerate(successful_folders[:5]):
            struct_count = sum(stats['structures_generated'].values())
            print(f"  {i+1}. {stats['folder_name']}: {stats['slab_atoms']} atoms, "
                  f"{stats['adsorption_sites']} adsorption sites, generated {struct_count} structures", flush=True)

    # Display failed folders
    failed_folders = [stats for stats in results['folder_stats']
                     if stats['status'] == 'failed']
    if failed_folders:
        print(f"\nFailed folders ({len(failed_folders)}):", flush=True)
        for stats in failed_folders[:5]:
            error_msg = stats.get('error', 'Unknown error')
            print(f"  - {stats['folder_name']}: {error_msg}", flush=True)
        if len(failed_folders) > 5:
            print(f"  ... and {len(failed_folders)-5} more failed folders", flush=True)
else:
    print("No processing results to display", flush=True)

# %% [markdown]
# ## 8. Output directory structure example

# %%
if results is not None:
    print(f"\nOutput directory structure:", flush=True)
    print(f"{output_base}/", flush=True)
    for mol_name in molecules_to_generate:
        print(f"├── Slab_{mol_name}/", flush=True)
        print(f"│   ├── [subfolder_name]/", flush=True)
        print(f"│   │   ├── {mol_name}_01/POSCAR", flush=True)
        print(f"│   │   ├── {mol_name}_02/POSCAR", flush=True)
        print(f"│   │   └── ... (up to {MAX_ADSORPTION_SITES})", flush=True)
        print(f"│   └── ...", flush=True)

    print(f"\nConstraint rules:", flush=True)
    print("• Original slab atoms: preserve the selective dynamics constraints in the original POSCAR", flush=True)
    print("• Newly added adsorbate atoms: T T T (fully movable)", flush=True)

# %% [markdown]
# ## 9. Verify generated structures

# %%
def verify_generated_structure():
    """
    Verify whether the generated structure is correct
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

                            print(f"\nVerifying structure: {sample_path}", flush=True)
                            print(f"  Total number of atoms: {len(structure)}", flush=True)
                            print(f"  Formula: {structure.composition}", flush=True)

                            if hasattr(poscar, 'selective_dynamics') and poscar.selective_dynamics is not None:
                                moveable = sum(1 for sd in poscar.selective_dynamics if any(sd))
                                fixed = len(structure) - moveable
                                print(f"  Selective dynamics: movable {moveable}, fixed {fixed}", flush=True)

                                # Display constraints for the last few atoms
                                # These should correspond to adsorbate atoms and should be T T T
                                print("  Constraints for the last 3 atoms (expected to be adsorbate atoms):", flush=True)
                                for i, sd in enumerate(poscar.selective_dynamics[-3:], len(poscar.selective_dynamics)-2):
                                    print(f"    Atom {i}: {sd}", flush=True)
                            else:
                                print("  No selective dynamics information", flush=True)

                            return True
                        except Exception as e:
                            print(f"  Verification failed: {e}", flush=True)
                            return False

    print("No generated structure found for verification", flush=True)
    return False

# Verify structures
if results is not None and results['processed_folders'] > 0:
    print("\n" + "=" * 70, flush=True)
    print("Structure verification", flush=True)
    print("=" * 70, flush=True)
    verify_generated_structure()

print("\nScript execution completed!", flush=True)
