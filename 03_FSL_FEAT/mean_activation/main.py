########################################################################################################################
# Create mean activation map
import os
import pandas as pd
import numpy as np
try:
    import nibabel as nib
except ModuleNotFoundError:
    raise RuntimeError(
        "Nibabel not installed. "
        "Install nibabel or run with the correct Python environment."
    )

# --- User settings: adjust these paths as needed ---
grouping_path = '/.../EMBARC/code/03_FSL_FEAT/Medication_grouping.xlsx'
data_root = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'
feat_dirname = 'FEAT_results_5'
contrast_filename = 'stats/cope2.nii.gz'
output_dir = '/.../EMBARC/data/03_FSL_FEAT/group_maps'

os.makedirs(output_dir, exist_ok=True)

# --- Load grouping metadata ---
meta = pd.read_excel(grouping_path)

# Define sessions and inclusion flags
sessions = {
    'ses-1': 'Inclusion-ses-1-new',
    'ses-2': 'Inclusion-ses-2-new'
}

# Custom binary erosion without scipy
def binary_erosion(mask, iterations=1):
    for _ in range(iterations):
        m = mask.copy()
        # intersect neighbors along each axis
        for ax in range(3):
            m &= np.roll(mask, 1, axis=ax)
            m &= np.roll(mask, -1, axis=ax)
        mask = m
    return mask

for ses, incl_col in sessions.items():
    for med_val, med_name in [(0, 'placebo'), (1, 'sertraline')]:
        sel = meta[(meta['Medication'] == med_val) & (meta[incl_col] == 1)]
        if sel.empty:
            print(f"No subjects for {med_name} in {ses}")
            continue

        cope_imgs, func_imgs = [], []
        affine = header = None

        for sid in sel['subject_id']:
            subj_folder = sid if sid.startswith('sub-') else f'sub-{sid}'
            subj_dir = os.path.join(
                data_root, subj_folder, ses, feat_dirname, f"{subj_folder}_{ses}.feat"
            )
            cope_path = os.path.join(subj_dir, contrast_filename)
            func_path = os.path.join(subj_dir, 'mean_func.nii.gz')
            if not os.path.exists(cope_path) or not os.path.exists(func_path):
                print(f"Missing cope2 or mean_func for {subj_folder} in {ses}")
                continue

            # Load and collect data
            img = nib.load(cope_path)
            data = img.get_fdata()
            cope_imgs.append(data)
            if affine is None:
                affine, header = img.affine, img.header

            func_data = nib.load(func_path).get_fdata()
            func_imgs.append(func_data)

        if not cope_imgs or not func_imgs:
            print(f"No valid data for {med_name} in {ses}")
            continue

        # Compute group means
        mean_cope = np.mean(np.stack(cope_imgs, axis=0), axis=0)
        mean_func = np.mean(np.stack(func_imgs, axis=0), axis=0)

        # Create brain mask: threshold + erosion
        # Adjust threshold_factor (e.g. 0.5) as needed to exclude skull
        threshold_factor = 0.4
        thresh = threshold_factor * np.max(mean_func)
        brain_mask = mean_func > thresh
        # Erode mask to remove peripheral skull
        # brain_mask = binary_erosion(brain_mask, iterations=2)

        # Apply mask to activation
        masked_mean = mean_cope * brain_mask

        # Save mask and masked activation
        mask_img = nib.Nifti1Image(brain_mask.astype(np.uint8), affine, header)
        nib.save(mask_img, os.path.join(output_dir, f'brain_mask_{med_name}_{ses}.nii.gz'))
        act_img = nib.Nifti1Image(masked_mean, affine, header)
        nib.save(act_img, os.path.join(output_dir, f'mean_cope2_new_{med_name}_{ses}_brainmasked.nii.gz'))

        print(f"Saved mask and masked map for {med_name} in {ses}")

print("Done.")


#####################################################################################################################
# Remove FEAT_results subfolders
#!/usr/bin/env python3
"""
Interactive cleanup of old FEAT_results directories, keeping only FEAT_results_5.
Lists to-be-removed folders first, then asks for confirmation.
"""
import os
import shutil

# --- User settings: adjust this path as needed ---
data_root = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'
old_dirs = ['FEAT_results',
            'FEAT_results_1',
            'FEAT_results_2',
            'FEAT_results_3',
            'FEAT_results_4']
keep_dir = 'FEAT_results_5'

to_remove = []

def gather_old_folders():
    for subj in os.listdir(data_root):
        subj_path = os.path.join(data_root, subj)
        if not os.path.isdir(subj_path) or not subj.startswith('sub-'):
            continue
        for ses in os.listdir(subj_path):
            ses_path = os.path.join(subj_path, ses)
            if not os.path.isdir(ses_path):
                continue
            for old in old_dirs:
                old_path = os.path.join(ses_path, old)
                if os.path.isdir(old_path):
                    to_remove.append(old_path)
    return to_remove

if __name__ == '__main__':
    folders = gather_old_folders()
    if not folders:
        print("No old FEAT_results directories found.")
        exit(0)

    print("The following directories will be REMOVED:")
    for path in folders:
        print(f"  {path}")

    confirm = input("Proceed with deletion? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted. No directories were removed.")
        exit(0)

    for path in folders:
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")

    print("Cleanup complete. Only FEAT_results_5 directories remain.")
