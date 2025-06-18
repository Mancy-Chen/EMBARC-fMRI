# Mancy Chen 29/01/2025


########################################################################################################################
# Visualize Schaefer atlas
import sys
sys.path.append('/.../miniconda3/lib/python3.10/site-packages')

# Import necessary libraries
from nilearn import datasets, plotting, image
import matplotlib.pyplot as plt

# Fetch the Schaefer 2018 atlas with 400 ROIs and 7 networks
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7,
                                               resolution_mm=1, verbose=1)

# Inspect the fetched atlas
print("Atlas keys:", schaefer.keys())
print("Number of labels:", len(schaefer.labels))
print("First 10 labels:", schaefer.labels[:10])

# The atlas maps file (NIfTI image)
atlas_filename = schaefer.maps
print("Atlas filename:", atlas_filename)

# Visualize the atlas using Nilearn's plotting functions

# Plot the atlas in an orthogonal view (sagittal, coronal, axial)
plotting.plot_roi(atlas_filename, title='Schaefer 2018 Atlas (400 ROIs, 7 Networks)',
                  display_mode='ortho', cut_coords=(0, 0, 0), cmap='Paired')

# Alternatively, plot the atlas as a stat map with one slicing direction
plotting.plot_stat_map(atlas_filename, title='Schaefer 2018 Atlas (Axial Slice)',
                       display_mode='z', cut_coords=5, cmap='Paired')

# Show the plots
plotting.show()

########################################################################################################################
# Create Schaefer atlas mask
import sys
sys.path.append('/.../miniconda3/lib/python3.10/site-packages')
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img
from nilearn import plotting

# Load your mean functional image
mean_func_path = '/.../EMBARC/data/04_radiomics/Whole-data/mean_func.nii.gz'
mean_func_img = nib.load(mean_func_path)

# Fetch the Schaefer atlas (400 regions, 7 networks)
atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
atlas_img = nib.load(atlas.maps)
# atlas_img = nib.load('/data/projects/EMBARC/data/04_radiomics/mask/Schaefer_atlas/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz')

# Resample the atlas image to the space of your mean functional image
atlas_resampled = resample_to_img(atlas_img, mean_func_img, interpolation='nearest', force_resample=True)

# Visualize the resampled atlas
plotting.plot_roi(atlas_resampled, title="Schaefer Atlas with 400 Regions", draw_cross=False)
plotting.show()

# Save the resampled atlas (with individual labels) as a NIfTI file.
output_filename = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/schaefer_atlas_resampled.nii.gz'
nib.save(atlas_resampled, output_filename)

print("Mask saved as 'schaefer_mask_resampled.nii.gz'")

########################################################################################################################
# Create Melbounre atlas mask
import sys
sys.path.append('/.../miniconda3/lib/python3.10/site-packages')
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

# Path to your reference mean functional image (MNI152 2mm)
mean_func_path = '/.../EMBARC/04_radiomics/Whole-data/mean_func.nii.gz'
mean_func_img = nib.load(mean_func_path)

# Path to your Melbourne subcortical atlas NIfTI file
# (Update this path to where your Melbourne atlas file is located)
melbourne_atlas_path = '/.../EMBARC/04_radiomics/mask/Melbourne_atlas/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S2_3T_2009cAsym.nii.gz'
melbourne_img = nib.load(melbourne_atlas_path)

# Resample the Melbourne atlas to the space of the mean functional image using nearest-neighbor interpolation
melbourne_resampled = resample_to_img(
    melbourne_img,
    mean_func_img,
    interpolation='nearest',
    force_resample=True,
    copy_header=True
)
# Create a binary mask: any voxel with a label > 0 is considered part of a subcortical region.
melbourne_data = melbourne_resampled.get_fdata()

# Save the mask as a new NIfTI file using the affine and header from the reference image
output_filename = '32_melbourne_all_regions_mask_resampled.nii.gz'
mask_img = nib.Nifti1Image(melbourne_data, mean_func_img.affine, mean_func_img.header)
nib.save(mask_img, output_filename)

print(f"Melbourne atlas mask saved as '{output_filename}'")

########################################################################################################################
# Eliminate the outbrain area
import nibabel as nib
import numpy as np

# Define paths for the masks
maskA_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/mask_moved_up_4.0mm_right_-1.5mm_and_front_1.5mm.nii.gz'
maskB_path = '/.../EMBARC/data/04_radiomics/mask/Mean_Brain_bet/mean_func_bet_mask.nii.gz'
maskC_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/filtered_mask.nii.gz'

# Load the images
maskA_img = nib.load(maskA_path)
maskB_img = nib.load(maskB_path)

# Get the image data as numpy arrays
maskA_data = maskA_img.get_fdata()
maskB_data = maskB_img.get_fdata()

# Multiply the two masks elementwise
maskC_data = maskA_data * maskB_data

# Create a new NIfTI image for maskC using the affine and header from MaskA (or MaskB if they share the same space)
maskC_img = nib.Nifti1Image(maskC_data, maskA_img.affine, maskA_img.header)

# Save the result
nib.save(maskC_img, maskC_path)

print(f"MaskC saved at: {maskC_path}")
######################################################################################################################
# Manually correct the mask by moving with pixels
import nibabel as nib
import numpy as np

# Path to your mask file
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/400_schaefer_atlas_resampled.nii.gz'
mask_img = nib.load(mask_path)

# Copy the affine matrix to modify it
affine = mask_img.affine.copy()
print("Original affine:\n", affine)

# Define the shifts in millimeters
shift_z = 4.0  # Move upward along Z-axis (superior direction)
shift_x = -1.5  # Move to the right along X-axis
shift_y = 1.5  # Move to the front along Y-axis (anterior direction)

# Adjust the affine matrix:
affine[2, 3] += shift_z  # Adjust Z translation
affine[0, 3] += shift_x  # Adjust X translation
affine[1, 3] += shift_y  # Adjust Y translation

print("Modified affine:\n", affine)

# Create a new NIfTI image with the modified affine
mask_shifted = nib.Nifti1Image(mask_img.get_fdata(), affine, mask_img.header)

# Create an output filename that includes the shift info
output_filename = ("/.../EMBARC/data/04_radiomics/mask/Processed_mask/"+f"mask_moved_up_{shift_z:.1f}mm_right_{shift_x:.1f}mm_and_front_{shift_y:.1f}mm.nii.gz")

# Save the shifted mask to a new file
nib.save(mask_shifted, output_filename)

print(f"Shifted mask saved as '{output_filename}'")

########################################################################################################################
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

# File paths
maskA_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/32_melbourne_all_regions_mask_resampled.nii.gz'
maskB_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/filtered_mask.nii.gz'
combined_mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA.nii.gz'

# Load maskA (32-region Melbourne atlas) and maskB (400-region cortical mask)
maskA_img = nib.load(maskA_path)
maskB_img = nib.load(maskB_path)

# Resample maskB to the space of maskA so that maskA's alignment is preserved.
maskB_resampled = resample_to_img(maskB_img, maskA_img, interpolation='nearest')

# Extract the data arrays
maskA_data = maskA_img.get_fdata()
maskB_data = maskB_resampled.get_fdata()

# Shift maskA's labels by adding 400 (so that regions 1-32 become 401-432)
maskA_shifted = np.where(maskA_data > 0, maskA_data + 400, 0)

# Create the combined mask:
# We start with the resampled maskB data and then replace voxels where maskA (shifted) has nonzero values.
combined_data = maskB_data.copy()
combined_data[maskA_shifted > 0] = maskA_shifted[maskA_shifted > 0]

# Create a new NIfTI image using maskA's affine and header (thus preserving maskA's spatial alignment)
combined_img = nib.Nifti1Image(combined_data, maskA_img.affine, maskA_img.header)

# Save the combined mask
nib.save(combined_img, combined_mask_path)
print(f"Combined mask saved as '{combined_mask_path}'")
########################################################################################################################
import os
import glob
import pandas as pd
from radiomics import featureextractor

# --------------------------
# Settings and Definitions
# --------------------------
# Define the mask path (assumed to be a multi‐label mask in MNI space)
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA.nii.gz'

# Define the directory containing all subjects
data_dir = '/.../EMBARC/data/04_radiomics/Whole-data/'

# Define ROI names with their corresponding label values in the mask.
roi_labels = {
    'left_amygdala': 420,
    'left_dACC': 169,
    'left_pgACC': 174,
    'left_sgACC': 177,
    'left_dlPFC': 106,
    # 'left_insula': 34,  # too many regions to decide
    'left_hippocampus': 417,
    'right_amygdala': 404,
    'right_dACC': 379,
    'right_pgACC': 381,
    'right_sgACC': 360,
    'right_dlPFC': 310,
    # 'right_insula': 235,  # too many regions to decide
    'right_hippocampus': 401
}

# Create the PyRadiomics feature extractor and configure it.
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')
extractor.enableFeatureClassByName('gldm')

print("Extraction settings:", extractor.settings)

# --------------------------
# Extraction Loop
# --------------------------
# We'll collect all results in a list of dictionaries.
results_list = []

# Loop over each subject folder (e.g., sub-CU0011, sub-MG0022, etc.)
subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub-*')))
print(f"Found {len(subject_dirs)} subject directories.")

for subject_dir in subject_dirs:
    subject = os.path.basename(subject_dir)
    print(f"\nProcessing subject: {subject}")

    # Process each session (ses-1 and ses-2)
    for session in ['ses-1', 'ses-2']:
        print(f"  Processing session: {session}")
        # Build the expected path for cope2.nii.gz
        cope_path = os.path.join(
            subject_dir, session, 'FEAT_results_4',
            f'{subject}_{session}.feat', 'stats', 'cope2.nii.gz'
        )
        print(f"    Looking for cope file: {cope_path}")
        if not os.path.exists(cope_path):
            print(f"    Skipping {subject} {session}: cope2 file not found.")
            continue

        # Loop over each ROI defined in roi_labels.
        for roi_name, label_value in roi_labels.items():
            try:
                # Extract features for the given ROI using the common mask.
                feature_result = extractor.execute(cope_path, mask_path, label=label_value)
                # Prepare a dictionary with subject, session, ROI info along with features.
                result_dict = {'subject': subject, 'session': session, 'ROI': roi_name}
                for key, value in feature_result.items():
                    # Skip any diagnostic information if present.
                    if not key.startswith('diagnostics'):
                        result_dict[key] = value
                results_list.append(result_dict)
                print(f"    Processed ROI: {roi_name}")
            except Exception as e:
                print(f"    Error processing ROI '{roi_name}' for {subject} {session}: {e}")

# Convert all results to a DataFrame.
df_all = pd.DataFrame(results_list)
print("\nExtraction completed.")

# --------------------------
# Pivoting and Saving CSVs
# --------------------------
# We want to create two CSV files:
#   - One for ses-1, one for ses-2.
# In each CSV, each row is a subject and each column is a feature computed for a given ROI,
# with column names in the form: ROI_FeatureName.

for session in ['ses-1', 'ses-2']:
    df_session = df_all[df_all['session'] == session]
    if df_session.empty:
        print(f"No data for {session}")
        continue

    # Drop the session column (since all rows here are from the same session).
    df_session = df_session.drop(columns=['session'])
    # Set index to subject and ROI, so that we have one row per subject and ROI becomes a level.
    df_pivot = df_session.set_index(['subject', 'ROI']).unstack('ROI')

    # The resulting columns are a MultiIndex with level 0 = feature name and level 1 = ROI.
    # We want to flatten it to "ROI_FeatureName".
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    output_csv = f"radiomics_features_{session}.csv"
    df_pivot.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
####################################################################################################################
# Merge all subregions in insula
import nibabel as nib
import numpy as np

# Define the path to your original mask and the new mask that will be saved
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA.nii.gz'
new_mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA_mergedInsula.nii.gz'

# Load the original mask image
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

# Define the original ROI labels for left insula
left_insula_labels = [
    34,  # 7Networks_LH_SomMot_4
    35,  # 7Networks_LH_SomMot_5
    97,  # 7Networks_LH_SalVentAttn_FrOperIns_2
    98,  # 7Networks_LH_SalVentAttn_FrOperIns_3
    99,  # 7Networks_LH_SalVentAttn_FrOperIns_4
    100, # 7Networks_LH_SalVentAttn_FrOperIns_5
    101, # 7Networks_LH_SalVentAttn_FrOperIns_6
    102, # 7Networks_LH_SalVentAttn_FrOperIns_7
    143  # 7Networks_LH_Cont_pCun_1
]

# Define the original ROI labels for right insula
right_insula_labels = [
    235, # 7Networks_RH_SomMot_6
    236, # 7Networks_RH_SomMot_7
    302, # 7Networks_RH_SalVentAttn_FrOperIns_2
    303, # 7Networks_RH_SalVentAttn_FrOperIns_3
    304, # 7Networks_RH_SalVentAttn_FrOperIns_4
    305, # 7Networks_RH_SalVentAttn_FrOperIns_5
    306, # 7Networks_RH_SalVentAttn_FrOperIns_6
    307, # 7Networks_RH_SalVentAttn_FrOperIns_7
    340  # 7Networks_RH_Cont_PFCl_1
]

# Choose new label values for the merged ROIs
new_left_insula_label = 500
new_right_insula_label = 501

# Merge left insula: Replace all voxels with any left insula label with new_left_insula_label
for label in left_insula_labels:
    mask_data[mask_data == label] = new_left_insula_label

# Merge right insula: Replace all voxels with any right insula label with new_right_insula_label
for label in right_insula_labels:
    mask_data[mask_data == label] = new_right_insula_label

# Optionally, check the unique labels to verify the merge
unique_labels = np.unique(mask_data)
print("Unique labels in the new mask:", unique_labels)

# Save the new mask
new_mask_img = nib.Nifti1Image(mask_data, affine=mask_img.affine, header=mask_img.header)
nib.save(new_mask_img, new_mask_path)
print("New mask with merged insula saved to:", new_mask_path)

##################################################################################################################################################################
# Convert the cope estimates into percent signal change.
import os
import glob
import subprocess


def process_cope2_files(base_dir, mean_func_path):
    # Check if the mean_func file exists.
    if not os.path.exists(mean_func_path):
        print(f"Mean functional file not found: {mean_func_path}")
        return

    # Compute the mean intensity from the mean_func image.
    try:
        result = subprocess.check_output(['fslstats', mean_func_path, '-M'])
        mean_intensity = float(result.strip())
        if mean_intensity == 0:
            print(f"Mean intensity is zero in {mean_func_path}. Aborting processing.")
            return
    except Exception as e:
        print(f"Error computing mean intensity for {mean_func_path}: {e}")
        return

    # Recursively find all cope2.nii.gz files in the FEAT directories.
    cope_files = glob.glob(os.path.join(base_dir, '**', 'FEAT_results', '*.feat', 'stats', 'cope2.nii.gz'),
                           recursive=True)
    print(f"Found {len(cope_files)} cope2 files.")

    for cope_file in cope_files:
        # Define the output path for the scaled cope2 file.
        output_file = os.path.join(os.path.dirname(cope_file), 'cope2_scaled.nii.gz')

        # Scale cope2 by dividing by the mean intensity and multiplying by 100.
        try:
            subprocess.run(['fslmaths', cope_file, '-div', str(mean_intensity), '-mul', '100', output_file], check=True)
            print(f"Processed: {cope_file} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error scaling {cope_file}: {e}")


if __name__ == '__main__':
    # Base directory where your data is stored.
    base_directory = '/.../EMBARC/data/03_FSL_FEAT/Data_augmented/400_volumes'
    # Provided path for the mean_func image.
    mean_func_file = '/.../EMBARC/data/04_radiomics/Whole-data/mean_func.nii.gz'

    process_cope2_files(base_directory, mean_func_file)

####################################################################################################################
# Z-score normalization
import nibabel as nib
import numpy as np
import glob
import os

# Define paths (update according to your actual directory structure)
feat_dirs = glob.glob('/.../EMBARC/data/03_FSL_FEAT/Whole-data/sub-*/ses-*/FEAT_results_5/sub-*_ses-*.feat')

for feat_dir in feat_dirs:
    func_file = os.path.join(feat_dir, 'stats/cope1.nii.gz')
    mask_file = os.path.join(feat_dir, 'mask.nii.gz')

    # Check if files exist
    if not (os.path.isfile(func_file) and os.path.isfile(mask_file)):
        print(f"Missing files in {feat_dir}, skipping...")
        continue

    # Load data
    img = nib.load(func_file)
    data = img.get_fdata()

    mask = nib.load(mask_file).get_fdata().astype(bool)

    # Compute mean and std within mask
    mean_val = np.mean(data[mask])
    std_val = np.std(data[mask])

    # Z-score normalization
    data_normalized = (data - mean_val) / std_val

    # Save normalized data
    normalized_img = nib.Nifti1Image(data_normalized, img.affine, img.header)
    output_file = os.path.join(feat_dir, 'stats/cope1_normalized.nii.gz')
    nib.save(normalized_img, output_file)

    print(f"Normalized data saved: {output_file}")

######################################################################################################################
# New normalization:
import os
import glob
import nibabel as nib
import numpy as np

# Set the base path where all subject directories are stored.
base_path = '/.../EMBARC/data/03_FSL_FEAT/Whole-data/'

# Find all .feat directories (assuming they end with .feat).
# This pattern looks for folders like sub-*/ses-*/FEAT_results_*/<subject_session>.feat
feat_folders = glob.glob(os.path.join(base_path, 'sub-*', 'ses-*', 'FEAT_results_5', '*.feat'))

# Loop over each .feat directory (i.e. each subject/session)
for feat_dir in feat_folders:
    # Define file paths for cope2, mask, and mean_func
    cope_path = os.path.join(feat_dir, 'stats', 'cope1.nii.gz')
    mask_path = os.path.join(feat_dir, 'mask.nii.gz')
    mean_func_path = os.path.join(feat_dir, 'mean_func.nii.gz')

    # Check if all files exist
    if not (os.path.exists(cope_path) and os.path.exists(mask_path) and os.path.exists(mean_func_path)):
        print(f"Missing one or more files in {feat_dir}. Skipping...")
        continue

    # Load the NIfTI images using nibabel
    cope_img = nib.load(cope_path)
    mask_img = nib.load(mask_path)
    mean_func_img = nib.load(mean_func_path)

    # Get the image data as numpy arrays
    cope_data = cope_img.get_fdata()
    mask_data = mask_img.get_fdata()
    mean_func_data = mean_func_img.get_fdata()

    # Compute the mean of mean_func only for voxels where the mask is nonzero
    nonzero_voxels = mask_data > 0
    if not np.any(nonzero_voxels):
        print(f"No nonzero voxels in mask for {feat_dir}. Skipping...")
        continue

    baseline_mean = np.mean(mean_func_data[nonzero_voxels])

    # Avoid division by zero
    if baseline_mean == 0:
        print(f"Baseline mean is zero in {feat_dir}. Skipping normalization.")
        continue

    # Normalize cope2 by dividing by the computed mean
    normalized_cope = cope_data / baseline_mean

    # Create a new NIfTI image for the normalized cope
    normalized_img = nib.Nifti1Image(normalized_cope, affine=cope_img.affine, header=cope_img.header)

    # Define an output path for the normalized image
    output_path = os.path.join(feat_dir, 'stats', 'cope1_normalized1.nii.gz')
    nib.save(normalized_img, output_path)
    print(f"Saved normalized cope2 image for {feat_dir} to {output_path}")

####################################################################################################################
# Extract features and create csv per subject
import os
import glob
import pandas as pd
import multiprocessing
import logging
from radiomics import featureextractor

# --------------------------
# Setup Logging
# --------------------------
class NoGLCMFilter(logging.Filter):
    def filter(self, record):
        # Filter out any log messages containing the unwanted GLCM message.
        return "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average" not in record.getMessage()

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers for general logs and error logs
# fh = logging.FileHandler('logs_whole_data_scaled.txt')
fh = logging.FileHandler('logs_new_norm_cope2.txt')
fh.setLevel(logging.DEBUG)
# eh = logging.FileHandler('error_logs_whole_data_scaled.txt')
eh = logging.FileHandler('error_logs_new_norm_cope2.txt')
eh.setLevel(logging.ERROR)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
eh.setFormatter(formatter)

# Add filter to ignore the unwanted GLCM message
fh.addFilter(NoGLCMFilter())
eh.addFilter(NoGLCMFilter())

# Add handlers to logger
logger.addHandler(fh)
logger.addHandler(eh)

# --------------------------
# Setup Radiomics Extraction
# --------------------------
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA_mergedInsula.nii.gz'
# data_dir = '/data/projects/EMBARC/data/04_radiomics/Whole-data'
data_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'

if not os.path.exists(mask_path):
    logger.error(f"ERROR: Mask file not found at {mask_path}")
else:
    logger.info(f"Mask file found: {mask_path}")

roi_labels = {
    'left_amygdala': 420,
    'left_dACC': 169,
    'left_pgACC': 174,
    'left_sgACC': 177,
    'left_dlPFC': 106,
    'left_insula': 500,
    'left_hippocampus': 417,
    'right_amygdala': 404,
    'right_dACC': 379,
    'right_pgACC': 381,
    'right_sgACC': 360,
    'right_dlPFC': 310,
    'right_insula': 501,
    'right_hippocampus': 401
}

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')
extractor.enableFeatureClassByName('gldm')

# Set a custom bin width to reduce the number of gray levels in GLCM
extractor.settings['binWidth'] = 25

logger.info("Extraction settings: " + str(extractor.settings))

# --------------------------
# Multiprocessing Extraction Functions
# --------------------------
def worker_extract(q, cope_path, mask_path, label):
    try:
        result = extractor.execute(cope_path, mask_path, label=label)
        q.put((result, None))
    except Exception as e:
        q.put((None, str(e)))

def extract_in_subprocess(cope_path, mask_path, label, timeout=600):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker_extract, args=(q, cope_path, mask_path, label))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, "Timeout"
    if not q.empty():
        result, error = q.get()
        return result, error
    else:
        return None, "No result returned"

# --------------------------
# Main Extraction Loop
# --------------------------
subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub-*')))
logger.info(f"Found {len(subject_dirs)} subject directories.")
if not subject_dirs:
    logger.error("ERROR: No subject directories found. Please check your data_dir path.")

for subject_dir in subject_dirs:
    subject = os.path.basename(subject_dir)
    logger.info(f"Processing subject: {subject}")

    for session in ['ses-1', 'ses-2']:
        logger.info(f"  Processing session: {session}")
        session_results = []  # List to store results for the current subject and session

        cope_path = os.path.join(
            subject_dir, session, 'FEAT_results_5',
            f'{subject}_{session}.feat', 'stats', 'normed_cope2_groupnorm.nii.gz'
        )
        logger.info(f"    Looking for cope file: {cope_path}")

        if not os.path.exists(cope_path):
            logger.error(f"    Skipping {subject} {session}: cope2 file not found.")
            continue

        for roi_name, label_value in roi_labels.items():
            result, error = extract_in_subprocess(cope_path, mask_path, label_value, timeout=120)
            if result is not None:
                result_dict = {'subject': subject, 'session': session, 'ROI': roi_name}
                for key, value in result.items():
                    if not key.startswith('diagnostics'):
                        result_dict[key] = value
                session_results.append(result_dict)
                logger.info(f"    Processed ROI: {roi_name}")
            else:
                logger.error(f"    Error processing ROI '{roi_name}' for {subject} {session}: {error}")

        if session_results:
            df_session = pd.DataFrame(session_results)
            output_csv = os.path.join(subject_dir, session, f"radiomics_features_cope2_new_groupnorm_{subject}_{session}.csv")
            try:
                df_session.to_csv(output_csv, index=False)
                logger.info(f"    Saved features to {output_csv}")
            except Exception as e:
                logger.error(f"    ERROR saving CSV for {subject} {session}: {e}")
        else:
            logger.info(f"    No features extracted for {subject} {session}.")


#######################################################################################################################
# Debug
# Set a custom bin width to reduce the number of gray levels in GLCM
extractor.settings['binWidth'] = 40

subject = "TX0010"
session = "ses-1"
cope_path = f"/.../EMBARC/data/04_radiomics/TX_Bugs/sub-{subject}/{session}/FEAT_results_4/sub-{subject}_{session}.feat/stats/cope2.nii.gz"
label = roi_labels['left_insula']  # or another ROI that fails
try:
    result = extractor.execute(cope_path, mask_path, label=label)
    print("Extraction succeeded:", result)
except Exception as e:
    print("Extraction error:", e)

########################################
import nibabel as nib
import numpy as np

# Debug code for TX0010 ses-1 left_insula extraction
subject = "TX0010"
session = "ses-1"
cope_path = f"/.../EMBARC/data/04_radiomics/Whole-data/sub-{subject}/{session}/FEAT_results_4/sub-{subject}_{session}.feat/stats/cope2.nii.gz"
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA_mergedInsula.nii.gz'
label = roi_labels['left_insula']  # expecting label 500

# Load cope image and mask using nibabel
cope_img = nib.load(cope_path)
cope_data = cope_img.get_fdata()
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

# Check the overall image properties
print("Cope image shape:", cope_data.shape)
print("Cope image stats: min =", np.min(cope_data), "max =", np.max(cope_data), "mean =", np.mean(cope_data))
print("Mask image shape:", mask_data.shape)
print("Unique labels in mask:", np.unique(mask_data))

# Create a binary mask for the ROI (left insula)
roi_mask = (mask_data == label)
print("Number of voxels in left insula ROI:", np.sum(roi_mask))

# Optional: check cope data statistics within the ROI
if np.sum(roi_mask) > 0:
    cope_roi_values = cope_data[roi_mask]
    print("Cope ROI stats: min =", np.min(cope_roi_values), "max =", np.max(cope_roi_values), "mean =", np.mean(cope_roi_values))
else:
    print("WARNING: No voxels found for left insula (label 500) in the mask.")

# Now try running the extraction
try:
    result = extractor.execute(cope_path, mask_path, label=label)
    print("Extraction succeeded:", result)
except Exception as e:
    print("Extraction error:", e)

unique_intensities = np.unique(cope_data[roi_mask])
print("Unique intensity values in ROI:", unique_intensities)
print("Number of unique intensity values:", len(unique_intensities))

#######################################################################################################################
# Merge all subject radiomics files into one file, separated by session
import os
import glob
import pandas as pd

# Define directories
base_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'
output_dir = '/.../EMBARC/data/04_radiomics'
sessions = ['ses-1', 'ses-2']

# List to collect log messages
log_lines = []

# Loop through sessions
for session in sessions:
    # Get a sorted list of subject directories (e.g., sub-CU0011, sub-TX0198, etc.)
    subject_dirs = sorted(glob.glob(os.path.join(base_dir, 'sub-*')))
    # This dictionary maps each ROI_feature (row) to a dictionary of {subject: value}
    combined_dict = {}
    subjects_included = []  # Keep track of subjects that were successfully processed

    for subj_dir in subject_dirs:
        subject = os.path.basename(subj_dir)  # e.g., "sub-CU0011"
        # Construct the expected CSV path.
        # Example: /data/projects/EMBARC/data/04_radiomics/Whole-data/sub-CU0011/ses-1/radiomics_features_binWidth100_sub-CU0011_ses-1.csv
        csv_path = os.path.join(subj_dir, session, f"radiomics_features_cope2_new_groupnorm_{subject}_{session}.csv")
        if not os.path.exists(csv_path):
            log_lines.append(f"Missing CSV for {subject} {session}: {csv_path} not found.")
            continue
        try:
            # Read the per-subject CSV.
            # Expecting the CSV to have a column named 'ROI' and remaining columns as features.
            df = pd.read_csv(csv_path)
            if 'ROI' not in df.columns:
                log_lines.append(f"CSV format error for {subject} {session}: Missing 'ROI' column in {csv_path}.")
                continue
            # Set 'ROI' as the index so rows represent each ROI.
            df = df.set_index('ROI')
            # Create a flattened dictionary with keys "ROI_Feature" and values from the CSV.
            subj_features = {}
            for roi in df.index:
                for feat in df.columns:
                    # Skip unwanted metadata columns
                    if feat.lower() in ['subject', 'session']:
                        continue
                    key = f"{roi}_{feat}"
                    subj_features[key] = df.loc[roi, feat]
            # Record that we processed this subject.
            subjects_included.append(subject)
            # Merge this subject's features into the combined_dict.
            for key, val in subj_features.items():
                if key not in combined_dict:
                    combined_dict[key] = {}
                combined_dict[key][subject] = val
        except Exception as e:
            log_lines.append(f"Error processing CSV for {subject} {session}: {e}")

    # Create a DataFrame from the combined dictionary if there is any data.
    if combined_dict:
        combined_df = pd.DataFrame(combined_dict).transpose()  # rows=ROI_feature, columns=subjects
        # Reorder columns in sorted order if needed.
        combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
        output_csv = os.path.join(output_dir, f"radiomics_{session.replace('-', '_')}.csv")
        try:
            combined_df.to_csv(output_csv)
            print(f"Saved combined CSV for {session} to {output_csv}")
        except Exception as e:
            log_lines.append(f"Error saving combined CSV for {session}: {e}")
    else:
        log_lines.append(f"No data available for session {session}.")

# Save the log messages to radiomics_logs.txt in the output directory.
log_path = os.path.join(output_dir, "radiomics_logs_scaled1.txt")
with open(log_path, "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"Processing complete. Logs written to {log_path}")

######################################################################################################################
# Separate to SER and PLA group
import pandas as pd
import os

# Directory where the combined CSVs are stored and where output files will be saved.
input_dir = '/.../EMBARC/data/04_radiomics/Radiomics_csv/Scaled/Processed/'
sessions = ['ses_1', 'ses_2']

for sess in sessions:
    input_csv = os.path.join(input_dir, f"radiomics_{sess}.csv")
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        continue
    # Read the CSV without using the first row as header.
    # The index column (first column) is assumed to be the ROI*feature names.
    df = pd.read_csv(input_csv, header=None, index_col=0)

    # The first row (after the index) holds subject IDs.
    subject_ids = df.iloc[0]
    # The second row holds Stage1TX values.
    stage1tx = df.iloc[1]
    # The remaining rows are feature values.
    data = df.iloc[2:]
    # Set column names to the subject IDs.
    data.columns = subject_ids

    # Determine which columns correspond to PLA and SER.
    pla_subjects = subject_ids[stage1tx == 'PLA'].tolist()
    ser_subjects = subject_ids[stage1tx == 'SER'].tolist()

    df_PLA = data.loc[:, pla_subjects]
    df_SER = data.loc[:, ser_subjects]

    # Save the separated files.
    output_csv_PLA = os.path.join(input_dir, f"radiomics_{sess}_PLA.csv")
    output_csv_SER = os.path.join(input_dir, f"radiomics_{sess}_SER.csv")
    df_PLA.to_csv(output_csv_PLA)
    df_SER.to_csv(output_csv_SER)

    print(f"Saved {output_csv_PLA} and {output_csv_SER}")

#####################################################################################################################
# Save Tier 1 features from tier 2 files
import os
import pandas as pd

# List of CSV files to process
files = [
    "/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Processed/selected_ses-1_PLA.csv",
    "/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Processed/selected_ses-1_SER.csv",
    "/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Processed/selected_ses-2_PLA.csv",
    "/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Processed/selected_ses-2_SER.csv"
]

# files = ['/data/projects/EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Original/radiomics_ses_1.csv',
#          '/data/projects/EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Original/radiomics_ses_2.csv']
# Variables to keep even if they do not contain 'Mean'
special_rows = [
    "bmi",
    "masq2_score_gd",
    "shaps_total_continuous",
    "w0_score_17",
    "w1_score_17",
    "w2_score_17",
    "w3_score_17",
    "w4_score_17",
    "w6_score_17",
    "interview_age",
    "is_male",
    "is_employed",
    "is_chronic",
    "Site",
    "age",
    "age_squared",
    "gender"
]

for file_path in files:
    # Read CSV (assuming the feature/variable names are in the DataFrame index)
    df = pd.read_csv(file_path, index_col=0)

    # Create a boolean mask: True if the index contains '_Mean' OR is in special_rows
    mask = df.index.str.contains("Mean$", regex=True) | df.index.isin(special_rows)

    # Filter rows
    df_filtered = df[mask]

    # Figure out the new filename by replacing "radiomics_" with "Tier1_"
    # Example: radiomics_ses_1_PLA.csv -> Tier1_ses_1_PLA.csv
    folder = os.path.dirname(file_path)
    old_filename = os.path.basename(file_path)
    new_filename = old_filename.replace("radiomics_", "Tier1_")
    new_file_path = os.path.join(folder, new_filename)

    # Save the filtered file under the new name
    df_filtered.to_csv(new_file_path)
    print(f"Filtered file saved to: {new_file_path}")

########################################################################################################################
# Handling the missing values of BMI and HAM-D scores
import pandas as pd

# Paths to your four CSV files
csv_files = [
    "/.../EMBARC/data/06_BART_regression/Input/x/Tier_1+clnical_variables+Combat/Tier1_ses_1_PLA.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Tier_1+clnical_variables+Combat/Tier1_ses_1_SER.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Tier_1+clnical_variables+Combat/Tier1_ses_2_PLA.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Tier_1+clnical_variables+Combat/Tier1_ses_2_SER.csv"
]

bmi_row_name = "bmi"
hamd_rows = ["w0_score_17", "w1_score_17", "w2_score_17",
             "w3_score_17", "w4_score_17", "w6_score_17"]

for file_path in csv_files:
    # 1. Read CSV, using the first column as row labels
    df = pd.read_csv(file_path, index_col=0)

    # 2. Replace "+AF8-" in the index
    df.index = df.index.str.replace("+AF8-", "_", regex=False)

    print(f"Processing: {file_path}")
    print("Row labels found:", df.index.tolist())

    # 3. Impute BMI row (if it exists) with mean
    if bmi_row_name in df.index:
        bmi_series = df.loc[bmi_row_name]
        # Convert to numeric, coercing invalid entries to NaN
        bmi_series = pd.to_numeric(bmi_series, errors="coerce")
        mean_bmi = bmi_series.mean()
        # Fill missing BMI with the mean
        bmi_series = bmi_series.fillna(mean_bmi)
        # Put the row back into the DataFrame
        df.loc[bmi_row_name] = bmi_series
        print(f"  -> Filled missing BMI values with mean = {mean_bmi:.2f}")
    else:
        print(f"  -> No row named '{bmi_row_name}' found. Skipping BMI imputation.")

    # 4. Linear interpolation for HAM-D rows across columns
    for row_name in hamd_rows:
        if row_name in df.index:
            # Convert row to numeric, coercing invalid entries to NaN
            row_series = pd.to_numeric(df.loc[row_name], errors="coerce")
            # Interpolate horizontally (one row across multiple columns)
            row_series_interp = row_series.interpolate(
                method="linear",
                limit_direction="both"
            )
            df.loc[row_name] = row_series_interp
            print(f"  -> Interpolated missing values in row '{row_name}'")
        else:
            print(f"  -> No row named '{row_name}' found. Skipping interpolation.")

    # 5. Save the updated data to a new CSV (avoiding overwrite)
    out_file = file_path.replace(".csv", "_imputed.csv")
    df.to_csv(out_file, index=True)
    print(f"Imputation complete and saved to: {out_file}\n")

############################################################################################################################
# New imputation of the missing values BMI, masq, is_employed
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. Read the data
file_path = "/.../EMBARC/data/06_BART_regression/Input/x/Tier_1+clnical_variables+Combat/Tier1_ses_2_PLA_imputed.csv"
df = pd.read_csv(file_path, index_col=0)
# Remove '+AF8' and '+AC0' from the table
df.index = df.index.str.replace('+AF8', '', regex=False)
df.index = df.index.str.replace('+AC0', '', regex=False)
df.columns = df.columns.str.replace('+AF8', '', regex=False)
df.columns = df.columns.str.replace('+AC0', '', regex=False)
df = df.replace(r'\+AF8', '', regex=True)
df = df.replace(r'\+AC0', '', regex=True)

df = df.transpose() # transpose to columns to better handel the missing values

# 2. Impute BMI with the median
if "BMI" in df.columns:
    df["BMI"] = df["BMI"].astype(float)  # Ensure numeric if needed
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())


# 3. Impute is_employed with the most frequent category
#    Note: mode() returns a Series; we take the first element
df["is-employed"] = df["is-employed"].fillna(df["is-employed"].mode()[0])

# 4. Use IterativeImputer for MASQ w0 columns using the corresponding w1 columns.
#    We'll collect all the relevant columns into one DataFrame, impute, and then put them back.
masq_cols = [
        "masq2-score-aa-w0", "masq2-score-aa-w1",
        "masq2-score-ad-w0", "masq2-score-ad-w1",
        "masq2-score-gd-w0", "masq2-score-gd-w1"
]

# Subset the dataframe to these columns
df_masq = df[masq_cols]

# Create an IterativeImputer instance
iter_imp = IterativeImputer(random_state=0)

# Fit and transform the subset
df_masq_imputed = iter_imp.fit_transform(df_masq)

# Round the imputed values and convert them to integer
df_masq_imputed = np.rint(df_masq_imputed).astype(int)

# Replace the original columns with the imputed values
df[masq_cols] = df_masq_imputed

# 5. Count the missing values for each column.
#    Note: df.iloc[:25] will select the first 25 rows (index 0 to 24).
missing_per_column = df.isna().sum()
print("Missing values in the first 25 rows (by column):")
print(missing_per_column)
df = df.transpose() # transpose back to rows

# 5. (Optional) Save the imputed dataset to a new CSV
out_file = file_path.replace(".csv", "_imputed.csv")
df.to_csv(out_file, index=True)

###################################################################################################################
# Select the subject from subject list
import pandas as pd

# Example: your subject list CSV has a column 'subject' with values like 'CU0011', 'MG0021', etc.
subjects_df = pd.read_csv('/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Original/subject-list_PLA_ses-1.xlsx')  # or header=None if it has no header
subjects_df.columns = subjects_df.columns.str.replace('+AF8', '', regex=False)
subject_names = subjects_df['subject-id'].tolist()

# Add the 'sub-' prefix to each subject in the list
subject_names_with_prefix = ["sub-" + s for s in subject_names]

# Read the features CSV where columns are like 'sub-CU0011', 'sub-MG0021', etc.
features_df = pd.read_csv('/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Original/radiomics_ses_1.csv')

# Subset only the columns that match our new list
selected_df = features_df[subject_names_with_prefix]

# Save to a new CSV
selected_df.to_csv('/.../EMBARC/data/04_radiomics/Radiomics_csv/04_New_normalization/Radiomics/Processed/selected_ses-1_PLA.csv', index=False)

#########################################################################################################################
# Print all the subject that has cope 2 file and save the one that missing ses-1/ses-2
import os
import pandas as pd
import csv


def get_subjects_with_cope2(base_dir):
    """
    Walk through the base directory and return a sorted list of subjects (with 'sub-' prefix)
    that have a cope2.nii.gz file in a FEAT_results_5 folder for session 1 only.
    """
    subjects = set()
    for root, dirs, files in os.walk(base_dir):
        # Check if we're in a FEAT_results_5 folder and if cope2.nii.gz exists
        if 'FEAT_results_5' in root and 'cope2.nii.gz' in files:
            # Split the path to extract subject and session information
            parts = root.split(os.sep)
            subject = next((p for p in parts if p.startswith('sub-')), None)
            session = next((p for p in parts if p.startswith('ses-')), None)
            # Only add subject if the session is exactly 'ses-1'
            if subject and session == 'ses-2':
                subjects.add(subject)
    return sorted(subjects)


def get_subject_list_B(excel_path):
    """
    Read the Excel file and return a list of subject IDs from the 'subject_id' column.
    Expected format in Excel: 'CU0011', 'CU0014', etc.
    """
    df = pd.read_excel(excel_path)
    list_B = df['subject_id'].dropna().astype(str).tolist()
    return list_B


def compare_lists_and_write_csv(list_B, list_A, output_csv):
    """
    Compare subject IDs from list_B (from the Excel file) with list_A (subjects with cope2 file in ses-1).
    Creates a CSV with two columns: 'subject id' and 'Missing fMRI in ses-1'.
    The 'Missing fMRI in ses-1' column is marked as "Missing" when a subject (after adding 'sub-' prefix)
    is in list_B but not in list_A.
    """
    rows = []
    for subj in list_B:
        # Add the 'sub-' prefix to match the format in list_A.
        subj_prefixed = f"sub-{subj}"
        missing_flag = "Missing" if subj_prefixed not in list_A else ""
        rows.append({
            "subject id": subj,
            "Missing fMRI in ses-2": missing_flag
        })

    # Write the comparison results to a CSV file.
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["subject id", "Missing fMRI in ses-2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV file '{output_csv}' created successfully.")


def main():
    # Get List A: subjects with cope2 file in session 1 only
    base_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'
    list_A = get_subjects_with_cope2(base_dir)
    print("Subjects with a cope2 file in FEAT_results_5 for ses-2:")
    print(list_A)

    # Get List B: subject IDs from the Excel file
    excel_path = '/.../EMBARC/data/06_BART_regression/Input/x/Previous_study_result/Clinical_scale.xlsx'
    list_B = get_subject_list_B(excel_path)

    # Create CSV that compares the two lists
    output_csv = '/.../EMBARC/data/03_FSL_FEAT/missing_fMRI_ses-2.csv'
    compare_lists_and_write_csv(list_B, list_A, output_csv)


if __name__ == '__main__':
    main()

#######################################################################################################################
# Extract features based on the label list
import logging
import os
import glob
import pandas as pd
import multiprocessing
from radiomics import featureextractor

# --------------------------
# Setup Logging
# --------------------------
class NoGLCMFilter(logging.Filter):
    def filter(self, record):
        # Filter out any log messages containing the unwanted GLCM message.
        return "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average" not in record.getMessage()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('logs_cope2_new2.txt')
fh.setLevel(logging.DEBUG)
eh = logging.FileHandler('error_logs_cope2_new2.txt')
eh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
eh.setFormatter(formatter)

fh.addFilter(NoGLCMFilter())
eh.addFilter(NoGLCMFilter())

logger.addHandler(fh)
logger.addHandler(eh)

# --------------------------
# Setup Radiomics Extraction (All Firstorder Features)
# --------------------------
mask_path = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/combined_mask_400_32_alignedA_mergedInsula.nii.gz'
data_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data'

if not os.path.exists(mask_path):
    logger.error(f"ERROR: Mask file not found at {mask_path}")
else:
    logger.info(f"Mask file found: {mask_path}")

# Read ROI names from the file and build the dictionary.
label_file = '/.../EMBARC/data/04_radiomics/mask/Processed_mask/List_of_label/Combined_432_Label_of_regions.txt'
try:
    with open(label_file, 'r') as f:
        lines = f.readlines()
    # Remove extra whitespace/newlines.
    roi_names = [line.strip() for line in lines if line.strip()]
    # Process lines if they include a "b'" prefix.
    roi_names = [name[2:-1] if name.startswith("b'") and name.endswith("'") else name for name in roi_names]
    # Build dictionary: key = ROI name from file, value = integer label (starting at 1)
    roi_labels = {roi_names[i]: i+1 for i in range(len(roi_names))}
    logger.info(f"Loaded {len(roi_labels)} ROI labels from file.")
except Exception as e:
    logger.error(f"Failed to load ROI labels from {label_file}: {e}")
    roi_labels = {}

# Create the extractor and disable all features.
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()

# Enable all features in the firstorder feature class.
extractor.enableFeatureClassByName('firstorder')
extractor.settings['binWidth'] = 25
# Optionally, lower the minimum voxel threshold to force computation on small ROIs.
extractor.settings['minimumVoxelCount'] = 1

logger.info("Extraction settings: " + str(extractor.settings))

# --------------------------
# Multiprocessing Extraction Functions
# --------------------------
def worker_extract(q, cope_path, mask_path, label):
    try:
        result = extractor.execute(cope_path, mask_path, label=label)
        q.put((result, None))
    except Exception as e:
        q.put((None, str(e)))

def extract_in_subprocess(cope_path, mask_path, label, timeout=600):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker_extract, args=(q, cope_path, mask_path, label))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, "Timeout"
    if not q.empty():
        result, error = q.get()
        return result, error
    else:
        return None, "No result returned"

# --------------------------
# Main Extraction Loop for Whole Data
# --------------------------
subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub-*')))
logger.info(f"Found {len(subject_dirs)} subject directories.")
if not subject_dirs:
    logger.error("ERROR: No subject directories found. Please check your data_dir path.")

all_results = []

for subject_dir in subject_dirs:
    subject = os.path.basename(subject_dir)
    logger.info(f"Processing subject: {subject}")

    for session in ['ses-1', 'ses-2']:
        logger.info(f"  Processing session: {session}")
        session_results = []  # To store results for the current subject and session

        cope_path = os.path.join(
            subject_dir, session, 'FEAT_results_5',
            f'{subject}_{session}.feat', 'stats', 'normed_cope2_groupnorm.nii.gz'
        )
        logger.info(f"    Looking for cope file: {cope_path}")

        if not os.path.exists(cope_path):
            logger.error(f"    Skipping {subject} {session}: cope file not found.")
        else:
            logger.info(f"    Found cope file for {subject} {session}: {cope_path}")

        for roi_name, label_value in roi_labels.items():
            result, error = extract_in_subprocess(cope_path, mask_path, label_value, timeout=120)
            if result is not None:
                result_dict = {'subject': subject, 'session': session, 'ROI': roi_name}
                # Use a key from the computed firstorder features.
                # Depending on your PyRadiomics version, the ROI-specific keys may vary.
                # Try 'firstorder_Mean' if available.
                mean_key = 'original_firstorder_Mean'
                result_dict['Mean'] = result.get(mean_key, None)
                session_results.append(result_dict)
                logger.info(f"    Processed {roi_name}")
            else:
                logger.error(f"    Error processing {roi_name} for {subject} {session}: {error}")

        if session_results:
            df_session = pd.DataFrame(session_results)
            output_csv = os.path.join(subject_dir, session, f"radiomics_firstorder_cope2_normalized2_{subject}_{session}.csv")
            try:
                df_session.to_csv(output_csv, index=False)
                logger.info(f"    Saved features to {output_csv}")
            except Exception as e:
                logger.error(f"    ERROR saving CSV for {subject} {session}: {e}")
            all_results.extend(session_results)
        else:
            logger.info(f"    No features extracted for {subject} {session}.")

# Optionally, save combined results for all subjects:
if all_results:
    df_all = pd.DataFrame(all_results)
    combined_csv = os.path.join(data_dir, "radiomics_firstorder_cope2_normalized2_all.csv")
    try:
        df_all.to_csv(combined_csv, index=False)
        logger.info(f"Saved combined features to {combined_csv}")
    except Exception as e:
        logger.error(f"ERROR saving combined CSV: {e}")




# Debugging code to extract and print all firstorder features for one ROI in one subject
extractor.enableFeatureClassByName('firstorder')

# Select one ROI from the loaded dictionary (e.g., the first one)
roi_name = list(roi_labels.keys())[0]  # gets the first ROI name
label_value = roi_labels[roi_name]  # its corresponding integer label

print(f"Testing extraction for ROI: '{roi_name}' with label value: {label_value}")

subject = "CU0009"
session = "ses-1"
cope_path = f"/.../EMBARC/data/03_FSL_FEAT/Whole-data/sub-{subject}/{session}/FEAT_results_5/sub-{subject}_{session}.feat/stats/cope2_normalized1.nii.gz"

try:
    result = extractor.execute(cope_path, mask_path, label=label_value)
    print("Extraction succeeded for ROI:", roi_name)

    # Print all firstorder features from the result (keys that start with 'firstorder')
    print("Extracted firstorder features:")
    for key, value in result.items():
        if key.startswith("original_firstorder"):
            print(f"{key}: {value}")

except Exception as e:
    print("Extraction error for ROI:", roi_name, "with label value:", label_value)
    print(e)

#######################################################################################################################
#############################################################################################################
# Imputation of y
# Path to your Excel file
file_path = '/.../EMBARC/data/06_BART_regression/Input/y/y_imputation.xlsx'
# Read the sheets.
# Note: In pandas, sheet indexing is 0-based. Here, sheet 2, 3, and 4 mean sheet indices 1, 2, and 3.
df_sheet2 = pd.read_excel(file_path, sheet_name=0)  # Sheet 2
df_sheet3 = pd.read_excel(file_path, sheet_name=1)  # Sheet 3
df_sheet4 = pd.read_excel(file_path, sheet_name=2)  # Sheet 4
# Merge sheet 3 scores into sheet 2 based on src_subject_id.
# We use a left join so that all subjects from sheet 2 are retained.
merged_df = pd.merge(df_sheet2, df_sheet3, on='src_subject_id', how='left', suffixes=('', '_sheet3'))
# Merge sheet 4 scores into the already merged dataframe.
merged_df = pd.merge(merged_df, df_sheet4, on='src_subject_id', how='left', suffixes=('', '_sheet4'))
# Write the merged data to a new Excel file.
merged_df.to_excel('/.../EMBARC/data/06_BART_regression/Input/y/merged_output.xlsx', index=False)

# Imputation
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# 1. Read in your data
df = pd.read_excel('/.../EMBARC/data/06_BART_regression/Input/y/Imputation/merged_output.xlsx')

# 2. (Optional) Rename the column if needed.
#    If your CSV file has 'w8_score_17' and you want to call it 'w8_xcore_17':
df.rename(columns={'w8_score_17': 'w8_xcore_17'}, inplace=True)

# 3. Ensure categorical variables are numeric (example for sex-w0).
# df['sex-w0'] = df['sex-w0'].map({'M': 0, 'F': 1})

# 4. Specify all columns used for imputation
cols_for_imputation = [
    'w8_xcore_17',            # The column to impute
    'w0_score_17',
    'w1_score_17',
    'w2_score_17',
    'w3_score_17',
    'w4_score_17',
    'w6_score_17',
    'interview_age-w0',
    'sex-w0',
    'edutot-w0',
    'shaps_total_continuous-w0',
    'masq2_score_aa-w0',
    'masq2_score_ad-w0',
    'masq2_score_gd-w0',
    'ss_vocabularyrawscore',
    'ss_matrixreasoningrawscore',
    'qids_eval_total'
]

# 5. Create a DataFrame with only these columns
df_for_imputation = df[cols_for_imputation].copy()

# 6. Create and configure the IterativeImputer (MICE-like)
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    sample_posterior=True,   # draws from posterior for each iteration
    max_iter=10,
    random_state=0
)

# 7. Fit and transform to impute missing values
imputed_array = imputer.fit_transform(df_for_imputation)

# 8. Convert to a new DataFrame
df_imputed = pd.DataFrame(
    imputed_array,
    columns=cols_for_imputation,
    index=df_for_imputation.index
)

# 9. Save the imputed column as a separate variable
imputed_y = df_imputed['w8_xcore_17'].copy()

# Save as CSV (without the index column)
imputed_y.to_csv('/.../EMBARC/data/06_BART_regression/Input/y/Imputation/y_imputation.xlsx', index=False)


##########################################################################################################################
# exclude the highest mean FD:
import os
import glob
import numpy as np

# Define the base directory where subject folders are located.
base_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data/'

# Create a search pattern to match FD files.
search_pattern = os.path.join(base_dir, 'sub-*/ses-2/*_framewise_displacement.txt')

# Get a list of all matching FD file paths.
fd_files = glob.glob(search_pattern)

# Dictionary to hold the mean FD for each subject/file.
mean_fd_dict = {}

# Loop over each file and compute the mean FD.
for fd_file in fd_files:
    with open(fd_file, 'r') as f:
        # Read lines and attempt to convert each to float, skipping any that fail.
        values = []
        for line in f:
            line = line.strip()
            try:
                value = float(line)
                values.append(value)
            except ValueError:
                # Print a message or log the skipped line if necessary.
                continue
        if values:
            mean_fd = np.mean(values)
            mean_fd_dict[fd_file] = mean_fd

# Sort the results by mean FD (ascending order).
sorted_means = sorted(mean_fd_dict.items(), key=lambda x: x[1])
num_subjects = len(sorted_means)

if num_subjects <= 11:
    print("Not enough subjects to exclude 11 highest FD subjects.")
else:
    # Exclude the 11 subjects with the highest FD.
    kept_subjects = sorted_means[:num_subjects - 11]
    excluded_subjects = sorted_means[num_subjects - 11:]

    # The threshold is the maximum mean FD among the kept subjects.
    threshold_fd = kept_subjects[-1][1]
    print("Threshold FD for exclusion: {:.5f}".format(threshold_fd))

    # Optionally, print details about the excluded subjects.
    print("\nExcluded subjects (file path, mean FD):")
    for file_path, fd in excluded_subjects:
        print(f"{file_path}: {fd:.5f}")
######################################################################################################################
# Rename the feature columns
import pandas as pd

# Example file paths
input_csv = "/.../EMBARC/data/04_radiomics/Radiomics_csv/03_whole_ROIs/Processed/selected_ses-1_PLA.csv"
output_csv = "/.../EMBARC/data/04_radiomics/Radiomics_csv/03_whole_ROIs/Processed/selected_ses-1_PLA_renamed.csv"

# 1. Read CSV
df = pd.read_csv(input_csv)

# 2. Convert to string (in case of bytes/mixed types)
df["FeatureName"] = df["FeatureName"].astype(str)

# 3. Remove all occurrences of "+AC0" and "+AF8"
df["FeatureName"] = (
    df["FeatureName"]
    .str.replace("+AC0", "", regex=False)
    .str.replace("+AF8", "", regex=False)
)

# Single-step regex to remove:
#  1) Leading digits + colon + optional spaces (^\d+:\s*)
#  2) The literal b'
#  3) The literal '-Mean at the end
#  And keep whatever is in between as \1
df["FeatureName"] = df["FeatureName"].str.replace(
    r"^\d+:\s*b'(.*)'-Mean$",
    r"\1",
    regex=True
)

# 5. Save the updated DataFrame
df.to_csv(output_csv, index=False)
print("Finished cleaning FeatureName column. Saved to:", output_csv)

##############################################################################################################################
# Imputation of X
import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Load the CSV file.
input_csv = '/.../EMBARC/data/06_BART_regression/Input/x/All_ROIs/selected_ses-1_PLA_clinical.csv'
X = pd.read_csv(input_csv)

# Remove "+AF8" and "+AC0" from the first row.
X.iloc[0] = X.iloc[0].astype(str).str.replace('+AF8', '', regex=False).str.replace('+AC0', '', regex=False)

# Optionally, print the updated first row to verify the changes.
print(X.iloc[0])


# Impute missing values in the 'BMI' column using the median.
if 'BMI' in X.columns:
    # Convert the 'BMI' column to numeric, forcing non-numeric values to NaN.
    X['BMI'] = pd.to_numeric(X['BMI'], errors='coerce')
    # Calculate the median (ignoring NaN) and fill missing values.
    bmi_median = X['BMI'].median()
    X['BMI'] = X['BMI'].fillna(bmi_median)
    print("Missing values in 'BMI' column have been imputed using median value:", bmi_median)
else:
    print("Column 'BMI' not found in X.")

# Impute missing values in the 'is-employed' column using the majority class.
if 'is-employed' in X.columns:
    majority_class = X['is-employed'].mode()[0]
    X['is-employed'] = X['is-employed'].fillna(majority_class)
    print("Missing values in 'is-employed' column have been imputed using majority class:", majority_class)
else:
    print("Column 'is-employed' not found in X.")

# Impute missing values for the "masq2-score-*-w0" columns using IterativeImputer.
cols_w0 = ["masq2-score-aa-w0", "masq2-score-ad-w0", "masq2-score-gd-w0"]
cols_w1 = ["masq2-score-aa-w1", "masq2-score-ad-w1", "masq2-score-gd-w1"]
cols_impute = cols_w0 + cols_w1

# Check if all necessary columns exist in X.
if all(col in X.columns for col in cols_impute):
    imputer = IterativeImputer(random_state=0)
    # Perform imputation on the combined set of columns.
    imputed_data = imputer.fit_transform(X[cols_impute])
    # Update only the w0 columns with the imputed values.
    X[cols_w0] = imputed_data[:, :len(cols_w0)]
    print("Missing values in wave0 masq2-score columns have been imputed using IterativeImputer.")
else:
    missing_cols = [col for col in cols_impute if col not in X.columns]
    print("The following required columns are missing for iterative imputation:", missing_cols)

# Check for any missing values in X.
missing_total = X.isnull().sum().sum()
if missing_total == 0:
    print("No missing values found in X.")
else:
    print(f"There are still {missing_total} missing values in X. Missing counts per column:")
    print(X.isnull().sum())

# Print the shape of the final DataFrame.
n_samples = X.shape[0]
n_features = X.shape[1]
print('After filtering by medication: n_samples:', n_samples, '; n_features:', n_features, '\n')

# Save the imputed DataFrame as a new CSV file.
imputed_csv_path = os.path.join('/.../EMBARC/data/06_BART_regression/Input/x/All_ROIs', "imputed_x_PLA.csv")
X.to_csv(imputed_csv_path, index=True)
print(f"Imputed X saved to {imputed_csv_path}")

#######################################################################################################################
# Make a list of manufacturers' name
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the base directory where your CU data is located.
base_dir = "/.../EMBARC/data/02_fMRIprep_preprocessing/fMRIprep_running_env/UM/data_bids/"

# Use glob to search for all JSON files under sub-*/ses-*/func/ directories.
json_files = glob.glob(os.path.join(base_dir, "sub-*/ses-*/func/*task-ert_bold.json"), recursive=True)
print(f"Found {len(json_files)} JSON files.")

data_list = []

for file in json_files:
    try:
        with open(file, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # Extract subject ID from the file path. Here we assume the file name starts with 'sub-...'
    subject = os.path.basename(file).split("_")[0]  # e.g., "sub-CU0116"

    # Get manufacturer and model; if not found, record as "NA"
    manufacturer = json_data.get("Manufacturer", "NA")
    model_name = json_data.get("ManufacturersModelName", "NA")

    data_list.append({
        "subject": subject,
        "Manufacturer": manufacturer,
        "ManufacturersModelName": model_name
    })

# Create a DataFrame from the collected data.
df = pd.DataFrame(data_list)
csv_filename = "UM_subject_manufacturer_info.csv"
df.to_csv(csv_filename, index=False)
print(f"CSV file created: {csv_filename}")

# -----------------------------
# Plot histogram for Manufacturer counts.
manufacturer_counts = df['Manufacturer'].value_counts()
plt.figure()
manufacturer_counts.plot(kind='bar')
plt.title("Distribution of Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
manufacturer_hist_filename = "UM_manufacturer_histogram.png"
plt.savefig(manufacturer_hist_filename)
plt.show()
print(f"Manufacturer histogram saved as {manufacturer_hist_filename}")

# -----------------------------
# Plot histogram for ManufacturersModelName counts.
model_counts = df['ManufacturersModelName'].value_counts()
plt.figure()
model_counts.plot(kind='bar')
plt.title("Distribution of Manufacturers Model Name")
plt.xlabel("ManufacturersModelName")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
model_hist_filename = "UM_manufacturer_model_histogram.png"
plt.savefig(model_hist_filename)
plt.show()
print(f"Model histogram saved as {model_hist_filename}")

##############################################################################################################
# Create mean func file per sites
import glob
import os
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Define the output directory and create it if it doesn't exist.
output_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data/meanact/ses-2'
os.makedirs(output_dir, exist_ok=True)

# Define the file pattern to include only files under FEAT_results_5.
file_pattern = '/.../EMBARC/data/03_FSL_FEAT/Whole-data/sub-*/ses-2/FEAT_results_5/*/mean_func.nii.gz'
files = glob.glob(file_pattern)

# Prepare a dictionary to hold file lists for each group.
groups = {
    "CU0009-CU0057": [],
    "CU0058-CU0062": [],
    "CU0064-CU0135": [],
    "TX": [],
    "MG": [],
    "UM": []
}


# Helper function to extract the subject ID (assumes it's the string after "sub-").
def extract_subject_id(filepath):
    parts = filepath.split(os.sep)
    for part in parts:
        if part.startswith("sub-"):
            return part.replace("sub-", "")
    return None


# Sort the files into groups based on the subject identifier.
for f in files:
    subj = extract_subject_id(f)
    if subj is None:
        continue
    # For CU subjects, filter based on numeric value.
    if subj.startswith("CU"):
        try:
            num = int(subj[2:])  # Convert the numeric part to int.
        except ValueError:
            continue
        if 9 <= num <= 57:
            groups["CU0009-CU0057"].append(f)
        elif 58 <= num <= 62:
            groups["CU0058-CU0062"].append(f)
        elif 64 <= num <= 135:
            groups["CU0064-CU0135"].append(f)
    # For other centers, check the prefix.
    elif subj.startswith("TX"):
        groups["TX"].append(f)
    elif subj.startswith("MG"):
        groups["MG"].append(f)
    elif subj.startswith("UM"):
        groups["UM"].append(f)

# Report the number of files found for each group.
for group, file_list in groups.items():
    print(f"Group {group}: {len(file_list)} file(s) found.")

# For each group, load images, resample if needed, compute the voxel-wise mean, and save the new image.
for group, file_list in groups.items():
    if not file_list:
        print(f"No files for group {group}. Skipping average computation.")
        continue

    # Load the reference image (first image in the group)
    ref_img = nib.load(file_list[0])
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine

    data_list = []
    for f in file_list:
        img = nib.load(f)
        # Check if image shape matches the reference shape.
        if img.shape != ref_shape:
            print(f"Resampling image {f} from shape {img.shape} to {ref_shape}.")
            img = resample_from_to(img, (ref_shape, ref_affine))
        data_list.append(img.get_fdata())

    # Convert list of arrays into a 4D array and compute the voxel-wise mean.
    data_array = np.array(data_list)
    avg_data = np.mean(data_array, axis=0)

    # Create a new NIfTI image using the reference affine.
    avg_img = nib.Nifti1Image(avg_data, affine=ref_affine)

    # Define the output filename and path.
    out_filename = f'avg_mean_func_{group}.nii.gz'
    out_filepath = os.path.join(output_dir, out_filename)
    nib.save(avg_img, out_filepath)
    print(f"Saved average image for group {group} as {out_filepath}")

###########################################################################################################
# Normalize based on the sites
import glob
import os
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Directory where group-average mean_func images are stored.
avg_mean_dir = '/.../EMBARC/data/03_FSL_FEAT/Whole-data/meanact/ses-2'

# Glob pattern to find all cope2 files for all subjects.
cope2_files = glob.glob(
    '/.../EMBARC/data/03_FSL_FEAT/Whole-data/sub-*/ses-2/FEAT_results_5/sub-*_ses-2.feat/stats/cope2.nii.gz'
)

print(f"Found {len(cope2_files)} cope2 files.")


# Helper function to extract the subject identifier from the file path.
def extract_subject_id(filepath):
    parts = filepath.split(os.sep)
    for part in parts:
        if part.startswith("sub-"):
            return part.replace("sub-", "")
    return None


# Process each cope2 file.
for cope2_path in cope2_files:
    subj = extract_subject_id(cope2_path)
    if subj is None:
        print(f"Could not extract subject id from {cope2_path}. Skipping.")
        continue

    # Determine group based on the subject id.
    group = None
    if subj.startswith("CU"):
        try:
            num = int(subj[2:])  # Extract numeric part.
        except ValueError:
            print(f"Invalid subject number for {subj}. Skipping.")
            continue
        if 9 <= num <= 57:
            group = "CU0009-CU0057"
        elif 58 <= num <= 62:
            group = "CU0058-CU0062"
        elif 64 <= num <= 135:
            group = "CU0064-CU0135"
    elif subj.startswith("TX"):
        group = "TX"
    elif subj.startswith("MG"):
        group = "MG"
    elif subj.startswith("UM"):
        group = "UM"

    if group is None:
        print(f"Could not determine group for subject {subj}. Skipping.")
        continue

    # Construct the path to the group-average mean_func image.
    avg_mean_func_path = os.path.join(avg_mean_dir, f"avg_mean_func_{group}.nii.gz")
    if not os.path.exists(avg_mean_func_path):
        print(f"Group-average mean_func not found for group {group} at {avg_mean_func_path}. Skipping subject {subj}.")
        continue

    print(f"Processing subject {subj} (group: {group})")

    # Load the cope2 image.
    cope2_img = nib.load(cope2_path)
    cope2_data = cope2_img.get_fdata()

    # Load the group-average mean_func image.
    avg_mean_img = nib.load(avg_mean_func_path)
    avg_mean_data = avg_mean_img.get_fdata()

    # If shapes don't match, resample the group-average image to match cope2 dimensions.
    if cope2_data.shape != avg_mean_data.shape:
        print(
            f"Resampling group-average mean_func for subject {subj} from shape {avg_mean_data.shape} to {cope2_data.shape}")
        avg_mean_img = resample_from_to(avg_mean_img, (cope2_data.shape, cope2_img.affine))
        avg_mean_data = avg_mean_img.get_fdata()

    # Avoid division by zero by replacing zeros in the group-average data with a small constant.
    epsilon = 1e-8
    avg_mean_data[avg_mean_data == 0] = epsilon

    # Normalize the cope2 data voxel-wise.
    normed_data = cope2_data / avg_mean_data

    # Create a new NIfTI image with the normalized data.
    normed_img = nib.Nifti1Image(normed_data, affine=cope2_img.affine)

    # Save the normalized image in the same stats folder.
    normed_path = os.path.join(os.path.dirname(cope2_path), 'normed_cope2_groupnorm.nii.gz')
    nib.save(normed_img, normed_path)

    print(f"Saved normalized image for subject {subj} (group {group}) at {normed_path}\n")
########################################################################################################################
# Plot first order statistics based on sites
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings

# Step 1: Read the table from a file
df = pd.read_csv('/.../EMBARC/data/06_BART_regression/Output/Feature_distribution/06_ComBat_Scaled/ses-1_SER/X4.csv')  # Change 'data.csv' to your file path
print("Original DataFrame:")
print(df.head())
# Define your output directory.
output_dir = "/.../EMBARC/data/06_BART_regression/Output/Feature_distribution/06_ComBat_Scaled/ses-1_SER/"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Clean Column and Index Labels
# Convert labels to string first and then remove the unwanted substrings.
df.columns = (
    df.columns.astype(str)
    .str.replace(r'\+AC0', '', regex=True)
    .str.replace(r'\+AF8', '', regex=True)
)
df.index = (
    df.index.astype(str)
    .str.replace(r'\+AC0', '', regex=True)
    .str.replace(r'\+AF8', '', regex=True)
)

# Step 3: Clean the DataFrame values
def clean_value(x):
    # Check if the value is a string. If it is, remove unwanted substrings.
    if isinstance(x, str):
        cleaned = x.replace("+AC0", "").replace("+AF8", "")
        # Optionally, try converting the cleaned string back to a numeric type.
        try:
            return pd.to_numeric(cleaned)
        except ValueError:
            return cleaned
    else:
        return x

# Apply the cleaning function to each element in the DataFrame.
df_cleaned = df.applymap(clean_value)

print("\nTransposed DataFrame (after cleaning both labels & values):")
print(df_cleaned.head())


# Step 4: Transpose the DataFrame
df_transposed = df_cleaned.transpose()
print("\nTransposed DataFrame:")
print(df_transposed.head())

# Print the row to confirm it contains the feature names
print("Feature names row:")
# print(df_transposed.loc["Unnamed: 0"])
print(df_transposed.loc["0"])

# Set the column names to the values of that row, then remove the row from the DataFrame
# df_transposed.columns = df_transposed.loc["Unnamed: 0"]
# df_transposed = df_transposed.drop("Unnamed: 0")
df_transposed.columns = df_transposed.loc["0"]
df_transposed = df_transposed.drop("0")

print("\nDataFrame with corrected column labels:")
print(df_transposed.head())


# Step 5: For each feature (column) that contains "firstorder", plot a histogram.
# Optionally, convert firstorder feature columns to numeric.
for col in df_transposed.columns:
    if "firstorder" in col:
        df_transposed[col] = pd.to_numeric(df_transposed[col], errors='coerce')

# Loop through each column whose name includes "firstorder"
for feature in df_transposed.columns:
    if "firstorder" in feature:
        # Create the plot title by removing "original-firstorder-"
        plot_title = feature.replace("original-firstorder-", "")

        plt.figure(figsize=(7, 5))
        ax = sns.histplot(
            data=df_transposed,
            x=feature,
            hue="Site",
            bins=30,
            element="step",
            alpha=0.7
        )

        plt.title(plot_title)
        plt.xlabel("Value")
        plt.ylabel("Count")

        # Try getting the automatically generated legend handles and labels.
        handles, labels = ax.get_legend_handles_labels()

        # If handles are empty, build a custom legend using the unique 'Site' values.
        if not handles or all(h is None for h in handles):
            unique_sites = sorted(df_transposed["Site"].dropna().unique())
            # Create a color palette with as many colors as unique sites.
            palette = sns.color_palette(n_colors=len(unique_sites))
            handles = [mpatches.Patch(color=palette[i], label=site)
                       for i, site in enumerate(unique_sites)]
            ax.legend(handles=handles, title="Site", loc="upper right")
        else:
            # Otherwise, use the retrieved handles (if they come out properly).
            ax.legend(handles=handles, labels=labels, title="Site", loc="upper right")

        plt.tight_layout()

        # Save the figure to the output directory.
        file_path = os.path.join(output_dir, f"{plot_title}.png")
        plt.savefig(file_path)
        plt.close()

#################################################################################################################################
# Create features csv with only first order features
import os
import pandas as pd

# List of CSV files to process
files = [
    "/.../EMBARC/data/06_BART_regression/Input/x/Site_normalization/Tier2a/Tier2a_selected_ses-1_SER.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Site_normalization/Tier2a/Tier2a_selected_ses-1_PLA.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Site_normalization/Tier2a/Tier2a_selected_ses-2_SER.csv",
    "/.../EMBARC/data/06_BART_regression/Input/x/Site_normalization/Tier2a/Tier2a_selected_ses-2_PLA.csv"
]

for file_path in files:
    # Read CSV (assuming the feature/variable names are in the DataFrame index)
    df = pd.read_csv(file_path, index_col=0)

    # Create a boolean mask: True if the index contains 'firstorder' (case insensitive)
    mask = df.index.str.contains("firstorder", case=False, na=False)

    # Filter rows based on the mask
    df_filtered = df[mask]

    # Define the new file name by prepending "Tier1_" to the original file name
    folder = os.path.dirname(file_path)
    original_filename = os.path.basename(file_path)
    new_filename = "Tier2c_" + original_filename  # Prefixing instead of using replace
    new_file_path = os.path.join(folder, new_filename)

    # Save the filtered DataFrame to the new file
    df_filtered.to_csv(new_file_path)
    print(f"Filtered file saved to: {new_file_path}")


