# Mancy Chen 25/11/2024

# Code for generating condition timing file (3 columns)
import pandas as pd
# Load the dataset
file_path = '/.../EMBARC/03_FSL_FEAT/Example-data/sub-CU0009_ses-1_task-ert_bold.txt'  # Replace with the correct path
df = pd.read_csv(file_path, sep='\t', engine='python')
# Define output file path
output_file = '/.../EMBARC/03_FSL_FEAT/Example-data/condition_timing.txt'  # Replace with the desired directory path

# Filter the necessary columns for creating a condition timing file
# Assuming the relevant columns are 'stim.OnsetTime', 'Condition', and 'congruency'
onset_times = df['stim.OnsetTime']
conditions = df['Condition']
congruencies = df['congruency']

# Define the condition to filter (e.g., only incongruent events)
filtered_df = df[df['Condition'] == 'incon']

# Extract onset times and set a constant duration (e.g., 2 seconds)
timing_data = []
for _, row in filtered_df.iterrows():
    onset_time = row['stim.OnsetTime']
    duration = 0.0  # Set duration for each event (adjust as needed)
    timing_data.append([onset_time, duration, 1])

# Save to a text file in FSL-compatible format
with open(output_file, 'w') as f:
    for line in timing_data:
        f.write(f"{line[0]} {line[1]} {line[2]}\n")

print(f"Condition timing file saved to {output_file}")

########################################################################################################################
# Create a proper confound file including the necessary regressor and exclude those are not needed.
import pandas as pd
import csv
# Load the confound timeseries file
file_path = '/.../EMBARC/03_FSL_FEAT/Example-data/CU0134/sub-CU0134_ses-2_task-ert_desc-confounds_timeseries.tsv'
confounds_df = pd.read_csv(file_path, sep='\t')

# Check the shape of the DataFrame to see if it loaded correctly
print(confounds_df.shape)

# Display the list of columns to understand which confound regressors are included
print(confounds_df.columns.tolist())

# Select the desired columns
selected_columns = [
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',  # Motion parameters
    # 'csf', 'csf_power2', 'white_matter',  # Physiological noise
    # 't_comp_cor_00', 't_comp_cor_01', 't_comp_cor_02',  # CompCor components
    # 'framewise_displacement'  # Motion outliers
]

# Create a new DataFrame with the selected columns
filtered_confounds = confounds_df[selected_columns]

# Check the shape of the new DataFrame to see if it filtered correctly
print(filtered_confounds.shape)

# Save the filtered confounds to a new txt file (not CSV!)
output_file_path = '/.../EMBARC/03_FSL_FEAT/Example-data/CU0134/filtered_confounds.txt'
filtered_confounds.to_csv(output_file_path, sep=' ', index=False, header=True)

########################################################################################################################
# Expand confound EVs to 397 volumes:
import numpy as np
import pandas as pd

# Define total number of volumes
num_volumes = 397

# Function to process EV file into a full-length vector
def process_ev_file(file_path, num_volumes):
    ev_data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    ev_full = np.zeros(num_volumes)
    time_points = ev_data[0].astype(int)  # First column: time points
    values = ev_data[1]                   # Second column: values
    for time_point, value in zip(time_points, values):
        if time_point < num_volumes:  # Ensure the time point is within range
            ev_full[time_point] = value
    return ev_full

# File paths for EV files
ev_file1 = "/.../EMBARC/03_FSL_FEAT/Example-data/CU0134/EV_fails_sub-CU0134.txt"
ev_file2 = "/.../EMBARC/03_FSL_FEAT/Example-data/CU0134/EV_postfails_sub-CU0134.txt"

# Process both EV files
ev1_full = process_ev_file(ev_file1, num_volumes)
ev2_full = process_ev_file(ev_file2, num_volumes)

# Combine EVs into a single array
ev_combined = np.column_stack([ev1_full, ev2_full])

# Save to a TXT file (whitespace-delimited)
output_file_path = "/.../EMBARC/03_FSL_FEAT/Example-data/CU0134/processed_confounds_EVs.txt"
np.savetxt(output_file_path, ev_combined, fmt="%.5f", delimiter=" ")

print(f"Processed confounds saved to {output_file_path}")



########################################################################################################################
# Batch copy the preprocessed files to the new directory

import os
import shutil

# Define source and destination base paths
src_base = "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env"
dest_base = "/.../EMBARC/03_FSL_FEAT/Whole-data"
log_file = "missing_files.txt"

# List of sites to process
sites = ["CU", "MG", "TX", "UM"]

# Open log file to record missing files
with open(log_file, "w") as log:
    log.write("Missing Files Log\n")
    log.write("=================\n")

    for site in sites:
        site_src = os.path.join(src_base, site, "data_output")

        # Get a list of subjects dynamically, filtering out .html files
        if not os.path.exists(site_src):
            print(f"Skipping missing site: {site_src}")
            continue

        subjects = [
            d for d in os.listdir(site_src)
            if d.startswith(f"sub-{site}") and os.path.isdir(os.path.join(site_src, d))  # Ensure it's a directory
        ]

        for sub in subjects:
            for ses in ["ses-1", "ses-2"]:
                src_file = os.path.join(site_src, sub, ses, "func",
                                        f"{sub}_{ses}_task-ert_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

                dest_folder = os.path.join(dest_base, sub, ses)
                dest_file = os.path.join(dest_folder, f"{sub}_{ses}_task-ert_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

                if os.path.isfile(src_file):  # Check if the file exists before copying
                    os.makedirs(dest_folder, exist_ok=True)  # Ensure destination folder exists
                    shutil.copy(src_file, dest_file)
                    # print(f"Copied: {src_file} -> {dest_file}")
                else:
                    print(f"File not found: {src_file}. Logging and deleting empty destination folder if it exists...")
                    log.write(f"{sub}, {ses}\n")  # Write missing file info to log

                    # If the destination folder was created earlier and is empty, remove it
                    if os.path.exists(dest_folder) and not os.listdir(dest_folder):
                        shutil.rmtree(dest_folder)
                        print(f"Deleted empty folder: {dest_folder}")

print(f"Missing files log saved to: {log_file}")

########################################################################################################################
# Batch copy confound time series edat files
import os
import shutil

# Define source and destination base paths
src_base = "/...EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env"
dest_base = "/.../EMBARC/03_FSL_FEAT/Whole-data"
log_file = "missing_edat_files.txt"

# List of sites to process
sites = ["CU", "MG", "TX", "UM"]

# Open log file to record missing files
with open(log_file, "w") as log:
    log.write("Missing .edat Files Log\n")
    log.write("=======================\n")

    for site in sites:
        site_src = os.path.join(src_base, site, "data_bids")

        # Get a list of subjects dynamically, filtering out .html files
        if not os.path.exists(site_src):
            print(f"Skipping missing site: {site_src}")
            continue

        subjects = [
            d for d in os.listdir(site_src)
            if d.startswith(f"sub-{site}") and os.path.isdir(os.path.join(site_src, d))  # Ensure it's a directory
        ]

        for sub in subjects:
            for ses in ["ses-1", "ses-2"]:
                src_file = os.path.join(site_src, sub, ses, "func",
                                        f"{sub}_{ses}_task-ert_bold.edat")

                dest_folder = os.path.join(dest_base, sub, ses)
                dest_file = os.path.join(dest_folder, f"{sub}_{ses}_task-ert_bold.edat")

                if os.path.isfile(src_file):  # Check if the file exists before copying
                    os.makedirs(dest_folder, exist_ok=True)  # Ensure destination folder exists
                    shutil.copy(src_file, dest_file)
                    # print(f"Copied: {src_file} -> {dest_file}")
                else:
                    print(f"File not found: {src_file}")
                    log.write(f"Missing .edat -> {sub}, {ses}\n")

print(f"Missing .edat files log saved to: {log_file}")

########################################################################################################################
# Batch copy all edat files into one folder
import os
import shutil

# Define source base folder where .edat files are stored
src_base_folder = "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env"

# Define the destination folder where all .edat files will be copied
dest_folder = "/.../EMBARC/03_FSL_FEAT/Example-data/whole_edat"

# List of sites to process
sites = ["CU", "MG", "TX", "UM"]

# Ensure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Loop through all sites and find .edat files
for site in sites:
    site_src = os.path.join(src_base_folder, site, "data_bids")

    if not os.path.exists(site_src):
        print(f"Skipping missing site: {site_src}")
        continue

    # Walk through all subdirectories to find .edat files
    for root, _, files in os.walk(site_src):
        for file in files:
            if file.endswith(".edat"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)

                # Ensure unique filenames by appending subject/session if needed
                if os.path.exists(dest_file):
                    subject_session = root.split("/")[-3] + "_" + root.split("/")[-2]  # Extract sub-CU0009_ses-1
                    dest_file = os.path.join(dest_folder, f"{subject_session}_{file}")

                # Copy the .edat file
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")

print("All .edat files copied to:", dest_folder)

#######################################################################################################################
# Batch extract error trials and post error trials
# Batch extract error trials and post error trials
import os
import pandas as pd
import numpy as np

# Define input and output directories
input_folder = "/.../EMBARC/03_FSL_FEAT/whole_edat_txt"
output_base = "/.../EMBARC/03_FSL_FEAT/Whole-data"

# Loop through all .txt files in the input folder
for file in os.listdir(input_folder):
    if file.endswith(".txt"):

        # Skip reward task files
        if "task-reward_bold" in file:
            print(f"Skipping reward task file: {file}")
            continue

        file_path = os.path.join(input_folder, file)

        # Extract subject and session info from filename
        parts = file.split("_")
        if len(parts) < 4:
            print(f"Skipping {file}: Unexpected filename format.")
            continue

        subject = parts[0]  # e.g., sub-TX0031
        session = parts[1]  # e.g., ses-2

        # Define output folder
        output_folder = os.path.join(output_base, subject, session)
        os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

        try:
            df = pd.read_csv(file_path, delimiter="\t", encoding="utf-16", engine='python')
            print(df.head())  # Display the first few rows
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Check if "CombinedACC" and "stim.OnsetTime" columns exist
        if "CombinedACC" not in df.columns or "stim.OnsetTime" not in df.columns:
            print(f"Skipping {file}: Required columns missing.")
            continue

        # Identify error trials (CombinedACC == 0)
        # Compute error onsets as the difference from the first trial's onset (converted to seconds)
        first_onset = df["stim.OnsetTime"].iloc[0]
        error_trials = df[df["CombinedACC"] == 0]
        error_onsets = (error_trials["stim.OnsetTime"].values - first_onset) / 1000

        # Identify post-error trials (the first valid trial following each error trial)
        post_error_onsets_list = []
        # Loop through indices of error trials
        for error_idx in error_trials.index:
            post_idx = error_idx + 1  # the row immediately following the error trial
            # Loop until a valid onset is found or until the end of the DataFrame
            while post_idx < len(df):
                if pd.notnull(df.loc[post_idx, "stim.OnsetTime"]):
                    onset = (df.loc[post_idx, "stim.OnsetTime"] - first_onset) / 1000
                    post_error_onsets_list.append(onset)
                    break  # Found a valid post-error trial, move on to the next error trial
                else:
                    post_idx += 1
            # If no valid trial is found after an error trial, then that error trial is skipped

        post_error_onsets = np.array(post_error_onsets_list)

        # Prepare output format with three columns: timing, 1, 1
        error_data = np.column_stack((error_onsets, np.full_like(error_onsets, 1), np.ones_like(error_onsets)))
        post_error_data = np.column_stack((post_error_onsets, np.full_like(post_error_onsets, 1), np.ones_like(post_error_onsets)))

        # Save error trials onsets with three columns
        error_output_file = os.path.join(output_folder, file.replace(".txt", "_error_onsets.txt"))
        np.savetxt(error_output_file, error_data, fmt="%.3f %.1f %d")

        # Save post-error trials onsets with three columns
        post_error_output_file = os.path.join(output_folder, file.replace(".txt", "_post_error_onsets.txt"))
        np.savetxt(post_error_output_file, post_error_data, fmt="%.3f %.1f %d")

        print(f"Processed: {file_path} -> {error_output_file}, {post_error_output_file}")

print("Batch extraction complete!")


########################################################################################################################
#Batch creating conditioning timing files
import os
import pandas as pd

# Define input and output base directories
input_folder = "/.../EMBARC/03_FSL_FEAT/whole_edat_txt"
output_base = "/.../EMBARC/03_FSL_FEAT/Whole-data"

# Maximum trial duration (1s for event-related tasks)
TRIAL_DURATION = 1.0

# Loop through all .txt files in the input folder
for file in os.listdir(input_folder):
    if file.endswith(".txt"):

        # Skip reward task files
        if "task-reward_bold" in file:
            print(f"Skipping reward task file: {file}")
            continue

        file_path = os.path.join(input_folder, file)

        # Extract subject and session info from filename (e.g., sub-CU0009_ses-1_task-ert_bold.txt)
        parts = file.split("_")
        if len(parts) < 4:
            print(f"Skipping {file}: Unexpected filename format.")
            continue

        subject = parts[0]  # e.g., sub-CU0009
        session = parts[1]  # e.g., ses-1

        # Define output directory
        output_folder = os.path.join(output_base, subject, session)
        os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

        # Read the .txt file as a tab-separated file with UTF-16 encoding
        try:
            df = pd.read_csv(file_path, sep="\t", encoding="utf-16", engine='python')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Check if required columns exist
        required_columns = ["stim.OnsetTime", "congruency"]
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file}: Missing required columns.")
            continue

        # Convert milliseconds to seconds
        df["stim.OnsetTime"] = df["stim.OnsetTime"] / 1000.0

        # Reset first onset time to 0 (align all trials to first stimulus onset)
        first_onset = df["stim.OnsetTime"].min()
        df["stim.OnsetTime"] = df["stim.OnsetTime"] - first_onset

        # Define conditions
        conditions = ["ii", "ci", "ic", "cc"]

        # Group data by congruency condition
        condition_files = {cond: [] for cond in conditions}

        for _, row in df.iterrows():
            onset_time = row["stim.OnsetTime"]
            duration = TRIAL_DURATION  # Set duration for each event
            value = 1  # Default value for all trials
            condition = row["congruency"]

            if condition in conditions:
                condition_files[condition].append([onset_time, duration, value])

        # Save timing files for each condition
        for condition, trials in condition_files.items():
            output_file = os.path.join(output_folder, f"{subject}_{session}_task-{condition}_timing.txt")
            with open(output_file, "w") as f:
                for line in trials:
                    f.write(f"{line[0]:.3f} {line[1]:.1f} {line[2]}\n")
            print(f"Saved: {output_file}")

print("Batch processing complete!")

###################################################################################################################
# Batch record the failed FSL FEAT logs
import os
import re

# Define the directory where SLURM logs are stored
log_dir = "/.../EMBARC/03_FSL_FEAT/code/slurm output 3/"  # Update this with the actual log folder path
output_file = "/.../EMBARC/03_FSL_FEAT/code/slurm output 3/skipped_summary_1.txt"

# Pattern to match "Skipping" lines and their reasons
skip_pattern = re.compile(r"Skipping (sub-\w+ \w+-\d+): (.+)")
missing_pattern = re.compile(r"\s+- Missing: (.+)")

# Dictionary to store skipped subjects and their reasons
skipped_data = {}

# Loop through all log files in the directory
for log_file in os.listdir(log_dir):
    log_path = os.path.join(log_dir, log_file)

    # Ensure it's a file
    if os.path.isfile(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()

        current_subject = None  # Track the current subject being processed

        for line in lines:
            skip_match = skip_pattern.search(line)
            missing_match = missing_pattern.search(line)

            if skip_match:
                current_subject = skip_match.group(1)
                reason = skip_match.group(2)
                skipped_data[current_subject] = {"reason": reason, "missing_files": []}

            elif missing_match and current_subject:
                skipped_data[current_subject]["missing_files"].append(missing_match.group(1))

# Write the summarized output to a file
with open(output_file, "w") as out:
    for subject, info in skipped_data.items():
        out.write(f"{subject}: {info['reason']}\n")
        for file in info["missing_files"]:
            out.write(f"   - Missing: {file}\n")
        out.write("\n")

print(f"Summary saved to {output_file}")


###############
# Save all logs in one file
import os
import glob

# Directory containing the txt files
log_dir = "/.../EMBARC/code/03_FSL_FEAT/slurm output 6/397_volumes"

# Output file that will contain all the combined text
output_file = "/.../EMBARC/code/03_FSL_FEAT/slurm output 6/combined_logs_397.txt"

# Create a list of all .txt files in the directory
txt_files = glob.glob(os.path.join(log_dir, "*.out"))

# Sort the files (optional, for consistent ordering)
txt_files.sort()

with open(output_file, "w") as out:
    for txt_file in txt_files:
        # Write a header for each file
        out.write("===== " + os.path.basename(txt_file) + " =====\n")
        with open(txt_file, "r") as infile:
            out.write(infile.read())
        out.write("\n\n")  # Separate files by a blank line

print(f"All text files have been combined and saved to {output_file}")

#######################################################################################################################
# Batch create six motion parameters from fMRIprep output
import os
import pandas as pd

# Define source and destination base paths
src_base = "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env"
dest_base = "/.../EMBARC/03_FSL_FEAT/Whole-data"
log_file = "missing_confounds_files.txt"

# List of sites to process
sites = ["CU", "MG", "TX", "UM"]

# Motion parameters to extract
motion_parameters = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

# Open log file to record missing confound files
with open(log_file, "w") as log:
    log.write("Missing Confounds Files Log\n")
    log.write("=======================\n")

    for site in sites:
        site_src = os.path.join(src_base, site, "data_output")

        # Get a list of subjects dynamically
        if not os.path.exists(site_src):
            print(f"Skipping missing site: {site_src}")
            continue

        subjects = [
            d for d in os.listdir(site_src)
            if d.startswith("sub-") and os.path.isdir(os.path.join(site_src, d))
        ]

        for sub in subjects:
            for ses in ["ses-1", "ses-2"]:
                src_file = os.path.join(site_src, sub, ses, "func",
                                        f"{sub}_{ses}_task-ert_desc-confounds_timeseries.tsv")
                dest_folder = os.path.join(dest_base, sub, ses)

                if os.path.isfile(src_file):  # Check if the file exists
                    os.makedirs(dest_folder, exist_ok=True)  # Ensure destination folder exists

                    # Load the confound timeseries file
                    confounds_df = pd.read_csv(src_file, sep='\t')

                    # Check if required columns exist
                    if all(param in confounds_df.columns for param in motion_parameters):
                        for param in motion_parameters:
                            param_file = os.path.join(dest_folder, f"{sub}_{ses}_{param}.txt")
                            confounds_df[[param]].to_csv(param_file, sep=' ', index=False, header=False)
                            print(f"Saved: {param_file}")
                    else:
                        print(f"Missing motion parameters in: {src_file}")
                        log.write(f"Missing columns -> {sub}, {ses}\n")
                else:
                    print(f"File not found: {src_file}")
                    log.write(f"Missing confounds file -> {sub}, {ses}\n")

print(f"Missing confounds files log saved to: {log_file}")

######################################################################################################################
# Print the subject list of FSL FEAT output
import os


def find_subjects_with_feat_results(base_path):
    subjects_with_feat = []

    # Ensure the base path exists
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return []

    # Iterate through all subjects in the whole_data directory
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)

        # Ensure it's a directory and follows the subject naming pattern
        if os.path.isdir(subject_path) and (
                subject.startswith("sub-CU") or subject.startswith("sub-TX") or subject.startswith(
                "sub-MG") or subject.startswith("sub-UM")):
            feat_results_stats_path = os.path.join(subject_path, "ses-2", "FEAT_results_1", f"{subject}_ses-2.feat",
                                                   "stats")

            # Check if FEAT_results_1/sub-<subject>_ses-1.feat/stats directory exists
            if os.path.exists(feat_results_stats_path):
                subjects_with_feat.append(subject)
                print(subject)

    return subjects_with_feat


# Define the base path to whole_data directory
base_directory = "/.../EMBARC/03_FSL_FEAT/Whole-data"

# Find and print subjects with FEAT_results_1/sub-<subject>_ses-1.feat/stats under ses-1
subjects = find_subjects_with_feat_results(base_directory)

#######################################################################################################################
# Extract the framewise_displacement and print histogram
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define base paths for each site
sites = {
    "CU": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/CU/data_output",
    "MG": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/MG/data_output",
    "TX": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/TX/data_output",
    "UM": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/UM/data_output"
}

# Define output base path
output_base_path = "/.../EMBARC/03_FSL_FEAT/Whole-data"


# Function to process FD values for a given subject and session
def process_fd(site, subject, session):
    file_path = f"{sites[site]}/{subject}/{session}/func/{subject}_{session}_task-ert_desc-confounds_timeseries.tsv"
    output_dir = f"{output_base_path}/{subject}/{session}"
    output_file = f"{output_dir}/{subject}_{session}_framewise_displacement.txt"

    if not os.path.exists(file_path):
        return None  # Skip if file does not exist

    df = pd.read_csv(file_path, sep='\t')

    if 'framewise_displacement' not in df.columns:
        return None  # Skip if the column does not exist

    # Save framewise displacement values
    os.makedirs(output_dir, exist_ok=True)
    df[['framewise_displacement']].to_csv(output_file, index=False, header=False)

    high_fd_count = (df['framewise_displacement'] > 1.0).sum()
    return high_fd_count, subject, session


# Iterate over sites, subjects, and sessions
subjects_range = range(1, 110)  # Adjust range as needed
sessions = ["ses-1", "ses-2"]
fd_counts = []
high_fd_subjects = []

for site in sites.keys():
    for i in subjects_range:
        subject_id = f"sub-{site}{str(i).zfill(4)}"
        for session in sessions:
            result = process_fd(site, subject_id, session)
            if result is not None:
                count, subject, session = result
                fd_counts.append(count)
                if count > 30:
                    high_fd_subjects.append(f"{subject} {session}")

# Print subjects with more than 30 FD > 1.0 occurrences
print("Subjects with more than 30 frames of FD > 1.0:")
for subject in high_fd_subjects:
    print(subject)

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(fd_counts, bins=20, edgecolor='black')
plt.xlabel("Number of Frames with FD > 1.0")
plt.ylabel("Number of Subjects")
plt.title("Distribution of Framewise Displacement > 1.0 across Sites")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#######################################################################################################################
# Draw a histogram on error trials' number, print a list and save.
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Define the base directory pattern for the error onset text files
base_dir = "/.../EMBARC/03_FSL_FEAT/Whole-data/"
pattern = os.path.join(base_dir, "sub-*/ses-*/sub-*_ses-*_task-ert_bold_error_onsets.txt")

# Find all matching files
error_files = glob.glob(pattern)

# Dictionary to store subject error counts
error_counts = {}

# Process each file
for file in error_files:
    # Extract subject ID from file path
    subject_id = file.split("/")[-3]  # Assuming subject ID is at this position

    # Read the file and count rows
    with open(file, 'r') as f:
        lines = f.readlines()
        error_counts[subject_id] = len(lines)

# Convert to DataFrame for plotting
error_df = pd.DataFrame(list(error_counts.items()), columns=["Subject", "Error Count"])

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(error_df["Error Count"], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Number of Errors")
plt.ylabel("Number of Subjects")
plt.title("Distribution of Errors per Subject")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Make a list of the subject that make mistakes more than 20%
# Extract subject name and session number from file paths
subject_session_data = []

for file in error_files:
    path_parts = file.split("/")
    subject_id = path_parts[-3]  # Extract subject ID
    session_id = path_parts[-2]  # Extract session ID

    # Get error count
    with open(file, 'r') as f:
        error_count = len(f.readlines())

    # Store in list
    subject_session_data.append([subject_id, session_id, error_count])

# Convert to DataFrame
subject_session_df = pd.DataFrame(subject_session_data, columns=["Subject", "Session", "Error Count"])

# Define the total number of trials per session.
total_trials = 148  # Example: 100 trials per session
# Calculate the 20% mistake threshold
mistake_threshold = total_trials * 0.20  # For 20% mistakes
# Filter for subjects who made ≥ 20% mistakes
high_mistake_subjects_sessions = subject_session_df[subject_session_df["Error Count"] >= mistake_threshold]
print(high_mistake_subjects_sessions)
# Save the DataFrame to CSV
high_mistake_subjects_sessions.to_csv("/.../EMBARC/03_FSL_FEAT/code/high_mistake_subjects_sessions.csv", index=False)

########################################################################################################################
# Print & save the list of framewise_displacement and draw a histograme
# Define the base directory pattern for the framewise displacement files
fd_pattern = os.path.join(base_dir, "sub-*/ses-*/sub-*_ses-*_framewise_displacement.txt")

# Find all matching files
fd_files = glob.glob(fd_pattern)

# List to store subject, session, and count of framewise displacement > 1
fd_data = []

# Process each file
for file in fd_files:
    path_parts = file.split("/")
    subject_id = path_parts[-3]  # Extract subject ID
    session_id = path_parts[-2]  # Extract session ID

    # Read the file and count values > 1
    fd_values = pd.read_csv(file, header=None).values.flatten()
    high_fd_count = (fd_values > 1).sum()

    # Store data
    fd_data.append([subject_id, session_id, high_fd_count])

# Convert to DataFrame
fd_df = pd.DataFrame(fd_data, columns=["Subject", "Session", "FD > 1 Count"])

# Filter for subjects with more than 30 instances of FD > 1
high_fd_subjects_sessions = fd_df[fd_df["FD > 1 Count"] > 30]

# Save the results to CSV
fd_output_csv_path = "/.../EMBARC/03_FSL_FEAT/code/high_framewise_displacement_subjects_sessions.csv"
high_fd_subjects_sessions.to_csv(fd_output_csv_path, index=False)
print(high_fd_subjects_sessions)

################################################################################################
# Comparing the common subject with MRI and clinical scale lists
# Load the new subject lists from files
file_mri_path = "/.../EMBARC/03_FSL_FEAT/code/List_of_subjects_ses-2.txt"
file_scale_path = "/.../EMBARC/03_FSL_FEAT/code/List_of_subjects_Scale.txt"

# Read the content of the files
with open(file_mri_path, "r") as file_mri, open(file_scale_path, "r") as file_scale:
    subjects_mri = {sub.replace("sub-", "") for sub in file_mri.read().splitlines()}  # Remove 'sub-'
    subjects_scale = set(file_scale.read().splitlines())

# Find subjects missing in either list
missing_in_mri = subjects_scale - subjects_mri
missing_in_scale = subjects_mri - subjects_scale

# Create a dataframe to display the missing subjects
missing_subjects_df = pd.DataFrame({
    "Missing in MRI List": list(missing_in_mri) + [""] * (max(len(missing_in_scale) - len(missing_in_mri), 0)),
    "Missing in Scale List": list(missing_in_scale) + [""] * (max(len(missing_in_mri) - len(missing_in_scale), 0))
})

print(missing_subjects_df)
missing_subjects_df.to_csv("/.../EMBARC/03_FSL_FEAT/code/missing_subjects_clinical_scale.csv", index=False)

#######################################################################################################################
# Create FD files with 0 or 1
import os
import pandas as pd
import glob

# Define the directory containing the framewise displacement files
fd_dir = "/.../EMBARC/03_FSL_FEAT/Whole-data"

# Search for all framewise displacement files
fd_files = glob.glob(os.path.join(fd_dir, "**", "*_framewise_displacement.txt"), recursive=True)

# Process each file
output_files = []
for fd_file in fd_files:
    # Load the file
    df = pd.read_csv(fd_file, header=None)

    # Apply thresholding
    df[0] = df[0].apply(lambda x: 1 if x > 1 else 0)

    # Generate output filename
    base_name = os.path.basename(fd_file).replace("_framewise_displacement.txt", "_FD.txt")
    output_file = os.path.join(os.path.dirname(fd_file), base_name)

    # Save without header
    df.to_csv(output_file, index=False, header=False)

    # Store the output file path
    output_files.append(output_file)
#######################################################################################################################
# Create CSF and white-matter file separately
import os
import pandas as pd

# Define base paths for each site
sites = {
    "CU": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/CU/data_output",
    "MG": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/MG/data_output",
    "TX": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/TX/data_output",
    "UM": "/.../EMBARC/02_fMRIprep_preprocessing/fMRIprep_running_env/UM/data_output"
}

# Define output base path
output_base_path = "/.../EMBARC/03_FSL_FEAT/Whole-data"


# Function to extract and save CSF and White Matter values
def extract_confounds(site, subject, session):
    file_path = f"{sites[site]}/{subject}/{session}/func/{subject}_{session}_task-ert_desc-confounds_timeseries.tsv"
    output_dir = f"{output_base_path}/{subject}/{session}"
    csf_output_file = f"{output_dir}/{subject}_{session}_csf.txt"
    wm_output_file = f"{output_dir}/{subject}_{session}_white-matter.txt"

    if not os.path.exists(file_path):
        return  # Skip if file does not exist

    df = pd.read_csv(file_path, sep='\t')

    os.makedirs(output_dir, exist_ok=True)

    if 'csf' in df.columns:
        df[['csf']].to_csv(csf_output_file, index=False, header=False)

    if 'white_matter' in df.columns:
        df[['white_matter']].to_csv(wm_output_file, index=False, header=False)


# Iterate over sites, subjects, and sessions
subjects_range = range(1, 110)  # Adjust range as needed
sessions = ["ses-1", "ses-2"]

for site in sites.keys():
    for i in subjects_range:
        subject_id = f"sub-{site}{str(i).zfill(4)}"
        for session in sessions:
            extract_confounds(site, subject_id, session)

print("CSF and White Matter extraction completed.")
#######################################################################################################################
# Create one-hot encoding files
# !/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd

# Global list to record post-error files with NaN values.
post_error_nan_files = []


def process_onsets_file_to_onehot_df(input_file, TR=2.0, npts=397):
    """
    Reads an onset file (error or post-error) assumed to be whitespace-delimited with one row per event.
    For each row:
      - Convert the first column (onset time in seconds) to numeric.
      - Report a warning if any NaN values occur.
      - Divide by TR and round to obtain a volume index.
      - Create a one-hot vector (length npts) with a 1 at that index.

    Returns a DataFrame of shape (npts, number_of_events) where each column is a one-hot vector.
    """
    try:
        df = pd.read_csv(input_file, delim_whitespace=True, header=None)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))

    if df.empty:
        print(f"{input_file} is empty; skipping.")
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))

    # Convert the first column to numeric, coercing errors to NaN.
    df[0] = pd.to_numeric(df[0], errors='coerce')
    if df[0].isna().any():
        num_nan = df[0].isna().sum()
        print(f"Warning: {input_file} contains {num_nan} NaN value(s) in onset times.")
        if "post_error" in input_file.lower():
            global post_error_nan_files
            post_error_nan_files.append(input_file)
    orig_rows = len(df)
    df = df.dropna(subset=[0])
    if len(df) < orig_rows:
        print(f"{input_file}: Dropped {orig_rows - len(df)} row(s) due to non-numeric onset values.")

    # Compute volume indices.
    vol_indices = (df[0] / TR).round().astype(int)

    one_hot_list = []
    for i, idx in enumerate(vol_indices):
        vec = np.zeros(npts, dtype=int)
        if 0 <= idx < npts:
            vec[idx] = 1
        else:
            print(f"Warning: In {input_file}, row {i + 1}: computed index {idx} is out of range (0 to {npts - 1}).")
        one_hot_list.append(vec)

    if not one_hot_list:
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))

    onehot_array = np.column_stack(one_hot_list)  # Shape: (npts, number_of_events)
    onehot_df = pd.DataFrame(onehot_array)
    return onehot_df


def process_fd_file_to_onehot_df(input_file, npts=397):
    """
    Processes an FD file assumed to contain 397 values (0 or 1) separated by whitespace or newlines.
    For each index where the FD value is 1, creates a one-hot vector (length npts) with a 1 at that index.

    Returns a DataFrame of shape (npts, number_of_FD_events) where each column is a one-hot vector.
    If no 1's are found, returns an empty DataFrame.
    """
    try:
        fd_vals = np.loadtxt(input_file, dtype=int)
    except Exception as e:
        print(f"Error reading FD file {input_file}: {e}")
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))

    if fd_vals.ndim > 1:
        fd_vals = fd_vals.flatten()
    if fd_vals.size == 0:
        print(f"FD file {input_file} is empty.")
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))
    if len(fd_vals) != npts:
        print(f"Warning: FD file {input_file} has {len(fd_vals)} values (expected {npts}).")

    one_hot_list = []
    for i, val in enumerate(fd_vals):
        if val == 1:
            vec = np.zeros(npts, dtype=int)
            vec[i] = 1
            one_hot_list.append(vec)
    if not one_hot_list:
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))

    onehot_array = np.column_stack(one_hot_list)
    onehot_df = pd.DataFrame(onehot_array)
    return onehot_df


def process_parameters_for_group(subject, session, base_dir, npts=397):
    """
    For a given subject and session, looks for parameter files:
       trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, csf, white-matter.
    Each file is expected at:
       {base_dir}/{subject}/{session}/{subject}_{session}_{param}.txt
    If found, each file is read (assumed to be whitespace-delimited, 397 rows, 1 column),
    its single column is labeled with the parameter name, and it is added.

    Returns a DataFrame containing the parameter columns in the given order.
    If a file is missing, that parameter is skipped.
    """
    params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white-matter"]
    param_dfs = []
    for p in params:
        file_path = os.path.join(base_dir, subject, session, f"{subject}_{session}_{p}.txt")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                if df.shape[0] != npts:
                    print(f"Warning: {file_path} has {df.shape[0]} rows (expected {npts}).")
                df.columns = [p]
                param_dfs.append(df)
            except Exception as e:
                print(f"Error reading parameter file {file_path}: {e}")
        else:
            print(f"Parameter file {file_path} not found; skipping.")
    if param_dfs:
        combined_params_df = pd.concat(param_dfs, axis=1)
        return combined_params_df
    else:
        return pd.DataFrame(np.zeros((npts, 0), dtype=int))


def combine_error_posterror_fd(error_file, post_error_file, fd_file, TR=2.0, npts=397):
    """
    Processes error trials, post-error trials, and FD files for one subject/session.

    For error and post-error files:
      - Converts each row’s onset (first column) to a one-hot vector.
      - If a file is empty, it is skipped.

    For the FD file:
      - Creates one-hot vectors for each occurrence of a 1.
      - If no 1's are found, FD is skipped.

    Returns a combined DataFrame (npts rows) with columns labeled:
      error events:      error_1, error_2, …
      post-error events: post_error_1, post_error_2, …
      FD events:         FD_1, FD_2, …
    """
    if error_file is not None and os.path.exists(error_file):
        error_df = process_onsets_file_to_onehot_df(error_file, TR, npts)
        if error_df.shape[1] > 0:
            error_df.columns = [f"error_{i + 1}" for i in range(error_df.shape[1])]
        else:
            error_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))
    else:
        error_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))

    if post_error_file is not None and os.path.exists(post_error_file):
        post_error_df = process_onsets_file_to_onehot_df(post_error_file, TR, npts)
        if post_error_df.shape[1] > 0:
            post_error_df.columns = [f"post_error_{i + 1}" for i in range(post_error_df.shape[1])]
        else:
            post_error_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))
    else:
        post_error_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))

    if fd_file is not None and os.path.exists(fd_file):
        fd_df = process_fd_file_to_onehot_df(fd_file, npts)
        if fd_df.shape[1] > 0:
            fd_df.columns = [f"FD_{i + 1}" for i in range(fd_df.shape[1])]
        else:
            fd_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))
    else:
        fd_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))

    dfs = [df for df in [error_df, post_error_df, fd_df] if df.shape[1] > 0]
    if dfs:
        combined_df = pd.concat(dfs, axis=1)
    else:
        combined_df = pd.DataFrame(np.zeros((npts, 0), dtype=int))
    return combined_df


def combine_all_events_for_group(subject, session, base_dir, TR=2.0, npts=397):
    """
    For a given subject and session, constructs file paths for:
      - Error trials:      {base_dir}/{subject}/{session}/{subject}_{session}_task-ert_bold_error_onsets.txt
      - Post-error trials: {base_dir}/{subject}/{session}/{subject}_{session}_task-ert_bold_post_error_onsets.txt
      - FD:                {base_dir}/{subject}/{session}/{subject}_{session}_FD.txt
    Processes these files to create a combined one-hot table.
    Then processes additional parameter files (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, csf, white-matter)
    from the same folder and concatenates them _in front_ of the event columns.

    Returns a final DataFrame with parameters (if any) in front, then event one-hot columns.
    """
    error_file = os.path.join(base_dir, subject, session, f"{subject}_{session}_task-ert_bold_error_onsets.txt")
    post_error_file = os.path.join(base_dir, subject, session,
                                   f"{subject}_{session}_task-ert_bold_post_error_onsets.txt")
    fd_file = os.path.join(base_dir, subject, session, f"{subject}_{session}_FD.txt")

    if not os.path.exists(error_file):
        error_file = None
    if not os.path.exists(post_error_file):
        post_error_file = None
    if not os.path.exists(fd_file):
        fd_file = None

    events_df = combine_error_posterror_fd(error_file, post_error_file, fd_file, TR, npts)
    params_df = process_parameters_for_group(subject, session, base_dir, npts)

    final_df = pd.concat([params_df, events_df], axis=1)
    return final_df


def process_all_subjects(base_dir="/.../EMBARC/03_FSL_FEAT/Whole-data", TR=2.0, npts=397):
    """
    Searches the base directory for all subject/session folders (matching sub-*/ses-*).
    For each subject/session group, calls combine_all_events_for_group to create a final table,
    and saves the table as a tab-delimited text file (with no header) in that session folder.
    After processing, reports all post-error files that contained NaN values.
    """
    session_dirs = glob.glob(os.path.join(base_dir, "sub-*", "ses-*"))
    print(f"Found {len(session_dirs)} subject-session directories.")
    for session_dir in sorted(session_dirs):
        subject = os.path.basename(os.path.dirname(session_dir))
        session = os.path.basename(session_dir)
        final_df = combine_all_events_for_group(subject, session, base_dir, TR, npts)
        output_file = os.path.join(session_dir, f"{subject}_{session}_combined_events.txt")
        try:
            final_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"Saved combined table for {subject}_{session} to {output_file} with shape {final_df.shape}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")

    if post_error_nan_files:
        print("\nThe following post-error trial files contained NaN values:")
        for f in post_error_nan_files:
            print(f"  {f}")
    else:
        print("\nNo NaN values detected in post-error trial files.")


if __name__ == "__main__":
    process_all_subjects()

######################################################################################################################
# Load csv and save as txt without header
# import os
# import glob
# import pandas as pd
#
#
# def convert_csv_to_txt(csv_file, sep='\t'):
#     """
#     Convert a CSV file to a TXT file with tab-separated values,
#     excluding the header and index.
#     """
#     try:
#         # Load the CSV file into a DataFrame
#         df = pd.read_csv(csv_file)
#     except Exception as e:
#         print(f"Error reading {csv_file}: {e}")
#         return
#
#     # Create the TXT file path by replacing the .csv extension with .txt
#     txt_file = os.path.splitext(csv_file)[0] + '.txt'
#
#     try:
#         # Write the DataFrame to a TXT file with tab separation, no header or index
#         df.to_csv(txt_file, sep=sep, index=False, header=False)
#         print(f"Converted: {csv_file} -> {txt_file}")
#     except Exception as e:
#         print(f"Error writing {txt_file}: {e}")
#
#
# def convert_all_csv_in_directory(root_folder):
#     """
#     Recursively find and convert all CSV files under the given root folder.
#     """
#     # Create a pattern that matches all .csv files in the directory tree
#     pattern = os.path.join(root_folder, '**', '*.csv')
#     csv_files = glob.glob(pattern, recursive=True)
#
#     if not csv_files:
#         print("No CSV files found.")
#         return
#
#     for csv_file in csv_files:
#         convert_csv_to_txt(csv_file)
#
#
# if __name__ == '__main__':
#     # Define the root folder where the CSV files are located
#     root_folder = '/data/projects/EMBARC/03_FSL_FEAT/Bugs'
#
#     # Convert all CSV files to TXT files
#     convert_all_csv_in_directory(root_folder)
import os
import glob
import pandas as pd

def convert_csv_to_txt(csv_file, sep=' ', float_format='%.20f', encoding='utf-8-sig'):
    """
    Convert a CSV file to a TXT file with values separated by a given separator,
    excluding the header and index, using a specified float format and encoding.
    Additionally, clean up known unwanted artifacts from the data.
    """
    try:
        # Read the CSV with the specified encoding.
        df = pd.read_csv(csv_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    # Remove unwanted encoding artifacts such as '+AC0'
    df = df.replace(r'\+AC0', '', regex=True)

    # Create the TXT file path by replacing the .csv extension with .txt
    txt_file = os.path.splitext(csv_file)[0] + '.txt'

    try:
        # Write the DataFrame to a TXT file using the specified separator,
        # without header or index, and with the chosen float format.
        df.to_csv(txt_file, sep=sep, index=False, header=False, float_format=float_format)
        print(f"Converted: {csv_file} -> {txt_file}")
    except Exception as e:
        print(f"Error writing {txt_file}: {e}")

def convert_all_csv_in_directory(root_folder, sep=' ', float_format='%.20f', encoding='utf-8-sig'):
    """
    Recursively find and convert all CSV files under the given root folder.
    """
    pattern = os.path.join(root_folder, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print("No CSV files found.")
        return

    for csv_file in csv_files:
        convert_csv_to_txt(csv_file, sep=sep, float_format=float_format, encoding=encoding)

if __name__ == '__main__':
    # Define the root folder where the CSV files are located
    root_folder = '/.../EMBARC/03_FSL_FEAT/Bugs'

    # Convert all CSV files to TXT files using space as the separator
    convert_all_csv_in_directory(root_folder, sep=' ', float_format='%.20f', encoding='utf-8-sig')


#####################################################################################################################
# Batch copy the subjects that selected in a list
import os
import shutil

# Source and destination base directories
SRC_BASE = "/.../EMBARC/03_FSL_FEAT/Whole-data"
DEST_BASE = "/.../EMBARC/04_radiomics/Whole-data"


# Function to copy files
def copy_files(session_file, session_name):
    with open(session_file, "r") as f:
        for subject in f:
            subject = subject.strip()
            src_dir = os.path.join(SRC_BASE, f"sub-{subject}", session_name)
            dest_dir = os.path.join(DEST_BASE, f"sub-{subject}", session_name)

            if os.path.exists(src_dir):
                os.makedirs(dest_dir, exist_ok=True)
                for item in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, item)
                    dest_path = os.path.join(dest_dir, item)
                    if os.path.isdir(src_path):
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path)
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)
                print(f"Copied {session_name} for {subject}")
            else:
                print(f"Source directory missing: {src_dir}")


# Copy files for ses-1
copy_files("/.../EMBARC/03_FSL_FEAT/code/List_of_session/80%Accuracy30FD/Ses-1.txt", "ses-1")

# Copy files for ses-2
copy_files("/.../EMBARC/03_FSL_FEAT/code/List_of_session/80%Accuracy30FD/Ses-2.txt", "ses-2")

print("Batch copy complete.")



