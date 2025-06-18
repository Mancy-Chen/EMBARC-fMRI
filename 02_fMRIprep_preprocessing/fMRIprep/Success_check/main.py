# Mancy Chen 04/11/2024
# Checking fMRIprep succeed or not

import os
# Directory containing the .out files
out_dir = "/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/slurm_output/TX"
success_message = "fMRIPrep finished successfully!"
missing_participants = []

# Loop through each .out file in the directory
for file_name in os.listdir(out_dir):
    if file_name.endswith(".out"):
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, 'r') as file:
            # Check if the success message is in the file
            content = file.read()
            if success_message not in content:
                # Extract participant name (assuming it’s part of the file name)
                participant_name = file_name.split('.')[0]  # Adjust split if necessary
                missing_participants.append(participant_name)

# Report results
if missing_participants:
    print(f"Participant list: {missing_participants}")
else:
    print("All .out files contain the success message: 'fMRIPrep finished successfully!'")




# Add slice timing and other json file debugging with TX and UM sites
import os
import json

# json files within this dir will be written to!
bids_dir = '/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/UM_Test1/data_bids'

slicetime = [0, 0.30769, 0.61538, 0.92308, 1.2308, 1.5385, 1.8462, 0.051282, 0.35897, 0.66667, 0.97436, 1.2821,
             1.5897, 1.8974, 0.10256, 0.41026, 0.71795, 1.0256, 1.3333, 1.641, 1.9487, 0.15385, 0.46154, 0.76923,
             1.0769, 1.3846, 1.6923, 0.20513, 0.51282, 0.82051, 1.1282, 1.4359, 1.7436, 0.25641, 0.5641, 0.87179,
             1.1795, 1.4872, 1.7949]

# Loop over subject folders within the BIDS structure
for sub in os.listdir(bids_dir):

    # Check if subject is site TX or UM - they use Philips scanners and need SliceTiming added
    if not(sub.startswith('sub-TX') or sub.startswith('sub-UM')):
        continue
    sub_dir = os.path.join(bids_dir, sub)
    # Loop over sessions
    for ses in os.listdir(sub_dir):
        ses_dir = os.path.join(sub_dir, ses, 'func/')
        json_files = [pos_json for pos_json in os.listdir(ses_dir) if pos_json.endswith('.json')]
        for f in json_files:
            with open(ses_dir+f, 'r') as file:
                dat = json.load(file)
            if "PhaseEncodingAxis" in dat.keys():
                dat["PhaseEncodingDirection"] = dat.pop("PhaseEncodingAxis")
            if "PhaseEncodingDirection" not in dat.keys():
                print(sub+ " missing PE - added direction = j")
                dat["PhaseEncodingDirection"] = "j"
            dat["SliceTiming"] = slicetime
            with open(ses_dir+f, 'w') as file:
                json.dump(dat, file, indent=4)


# Change EstimatedTotalReadoutTime to TotalReadoutTime
import os
import json
from collections import OrderedDict

# json files within this dir will be written to!
bids_dir = '/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/UM_Test1/data_bids'
folder_name = ['anat', 'fmap', 'func']

folder_name = ['anat', 'fmap', 'func']

# Loop over subject folders within the BIDS structure
for sub in os.listdir(bids_dir):
    # Check if subject is site TX or UM - they use Philips scanners and need SliceTiming added
    if not (sub.startswith('sub-TX') or sub.startswith('sub-UM')):
        continue
    sub_dir = os.path.join(bids_dir, sub)
    # Loop over sessions
    for ses in os.listdir(sub_dir):
        for folder in folder_name:
            ses_dir = os.path.join(sub_dir, ses, folder)
            if not os.path.exists(ses_dir):
                continue  # Skip if the folder doesn't exist
            json_files = [pos_json for pos_json in os.listdir(ses_dir) if pos_json.endswith('.json')]
            for f in json_files:
                file_path = os.path.join(ses_dir, f)
                with open(file_path, 'r') as file:
                    dat = json.load(file, object_pairs_hook=OrderedDict)

                if "EstimatedTotalReadoutTime" in dat:
                    # Create a new OrderedDict while replacing the key
                    new_dat = OrderedDict()
                    for key, value in dat.items():
                        if key == "EstimatedTotalReadoutTime":
                            new_dat["TotalReadoutTime"] = value
                        else:
                            new_dat[key] = value

                    # Write back the updated dictionary to the file
                    with open(file_path, 'w') as file:
                        json.dump(new_dat, file, indent=4)

