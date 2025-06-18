#!/bin/bash

# Define the base directory
DATA_DIR="/.../EMBARC/03_FSL_FEAT/Whole-data"

# Loop through all subjects
for subject in $(ls $DATA_DIR); do
    if [[ $subject == "sub-"* ]]; then  # Ensure it's a valid subject folder

        for session in $(ls $DATA_DIR/$subject); do
            if [[ $session == "ses-"* ]]; then  # Ensure it's a valid session folder

                # Define the FEAT results folder
                FEAT_DIR="$DATA_DIR/$subject/$session/FEAT_results"

                # Check if FEAT results exist before deleting
                if [[ -d "$FEAT_DIR" ]]; then
                    echo "Removing FEAT results for: $subject $session"
                    rm -rf "$FEAT_DIR"/*
                else
                    echo "Skipping $subject $session: No FEAT results found."
                fi
            fi
        done
    fi
done

echo "Batch FEAT results removal complete!"


# find /.../EMBARC/03_FSL_FEAT/Whole-data -type f -name "sub-*_ses-*_task-ert_bold_error_onsets_combined_onehot_table.csv" -delete


