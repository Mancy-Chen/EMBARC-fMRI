#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/.../depredict/repositories/EMBARC/data/data_bids/derivatives"
DEST_DIR="/.../EMBARC/03_FSL_FEAT/Whole-data"

# Loop through all subjects
for subject in $(ls $DEST_DIR); do
    if [[ $subject == "sub-"* ]]; then  # Ensure it's a valid subject folder

        for session in $(ls $DEST_DIR/$subject); do
            if [[ $session == "ses-"* ]]; then  # Ensure it's a valid session folder

                # Define source and destination file paths
                SOURCE_MASK="$SOURCE_DIR/$subject/$session/anat/${subject}_${session}_aparc+aseg.nii.gz"
                DEST_FOLDER="$DEST_DIR/$subject/$session/"
                DEST_MASK="$DEST_FOLDER/${subject}_${session}_aparc+aseg.nii.gz"

                # Ensure destination directory exists
                mkdir -p "$DEST_FOLDER"

                # Check if source file exists before copying
                if [[ -f "$SOURCE_MASK" ]]; then
                    echo "Copying mask for: $subject $session"
                    cp "$SOURCE_MASK" "$DEST_MASK"
                else
                    echo "Skipping $subject $session: Mask file not found ($SOURCE_MASK)"
                fi
            fi
        done
    fi
done

echo "Batch mask copying complete!"

