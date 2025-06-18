#!/bin/bash

# Define the directories
DATA_DIR="/.../EMBARC/03_FSL_FEAT/Whole-data"
FSF_TEMPLATE="/.../EMBARC/03_FSL_FEAT/FSL_FEAT_model/Model.fsf"

# Loop through all subjects
for subject in $(ls $DATA_DIR); do
    if [[ $subject == "sub-"* ]]; then  # Ensure it follows subject naming convention

        for session in $(ls $DATA_DIR/$subject); do
            if [[ $session == "ses-"* ]]; then

                # Define input file path
                FUNC_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-ert_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

                # Correct output path (inside each subject/session folder)
                OUTPUT_DIR="$DATA_DIR/$subject/$session/FEAT_results/${subject}_${session}.feat"
                FSF_FILE="$DATA_DIR/$subject/$session/FEAT_results/${subject}_${session}_design.fsf"

                # Condition Timing Files (ii, ci, ic, cc)
                II_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-ii_timing.txt"
                CI_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-ci_timing.txt"
                IC_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-ic_timing.txt"
                CC_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-cc_timing.txt"

                # Error & Post-Error Trials (also used for confounds)
                ERROR_EV_FILE="$DATA_DIR/$subject/$session/${subject}_${session}_task-ert_bold_error_EV.txt"

                # Ensure output directory exists
                mkdir -p "$(dirname "$OUTPUT_DIR")"

                # Check if all required files exist before running FEAT
                missing_files=()
                for file in "$FUNC_FILE" "$II_FILE" "$CI_FILE" "$IC_FILE" "$CC_FILE" "$ERROR_EV_FILE"; do
                    if [[ ! -f "$file" ]]; then
                        missing_files+=("$file")
                    fi
                done

                # If any files are missing, print them and skip this subject/session
                if [[ ${#missing_files[@]} -gt 0 ]]; then
                    echo "Skipping $subject $session: Missing required files."
                    for missing in "${missing_files[@]}"; do
                        echo "   - Missing: $missing"
                    done
                    continue
                fi

                # Create a subject-specific fsf file by replacing variables in Model.fsf
                sed -e "s|SUBJECT|${subject}|g" \
                    -e "s|SESSION|${session}|g" \
                    -e "s|FUNC_FILE|${FUNC_FILE}|g" \
                    -e "s|OUTPUT_DIR|${OUTPUT_DIR}|g" \
                    -e "s|II_FILE|${II_FILE}|g" \
                    -e "s|CI_FILE|${CI_FILE}|g" \
                    -e "s|IC_FILE|${IC_FILE}|g" \
                    -e "s|CC_FILE|${CC_FILE}|g" \
                    -e "s|ERROR_EV_FILE|${ERROR_EV_FILE}|g" \
                    -e "s|CONFOUND_FILE|${ERROR_EV_FILE}|g" \
                    $FSF_TEMPLATE > $FSF_FILE

                # Run FEAT on the generated fsf file
                echo "Running FEAT for $subject $session..."
                feat $FSF_FILE
            fi
        done
    fi
done

echo "Batch FEAT processing complete!"

