#!/bin/bash

DATA_DIR="/.../EMBARC/03_FSL_FEAT/Whole-data"

# Find all subject/session directories
TASK_LIST=($(find $DATA_DIR -mindepth 2 -maxdepth 2 -type d -name "ses-*"))

missing_files=()

for TASK_PATH in "${TASK_LIST[@]}"; do
    subject=$(basename $(dirname $TASK_PATH))
    session=$(basename $TASK_PATH)

    ERROR_FILE="$TASK_PATH/${subject}_${session}_task-ert_bold_error_onsets.txt"
    POST_ERROR_FILE="$TASK_PATH/${subject}_${session}_task-ert_bold_post_error_onsets.txt"

    if [[ ! -f "$ERROR_FILE" || ! -f "$POST_ERROR_FILE" ]]; then
        missing_files+=("$subject $session")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "Missing error-related files for the following subjects/sessions:"
    printf '%s\n' "${missing_files[@]}"
else
    echo "All subjects and sessions contain the required error-related files."
fi

