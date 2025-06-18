#!/bin/bash
#SBATCH --job-name=fsl_feat_batch
#SBATCH --cpus-per-task=4  # Adjust CPU allocation
#SBATCH --mem=16G  # Memory per FEAT job
#SBATCH --time=24:00:00  # Adjust runtime
#SBATCH --nice=10 # Higher priority
#SBATCH --array=0-60 #Adjust based on number of subjects/sessions

# Load FSL module (modify if necessary)
module load fsl/6.0.3

# Define directories
DATA_DIR="/.../EMBARC/data/03_FSL_FEAT/Data_augmented/400_volumes/"
FSF_TEMPLATE="/.../EMBARC/code/03_FSL_FEAT/FSL_FEAT_model/Model9/Model7.fsf"

# Create an array of all subject/session paths
TASK_LIST=($(find $DATA_DIR -mindepth 2 -maxdepth 2 -type d -name "ses-*"))
TOTAL_TASKS=${#TASK_LIST[@]}

# Assign task based on SLURM_ARRAY_TASK_ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_TASKS ]]; then
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

TASK_PATH=${TASK_LIST[$SLURM_ARRAY_TASK_ID]}
subject_base=$(basename $(dirname $TASK_PATH))
subject="${subject_base%_*}"
augmentation_number="${subject_base##*_}"
echo $subject
echo $augmentation_number
session=$(basename $TASK_PATH)

# Define input files
FUNC_FILE="$DATA_DIR/$subject_base/$session/${subject}_${session}_task-ert_space-MNI152NLin2009cAsym_desc-preproc_bold_augmented${augmentation_number}.nii.gz"
OUTPUT_DIR="$DATA_DIR/$subject_base/$session/FEAT_results_NEW/${subject}_${session}_augmented${augmentation_number}.feat"
FSF_FILE="$DATA_DIR/$subject_base/$session/FEAT_results_NEW/${subject}_${session}_augmented${augmentation_number}_design.fsf"

# Condition Timing Files (ii, ci, ic, cc)
II_FILE="$DATA_DIR/$subject_base/$session/${subject}_${session}_task-ii_timing.txt"
CI_FILE="$DATA_DIR/$subject_base/$session/${subject}_${session}_task-ci_timing.txt"
IC_FILE="$DATA_DIR/$subject_base/$session/${subject}_${session}_task-ic_timing.txt"
CC_FILE="$DATA_DIR/$subject_base/$session/${subject}_${session}_task-cc_timing.txt"

# Counfound files for motion, CSF, White Matter, error trials, post error trials and framewise_displacement
CONFOUNDS="$DATA_DIR/$subject_base/$session/${subject}_${session}_combined_events.txt"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_DIR")"

# Check if required files exist before running FEAT
missing_files=()
for file in "$FUNC_FILE" "$II_FILE" "$CI_FILE" "$IC_FILE" "$CC_FILE" "$CONFOUNDS"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

# If any files are missing, print them and exit
if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "Skipping $subject $session: Missing required files."
    for missing in "${missing_files[@]}"; do
        echo "   - Missing: $missing"
    done
    exit 1
fi

# Generate subject-specific fsf file
sed -e "s|SUBJECT|${subject}|g" \
    -e "s|SESSION|${session}|g" \
    -e "s|FUNC_FILE|${FUNC_FILE}|g" \
    -e "s|OUTPUT_DIR|${OUTPUT_DIR}|g" \
    -e "s|II_FILE|${II_FILE}|g" \
    -e "s|CI_FILE|${CI_FILE}|g" \
    -e "s|IC_FILE|${IC_FILE}|g" \
    -e "s|CC_FILE|${CC_FILE}|g" \
    -e "s|CONFOUNDS|${CONFOUNDS}|g" \
    $FSF_TEMPLATE > $FSF_FILE

# Run FEAT
echo "Running FEAT for $subject_base $session..."
feat $FSF_FILE

echo "Finished"
