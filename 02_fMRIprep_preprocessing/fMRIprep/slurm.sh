#!/bin/bash
#SBATCH --job-name=fmriprep      # Name of the job
#SBATCH --cpus-per-task=8        # Number of CPU cores
#SBATCH --mem=32G                # Maximum system memory (RAM)
#SBATCH --time=0-24:00           # Time limit (DD-HH:MM)
#SBATCH --nice=1                 # Lower priority to allow higher-priority jobs to run first

# Define paths
BIDS_DIR=/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/data_bids
OUT_DIR=/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/data_output
WORK_DIR=/.../fMRIprep/work  # Separate working directory
FREESURFER_LICENSE=/.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/license.txt.txt
SIF_IMAGE=/.../fMRIprep/fmriprep_latest.sif

# Load Apptainer/Singularity module if required by your environment
module load apptainer

# Loop through each subject in the BIDS directory
for SUBJECT in $(ls ${BIDS_DIR} | grep -E '^sub-'); do
    # Run the actual fMRIprep job using Apptainer
    apptainer run --cleanenv \
        -B ${BIDS_DIR}:/data \
        -B ${OUT_DIR}:/out \
        -B ${WORK_DIR}:/work \
        -B ${FREESURFER_LICENSE}:/license/license.txt \
        ${SIF_IMAGE} \
        /data /out participant \
        --participant-label ${SUBJECT} \
        --fs-license-file /license/license.txt \
        -w /work

    # Print the selected CPU information
    echo "Running fMRIPrep for ${SUBJECT} on CPU cores: $SLURM_CPUS_PER_TASK"
done

