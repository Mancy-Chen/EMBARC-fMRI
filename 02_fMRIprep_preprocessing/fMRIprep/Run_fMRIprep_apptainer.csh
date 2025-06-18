# run fMRIprep apptainer in terminal [under /scratch folder!]
apptainer run --cleanenv \
    -B /.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/data_bids:/data \
    -B /.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/data_output:/out \
    -B /.../EMBARC/fMRIprep_preprocessing/fMRIprep_running_env/license.txt:/license/license.txt \
    fmriprep_latest.sif \
    /data /out participant --fs-license-file /license/license.txt 


