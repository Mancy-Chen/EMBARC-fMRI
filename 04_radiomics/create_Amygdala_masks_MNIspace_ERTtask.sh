fslroi HarvardOxford-sub-prob-1mm.nii.gz AmygdalaLeft9 9 1
fslroi HarvardOxford-sub-prob-1mm.nii.gz AmygdalaRight19 19 1

# does not work in flirt - use SPM instead
#flirt -in AmygdalaLeft9 -ref mean_func -applyxfm -init IDtfm.mat -out AmygdalaLeft9_lores 
#flirt -in AmygdalaRight19 -ref mean_func -applyxfm -init IDtfm.mat -out AmygdalaRight19_lores 

# SPM syntax in matlab:
# mean_func: example file from one subject, assuming all are in MNI space, resampled/resliced to identical matrix size 
# spm_reslice({'mean_func.nii','AmygdalaLeft9.nii','AmygdalaRight19.nii'})



