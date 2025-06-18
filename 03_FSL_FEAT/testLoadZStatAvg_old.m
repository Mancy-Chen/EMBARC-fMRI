maindir='/.../EMBARC/03_FSL_FEAT/Whole-data'
%files=cellstr(spm_select('fplistrec',maindir,'^thresh_zstat1.*\.nii\.gz$'))

files=cellstr(spm_select('fplistrec',maindir,'^cope1.*\.nii\.gz$'))
savenameact='meanact1.nii'

% select FEAT_results_1 output
studyLabel='FEAT_results_1'
files=files(~cellfun('isempty',strfind(files,studyLabel)))


%% load all activation maps
for iFile=length(files):-1:1
  iFile
  nii=load_untouch_nii(files{iFile});
  tmp=double(nii.img);
  sz=size(tmp);
  try
    vol(:,:,:,iFile)=tmp;
  catch
    disp(iFile)  
  end
end

%%
tag='^mean_func\.nii\.gz$'
files=cellstr(spm_select('fplistrec',maindir,tag))
files=files(~cellfun('isempty',strfind(files,studyLabel)))

%% load all mean_func
clear anat
for iFile=length(files):-1:1
  iFile
  nii=load_untouch_nii(files{iFile});
  tmp=double(nii.img);
  sz=size(tmp);
  try
    anat(:,:,:,iFile)=tmp;
  catch
    disp(iFile)  
  end
end

%% save to disk
cd(maindir)
spacing=[3 3 3];
avganat=mean(anat,4);
snii=make_nii(avganat,spacing);
n=nifti('mean_func.nii')
n.dat.fname='meananat.nii';
create(n);
n.dat(:,:,:)=avganat;

avgact=mean(vol,4);
%anii=make_nii(avgact,spacing);
%save_nii(anii,savenameact)
n=nifti('mean_func.nii')
n.dat.fname=savenameact;
create(n);
n.dat(:,:,:)=avgact;

%sub-CU0009_ses-1_task-ert_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
