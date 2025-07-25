%% script to calculate motion based on an fmriprep pipeline

% define directories for all subjects and set the scrub-threshold (max
% motion in framewise displacement)
addpath('/.../fMRIprep output/')
maindir='/.../fMRIprep output/';
outdir='/.../EMBARC/output_motion/';
analysisdir=fullfile(outdir,'analysis_dir');
dirs=dir(fullfile(maindir,'sub-*'));
dirFlags=[dirs.isdir];
subjects=dirs(dirFlags);
scrub_threshold=2;
ses={'ses-1', 'ses-2'};
runs=[1];
delete_volumes=1;

for sub=1:length(subjects);
disp(['working on subject ' subjects(sub).name])
    for session=1:length(ses)
        for run=1:length(runs);
          
          filename=fullfile(maindir,subjects(sub).name, 'Output', subjects(sub).name, ses{session}, 'func', [subjects(sub).name '_' ses{session} '_task-ert_desc-confounds_timeseries.tsv'])
                                                                        
          % extract subject number so that it can be added to the matrix
          subj_name(sub) = str2double(regexp((subjects(sub).name),'\d*','Match'));
        
          % extract FD from .tsv file from fmriprep (version 1.2.3) and
          % calculate mean and number of scrubbed volumes
          if exist(filename)==2;
          disp("File found: " +filename);
          FD_full_length=getFramewiseDisplacement_epod(filename);
          FD=FD_full_length(delete_volumes+1:end);
          meanFD(sub,session,run)=mean(FD); 
          idx=FD>scrub_threshold;
          percentage_scrubbed(sub,session,run)=sum(idx(:))/size(FD,1);

%           % write confound matrix
          if sum(idx)~=0      
          matrix=zeros(length(idx),sum(idx));
            for i=1:sum(idx);
                V1=find(idx==1);
                matrix(V1(i),i)=1;
            end
                subdir=fullfile(analysisdir,subjects(sub).name);
                if exist(subdir)==0; mkdir(subdir); end          
                confound_name=fullfile(subdir,[subjects(sub).name '_' ses{session} '_run' num2str(runs(run)) '_confound_ev.txt']);
                dlmwrite(confound_name, matrix,'delimiter','\t');         
          end

          else   
          meanFD(sub,session,run)=NaN;  
          percentage_scrubbed(sub,session,run)=NaN;
          end
        end 
    end 
end

cd(analysisdir)
squeeze_meanFD=[squeeze(meanFD(:,:,1)),squeeze(meanFD(:,:,2))]; meanFD_combined= [subj_name' squeeze_meanFD]; 
csvwrite('meanFD_ert_leftovers.csv',meanFD_combined);
squeeze_percentage_scrubbed=[squeeze(percentage_scrubbed(:,:,1)),squeeze(percentage_scrubbed(:,:,2))]; percentage_scrubbed_combined= [subj_name' squeeze_percentage_scrubbed]; 
csvwrite('percentage_scrubbed_ert_leftovers.csv',percentage_scrubbed_combined);
