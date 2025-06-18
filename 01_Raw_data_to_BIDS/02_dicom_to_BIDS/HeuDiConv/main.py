# Mingshi Chen 15/10/2024
# a heuristic file to map the extracted raw data to the BIDS format
# ASL and DWI files excluded
import re


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong to which BIDS data types."""
    fmap_magnitude1 = create_key('sub-{subject}/ses-{session}/fmap/sub-{subject}_ses-{session}_acq-func_magnitude1')
    fmap_magnitude2 = create_key('sub-{subject}/ses-{session}/fmap/sub-{subject}_ses-{session}_acq-func_magnitude2')
    fmap_phase1 = create_key('sub-{subject}/ses-{session}/fmap/sub-{subject}_ses-{session}_acq-func_phase1')
    fmap_phase2 = create_key('sub-{subject}/ses-{session}/fmap/sub-{subject}_ses-{session}_acq-func_phase2')
    bold_ert = create_key('sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-ert_bold')
    bold_resting1 = create_key('sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest1_bold')
    bold_resting2 = create_key('sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest2_bold')
    bold_reward = create_key('sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-reward_bold')
    t1w = create_key('sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w')

    info = {fmap_magnitude1: [], fmap_magnitude2: [], fmap_phase1: [], fmap_phase2: [],
            bold_ert: [], bold_resting1: [], bold_resting2: [], bold_reward: [], t1w: []}

    for s in seqinfo:
        if re.search(r"B0 Map for BOLD 39 sl", s.series_description) and "TE6.50" in s.series_id:
            info[fmap_magnitude1].append(s.series_id)
        elif re.search(r"B0 Map for BOLD 39 sl", s.series_description) and "TE8.50" in s.series_id:
            info[fmap_magnitude2].append(s.series_id)
        elif re.search(r"B0 Map for BOLD 39 sl", s.series_description) and "phase1" in s.series_description.lower():
            info[fmap_phase1].append(s.series_id)
        elif re.search(r"B0 Map for BOLD 39 sl", s.series_description) and "phase2" in s.series_description.lower():
            info[fmap_phase2].append(s.series_id)
        elif "ESTROOP 397 5 dummy 39 sl" in s.series_description:
            info[bold_ert].append(s.series_id)
        elif "RESTING 180 5 DUM 39 sl" in s.series_description:
            if "resting1" in s.series_id.lower() or "bold_resting1" in s.series_id.lower():
                info[bold_resting1].append(s.series_id)
            elif "resting2" in s.series_id.lower() or "bold_resting2" in s.series_id.lower():
                info[bold_resting2].append(s.series_id)
        elif "REWARD 240 3 DUM" in s.series_description:
            info[bold_reward].append(s.series_id)
        elif "SAG3D FSPGR 11 Flip 1 NEX" in s.series_description:
            info[t1w].append(s.series_id)

    return info
# terminal command script:
# ses-1:
# heudiconv -d /scratch/mchen/fMRIprep_BUG/extracted_data/sub-{subject}/{session}/*/*.dcm \
#           -s CU0024 -ss ses-1 -f /scratch/mchen/fMRIprep_BUG/HeuDiConv/main.py \
#           -c dcm2niix -b -o /scratch/mchen/BIDS_output/

# # Check with series description
# import pydicom
# dicom_file = "/scratch/mchen/fMRIprep_BUG/extracted_data/sub-CU0024/ses-1/CU0024CUMR1R1_b0map_bold/B0MapforBOLD39sl_extracted_TE8.50_000144.dcm"
# ds = pydicom.dcmread(dicom_file)
# print(f"Series Description: {ds.SeriesDescription}")
