import sys
sys.path.append('/scratch/mchen/miniconda3/lib/python3.10/site-packages)')
import pydicom
# Load the DICOM file
dicom_file = pydicom.dcmread('/.../EMBARC/Raw_data_to_BIDS/extracted_data/UM0083/UM0083UMMR2R1_bold_ert/UM0083UMMR2R1/000052.dcm')

# Access Slice Timing
if hasattr(dicom_file, 'SliceTiming'):
    slice_timing = dicom_file.SliceTiming
    print("Slice Timing:", slice_timing)
else:
    print("Slice Timing not found.")

# Access Phase Encoding Direction
if hasattr(dicom_file, 'InPlanePhaseEncodingDirection'):
    encoding_direction = dicom_file.InPlanePhaseEncodingDirection
    print("Phase Encoding Direction:", encoding_direction)
else:
    print("Phase Encoding Direction not found.")
