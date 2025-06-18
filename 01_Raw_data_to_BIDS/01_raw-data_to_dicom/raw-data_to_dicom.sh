#!/bin/bash

# Extract the dicom files to the current folder
cd .../Raw_data_to_BIDS/fMRIprep_BUG/raw_data/UM0083/
for file in *.tgz; do
    base=$(basename "$file" .tgz)
    mkdir -p .../Raw_data_to_BIDS/fMRIprep_BUG/extracted_data/UM0083/$base
    tar -xvzf "$file" -C .../Raw_data_to_BIDS/fMRIprep_BUG/extracted_data/UM0083/$base
done

