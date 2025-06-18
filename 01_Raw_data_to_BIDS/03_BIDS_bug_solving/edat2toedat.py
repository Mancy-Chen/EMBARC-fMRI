import os

def batch_rename_edat2_to_edat(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".edat2"):
                old_file = os.path.join(root, file)
                new_file = os.path.join(root, file.replace(".edat2", ".edat"))
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} -> {new_file}")

# Specify the directory where the .edat2 files are located
directory = "/.../fMRIprep/data_bids/"

# Call the function to batch rename
batch_rename_edat2_to_edat(directory)
