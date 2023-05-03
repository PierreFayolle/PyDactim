import os

def get_name_of_path(path):
    if not "sub" in path and not "ses" in path:
        return path
    else:
        return " ".join(path.replace(".nii.gz", "").replace(".nii", "").split("_")[2:])

def load_dicom(dicom_path):
    """Read a dicom as a binary (CSA Header) and create a dictionnary from each sequence parameter

    Args:
        dicom_path (str) : the absolute path of the dicom

    Returns:
        dict : the dictionnary that holds all the sequence parameters
    """
    f = str(open(dicom_path, 'rb').readlines())
    f = f[f.find("b'### ASCCONV BEGIN"):f.find(", b'### ASCCONV END ###")]
    f = f.replace("\\n", "").replace("\\t", "").replace(" b'", "").replace("'", "").replace("NE1,", "NE1 ").split(",")
    dcm = {row.split(' = ')[0] : row.split(' = ')[1] for row in f[1:]}
    return dcm

if __name__ == '__main__':
    print(get_name_of_path(r"D:\Results\GLIOBIOPSY\derivative\sub-003\ses-01\anat\sub-003_ses-01_FLAIR_brain_1mm.nii.gz"))