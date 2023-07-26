import os, time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        seconds = round(end - start, 2)
        minutes = int((seconds % 3600) // 60)
        hours = int(seconds // 3600)
        remaining_seconds = round(seconds % 60)

        # Build the formatted time string
        formatted_time = ""
        if hours > 0:
            formatted_time += f"{hours}h"
        if minutes > 0:
            formatted_time += f"{minutes}min"
        if remaining_seconds > 0:
            formatted_time += f"{remaining_seconds}sec"
        
            
        print(f"INFO - {func.__name__} ran in {formatted_time}")
        return result
    return wrapper

def is_native(path):
    path = os.path.basename(path)
    temp = path.split("_")[2:]
    if   temp[0] == "ce-GADOLINIUM": temp = temp[1:]
    elif temp[0] == "task"    and temp[1] == "rest":   temp = temp[1:]
    elif temp[0] == "dwi"     and temp[1] == "pa":     temp = temp[1:]
    elif temp[0] == "resolve" and temp[1] == "tracew": temp = temp[1:]
    elif temp[0] == "resolve" and temp[1] == "adc":    temp = temp[1:]
    if len(temp) > 1: return False
    return True
    
def is_useful(path):
    path = os.path.basename(path)
    temp = path.split("_")[2:]
    if   temp[0] == "ce-GADOLINIUM": temp = temp[1:]
    elif temp[0] == "task"    and temp[1] == "rest":   temp = temp[1:]
    elif temp[0] == "dwi"     and temp[1] == "pa":     temp = temp[1:]
    elif temp[0] == "resolve" and temp[1] == "tracew": temp = temp[1:]
    elif temp[0] == "resolve" and temp[1] == "adc":    temp = temp[1:]
    seq = temp[0].split(".")[0]
    not_useful = ["rest", "B0map", "perf", "dwi", "pa", "tracew"]
    if seq in not_useful:
        return False
    return True

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