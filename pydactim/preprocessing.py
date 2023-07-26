import pydactim as pyd
import os

global MODEL_PATH, FORCE

MODEL_PATH = None
FORCE = False

def init(model_path=None, force=False):
    global MODEL_PATH, FORCE
    if model_path is not None and not os.path.isdir(model_path):
        raise ValueError(f"ERROR - Could not find model folds in the following directory: {model_path}")
    else:
        MODEL_PATH = model_path
    FORCE = force

def get_model():
    global MODEL_PATH
    return MODEL_PATH

def get_force():
    global FORCE
    return FORCE

@pyd.timed
def preproc(sub_path, ses="ses-01", ref="T1w", normalize=False, keep_all=True):
    # Checking errors
    sub = os.path.basename(sub_path)
    if "sub" not in sub:
        raise ValueError("ERROR - Could not find a sub number in the data_path, make sure it is bids compliant")

    ses_path = os.path.join(sub_path, ses)
    if not os.path.isdir(ses_path):
        raise ValueError(f"ERROR - Could not find a directory with the following session number {ses}")

    modalities = os.listdir(ses_path)
    if "anat" not in modalities:
        raise ValueError("ERROR - Can not start process without the anat directory")

    anat_path = os.path.join(ses_path, "anat")
    ref_path = f"{sub}_{ses}_{ref}.nii.gz"
    if ref_path not in os.listdir(anat_path):
        raise FileNotFoundError(f"ERROR - The following reference filename could not be found:\n\t{ref_path}")

    # Starting to preproc the reference sequence
    print(f"INFO - Starting preprocessing for the reference image at\n\t{ref_path}")
    ref_path = os.path.join(anat_path, ref_path)
    ref_corrected, ref_brain_mask, ref_crop = ref_preproc(ref_path, normalize)

    # Starting to loop through each sequence
    print(f"INFO - Starting preprocessing for the following modalities:\n\t{', '.join(modalities)}")
    print("INFO - Starting with anatomic sequences")
    for seq in os.listdir(anat_path):
        seq_path = os.path.join(anat_path, seq)
        # Check if the file is a nii.gz
        if seq_path.endswith("nii.gz") and seq_path != ref_path and pyd.is_native(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

    print("INFO - Continuing with diffusion sequences")
    dwi_path = os.path.join(ses_path, "dwi")
    for seq in os.listdir(dwi_path):
        seq_path = os.path.join(dwi_path, seq)
        # Check if the file is a nii.gz
        if seq_path.endswith("nii.gz") and pyd.is_native(seq_path) and pyd.utils.is_useful(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

    print("INFO - Continuing with perfusion sequences")
    perf_path = os.path.join(ses_path, "perf")
    for seq in os.listdir(perf_path):
        seq_path = os.path.join(perf_path, seq)
        # Check if the file is a nii.gz
        if seq_path.endswith("nii.gz") and pyd.is_native(seq_path) and pyd.utils.is_useful(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

@pyd.timed
def ref_preproc(ref_path, normalize):
    # crop => resample => n4 bias field correction => skull stripping => crop
    ref_path_cropped, crop_idx_1 = pyd.crop(ref_path, force=get_force())
    ref_path_resampled = pyd.resample(ref_path_cropped, 1)[0]
    ref_path_corrected, ref_path_corrected_mask = pyd.n4_bias_field_correction(ref_path_resampled, mask=True, force=get_force())
    ref_path_brain, ref_path_brain_mask = pyd.skull_stripping(ref_path_corrected, get_model(), mask=True, force=get_force())
    ref_path_brain_cropped, crop_idx_2 = pyd.crop(ref_path_brain, force=get_force())
    ref_path_brain_mask_cropped = pyd.apply_crop(ref_path_brain_mask, crop_idx_2, force=get_force())
    if normalize: ref_path_normalized = pyd.normalize(ref_path_brain_cropped, force=get_force())
    return ref_path_corrected, ref_path_brain_mask, crop_idx_2

@pyd.timed
def other_preproc(seq_path, ref_brain, ref_brain_mask, ref_crop, normalize):
    # registration => apply brain mask => apply crop => n4 bias field correction
    seq_path_registered, matrix_path = pyd.registration(ref_brain, seq_path, force=get_force())
    seq_path_brain = pyd.apply_mask(seq_path_registered, ref_brain_mask, suffix="brain", force=get_force())
    seq_path_cropped = pyd.apply_crop(seq_path_brain, crop=ref_crop, force=get_force())
    seq_path_corrected, seq_path_corrected_mask = pyd.n4_bias_field_correction(seq_path_cropped, mask=True, force=get_force())
    if normalize: seq_path_normalized = pyd.normalize(seq_path_corrected, force=get_force())