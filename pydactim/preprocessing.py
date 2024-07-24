import os

from .utils import *
from .transformation import *

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

@timed
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
        if seq_path.endswith("nii.gz") and seq_path != ref_path and is_native(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

    print("INFO - Continuing with diffusion sequences")
    dwi_path = os.path.join(ses_path, "dwi")
    for seq in os.listdir(dwi_path):
        seq_path = os.path.join(dwi_path, seq)
        # Check if the file is a nii.gz
        if seq_path.endswith("nii.gz") and is_native(seq_path) and is_useful(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

    print("INFO - Continuing with perfusion sequences")
    perf_path = os.path.join(ses_path, "perf")
    for seq in os.listdir(perf_path):
        seq_path = os.path.join(perf_path, seq)
        # Check if the file is a nii.gz
        if seq_path.endswith("nii.gz") and is_native(seq_path) and is_useful(seq_path):
            print(f"INFO - Nifti file found: {seq_path}")
            # Starting preproc for the current sequence path
            seq_path = other_preproc(seq_path, ref_corrected, ref_brain_mask, ref_crop, normalize)

@timed
def ref_preproc(ref_path, normalize):
    # crop => resample => n4 bias field correction => skull stripping => crop
    ref_path_cropped, crop_idx_1 = crop(ref_path, force=get_force())
    ref_path_resampled = resample(ref_path_cropped, 1)[0]
    ref_path_corrected, ref_path_corrected_mask = n4_bias_field_correction(ref_path_resampled, mask=True, force=get_force())
    ref_path_brain, ref_path_brain_mask = skull_stripping(ref_path_corrected, get_model(), mask=True, force=get_force())
    ref_path_brain_cropped, crop_idx_2 = crop(ref_path_brain, force=get_force())
    ref_path_brain_mask_cropped = apply_crop(ref_path_brain_mask, crop_idx_2, force=get_force())
    if normalize: ref_path_normalized = normalize(ref_path_brain_cropped, force=get_force())
    return ref_path_corrected, ref_path_brain_mask, crop_idx_2

@timed
def other_preproc(seq_path, ref_brain, ref_brain_mask, ref_crop, normalize):
    # registration => apply brain mask => apply crop => n4 bias field correction
    seq_path_registered, matrix_path = registration(ref_brain, seq_path, force=get_force())
    seq_path_brain = apply_mask(seq_path_registered, ref_brain_mask, suffix="brain", force=get_force())
    seq_path_cropped = apply_crop(seq_path_brain, crop=ref_crop, force=get_force())
    seq_path_corrected, seq_path_corrected_mask = n4_bias_field_correction(seq_path_cropped, mask=True, force=get_force())
    if normalize: seq_path_normalized = normalize(seq_path_corrected, force=get_force())

# def prediction_glioma(input_path, model_path, landmarks_path, force=True, suffix="predicted"):
#     print(f"INFO - Starting glioma prediction for\n\t{input_path :}")
#     output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
#     if os.path.exists(output_path) and not force:
#         print(f"INFO - Glioma segmentation already done for\n\t{input_path :}")
#         return output_path
    
#     from monai.inferers import sliding_window_inference
#     from monai.networks.nets import UNETR
#     import torch
#     import torchio as tio
#     import nibabel as nib
#     import numpy as np
#     model = UNETR(
#             in_channels=1,
#             out_channels=2,
#             img_size=(176, 208, 160),
#             feature_size=16,
#             hidden_size=768,
#             mlp_dim=3072,
#             num_heads=12,
#             pos_embed="perceptron",
#             norm_name="instance",
#             res_block=True,
#             dropout_rate=0.2,
#     ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     transform = tio.Compose([
#             tio.ToCanonical(),
#             tio.Resample(1),
#             tio.CropOrPad((176, 208, 160)),
#             tio.HistogramStandardization({"image": np.load(landmarks_path)}),
#             tio.ZNormalization(masking_method=tio.ZNormalization.mean),
#     ])

#     ds = tio.SubjectsDataset([
#         tio.Subject(image = tio.ScalarImage(input_path))], 
#         transform=transform
#     )[0]

#     affine = nib.load(input_path).affine
#     with torch.no_grad():
#         img = ds["image"]["data"]
#         val_inputs = torch.unsqueeze(img, 1)
#         val_outputs = sliding_window_inference(val_inputs.cuda(), (176, 208, 160), 4, model, overlap=0.25)
#         val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0].astype(float)
#         pred_map = tio.LabelMap(tensor=np.expand_dims(val_outputs, 0), affine=ds.image.affine)
#         ds.add_image(pred_map, "pred")
#         ds_inv = ds.apply_inverse_transform(warn=True)
#         val_outputs = ds_inv["pred"].data.numpy().squeeze()

#         nib.save(nib.Nifti1Image(val_outputs.squeeze(), affine), output_path)
    
#     output_path = remove_small_object(output_path, 5000, force=True)
#     print(f"INFO - Saving generated image at\n\t{output_path :}")
#     return output_path

# def uncertainty_prediction_glioma(input_path, model_path, force=True, suffix="uncertainty"):
#     print(f"INFO - Starting glioma uncertainty prediction for\n\t{input_path :}")
#     output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
#     if os.path.exists(output_path) and not force:
#         print(f"INFO - Uncertainty glioma segmentation already done for\n\t{input_path :}")
#         return output_path
    
#     from monai.inferers import sliding_window_inference
#     from monai.networks.nets import UNETR
#     import torch
#     import torchio as tio
#     import nibabel as nib
#     import numpy as np
#     from tqdm.auto import tqdm, trange
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNETR(
#             in_channels=1,
#             out_channels=2,
#             img_size=(176, 208, 160),
#             feature_size=16,
#             hidden_size=768,
#             mlp_dim=3072,
#             num_heads=12,
#             pos_embed="perceptron",
#             norm_name="instance",
#             res_block=True,
#             dropout_rate=0.2,
#     ).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     transform = tio.Compose([
#             tio.ToCanonical(),
#             tio.Resample(1),
#             tio.CropOrPad((176, 208, 160)),
#             tio.HistogramStandardization({"image": np.load("E:/Leo/script/results/landmarks.npy")}),
#             tio.ZNormalization(masking_method=tio.ZNormalization.mean),
#             tio.RandomFlip(),
#             tio.RandomAffine(p=0.5),
#     ])

#     subject = tio.Subject(image = tio.ScalarImage(input_path)) 
#     affine = nib.load(input_path).affine

#     results = []
#     for _ in trange(20):
#         subject = transform(subject)
#         inputs = subject.image.data.to(device)
        
#         with torch.no_grad():
#             inputs = torch.unsqueeze(inputs, 1)
#             outputs = sliding_window_inference(inputs, (176, 208, 160), 4, model, overlap=0.25)
#         outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0].astype(float)
#         pred_map = tio.LabelMap(tensor=np.expand_dims(outputs, 0), affine=subject.image.affine)
#         subject.add_image(pred_map, "pred")
#         subject_inv = subject.apply_inverse_transform(warn=True)
#         results.append(subject_inv["pred"].data)

#     result = torch.stack(results).long()
#     tta_result_tensor = result.mode(dim=0).values

#     different = torch.stack([
#         tensor != tta_result_tensor
#         for tensor in results
#     ])
#     uncertainty = different.float().mean(dim=0)
#     uncertainty_img = tio.ScalarImage(tensor=uncertainty, affine=subject.image.affine)
#     subject.add_image(uncertainty_img, "uncertainty")

#     uncertainty_img = uncertainty_img.data.numpy().squeeze()
#     nib.save(nib.Nifti1Image(uncertainty_img, affine), output_path)

#     print(f"INFO - Saving generated image at\n\t{output_path :}")
#     return output_path

# def add_tissue_class(input_path, mask_path, num_class, force=True, suffix="masked"):
#     import nibabel as nib

#     print(f"INFO - Starting to add a new class for\n\t{input_path :}")
#     output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
#     if os.path.exists(output_path) and not force:
#         print(f"INFO - New added class already done for\n\t{input_path :}")
#         return output_path
    
#     img = nib.load(input_path)
#     img_data = img.get_fdata()

#     mask = nib.load(mask_path)
#     mask_data = mask.get_fdata()

#     img_data[mask_data > 0] = num_class

#     nib.save(nib.Nifti1Image(img_data, img.affine), output_path)
#     print(f"INFO - Saving generated image at\n\t{output_path :}")
#     return output_path

# def extract_dim(input_path, dim, force=True, suffix=""):
#     import nibabel as nib

#     print(f"INFO - Starting to extract the dimension {dim} for\n\t{input_path :}")
#     output_path = input_path.replace(".nii.gz", "_dim" + str(dim) + ".nii.gz")
#     if os.path.exists(output_path) and not force:
#         print(f"INFO - Extracted dimension already done for\n\t{input_path :}")
#         return output_path
    
#     img = nib.load(input_path)
#     img_data = img.get_fdata()

#     img_data = img_data[..., dim]
 
#     nib.save(nib.Nifti1Image(img_data, img.affine), output_path)
#     print(f"INFO - Saving generated image at\n\t{output_path :}")
#     return output_path

class Pipeline:
    def __init__(self, data):
        self.data = data
        self.clean()

        if not os.path.exists(self.data): 
            raise FileNotFoundError(f"ERROR - The following data file is not existing: {self.data}")

    def clean(self):
        self.cropped_idx = None
        self.matrix = None
        self.bias = None
        self.mask = None
        self.pred = None
        self.pve = None

    def susan(self, size):
        self.data = susan(self.data, size)

    def remove_small_object(self, threshold):
        self.data = remove_small_object(self.data, threshold)

    def tissue_classifier(self):
        self.data, self.pve = tissue_classifier(self.data, force=False)

    def skull_stripping(self, mask, model):
        if not os.path.isdir(model): 
            raise NotADirectoryError(f"ERROR - The following model path is not a directory: {model}")
        
        if mask:
            self.data, self.mask = skull_stripping(self.data, model, mask, force=False)
        else:
            self.data = skull_stripping(self.data, model, mask, force=False)

    def registration(self, ref):
        self.data, self.matrix = registration(ref, self.data, force=False)

    def apply_transformation(self, ref, matrix):
        if not os.path.exists(ref): 
            raise FileNotFoundError(f"ERROR - The following reference file is not existing: {ref}")
        if not os.path.exists(matrix): 
            raise FileNotFoundError(f"ERROR - The following matrix file is not existing: {matrix}")
        self.data = apply_transformation(ref, self.data, matrix)

    def crop(self):
        self.data, self.cropped_idx = crop(self.data)
        print(f"DEBUG - Cropping at {self.cropped_idx[0]}, {self.cropped_idx[1]}, {self.cropped_idx[2]}, {self.cropped_idx[3]}, {self.cropped_idx[4]}, {self.cropped_idx[5]}")

    def resample(self, obj):
        self.data = resample(self.data, obj)[0]

    def apply_crop(self, crop=None):
        if self.cropped_idx is None:
            if crop is not None:
                self.data = apply_crop(self.data, crop)
            else:
                raise ValueError("ERROR - If no previous crop done, you need to have a crop argument")
        else:
            self.data = apply_crop(self.data, self.cropped_idx)

    def apply_mask(self, mask):
        self.data = apply_mask(self.data, mask)

    def normalize(self):
        self.data = normalize(self.data)

    def n4_bias_field_correction(self, mask):
        if mask:
            self.data, self.bias = n4_bias_field_correction(self.data, mask, force=False)
        else:
            self.data = n4_bias_field_correction(self.data, mask, force=False)

    def prediction_glioma(self, model, landmarks):
        self.pred = prediction_glioma(self.data, model, landmarks)

    def add_tissue_class(self, ref, num):
        self.data = add_tissue_class(ref, self.pred, num)

    def susan(self):
        self.data = susan(self.data, force=False)

    def extract_dim(self, dim):
        self.data = extract_dim(self.data, dim)

    def change_data(self, new_data):
        self.data = new_data

if __name__ == '__main__':
    sub = "sub-031"
 
    # Correction IA
    """
    pipeline = Pipeline(f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_T1w.nii.gz")
    pipeline.pred = f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_FLAIR_flirt_mask_susan_predicted_filtered.nii.gz"
    tissues = f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_T1w_cropped_resampled_corrected_brain_fast.nii.gz"
    pipeline.add_tissue_class(tissues, num=4)
    raise
    """

    # T1
    pipeline = Pipeline(f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_T1w.nii.gz")
    pipeline.crop()
    pipeline.resample(1)
    pipeline.n4_bias_field_correction(mask=True)
    pipeline.skull_stripping(mask=True, model=f"C:/Users/467355/Documents/HD-BET-master/HD_BET/hd-bet_params")
    path = pipeline.data
    pipeline.tissue_classifier()
    tissues = pipeline.data
    # T2 FLAIR
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_FLAIR.nii.gz")
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    pipeline.susan()
    pipeline.prediction_glioma(model="E:/Leo/script/results/new_finetuning_susan_aug/not_best_metric_model_75000.pth", landmarks="E:/Leo/script/results/landmarks.npy")
    pipeline.add_tissue_class(tissues, num=4)
    # T1ce
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_ce-GADOLINIUM_T1w.nii.gz")
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # SWI
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/anat/{sub}_ses-01_SWI.nii.gz")
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # Resolve ADC
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/dwi/{sub}_ses-01_resolve_adc.nii.gz")
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # Resolve Tracew
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/dwi/{sub}_ses-01_resolve_tracew.nii.gz")
    pipeline.extract_dim(1)
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # FA
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/dwi/{sub}_ses-01_fa.nii.gz")
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # T2* DSC PERFUSION
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/perf/{sub}_ses-01_perf.nii.gz")
    pipeline.extract_dim(0)
    pipeline.registration(ref=path)
    pipeline.apply_mask(mask=pipeline.mask)
    # CBV
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/perf/{sub}_ses-01_rCBV.nii.gz")
    pipeline.apply_transformation(ref=path, matrix=pipeline.matrix)
    pipeline.apply_mask(mask=pipeline.mask)
    # CBF
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/perf/{sub}_ses-01_rCBF.nii.gz")
    pipeline.apply_transformation(ref=path, matrix=pipeline.matrix)
    pipeline.apply_mask(mask=pipeline.mask)
    # MTT
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/perf/{sub}_ses-01_MTT.nii.gz")
    pipeline.apply_transformation(ref=path, matrix=pipeline.matrix)
    pipeline.apply_mask(mask=pipeline.mask)
    # TPP
    pipeline.change_data(f"E:/Haissam/data/{sub}/ses-01/perf/{sub}_ses-01_TTP.nii.gz")
    pipeline.apply_transformation(ref=path, matrix=pipeline.matrix)
    pipeline.apply_mask(mask=pipeline.mask)
