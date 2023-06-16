from dactim_mri.sorting import sort_dicom
from dactim_mri.conversion import convert_dicom_to_nifti
from dactim_mri.transformation import resample, registration

registration(r"C:\Users\467355\Desktop\data\Glioma\2_3d_flair_brain.nii.gz", r"C:\Users\467355\Desktop\data\Glioma\3_3d_flair_brain_resampled.nii.gz")