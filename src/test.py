from dactim_mri.sorting import sort_dicom
from dactim_mri.conversion import convert_dicom_to_nifti

import itk

brainMask = itk.StripTsImageFilter(img, atlas, atlasMask)



