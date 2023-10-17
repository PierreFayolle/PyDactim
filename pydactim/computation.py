import nibabel as nib
import numpy as np

def calc_volume(mask_path):
    """ Calculate the volume of a mask 

    Parameters
    ----------
    mask_path : str
        Path of the mask

    """
    mask = nib.load(mask_path)
    pixdim = mask.header["pixdim"][1:4]
    volume = np.count_nonzero(mask.get_fdata())
    volumetrie = volume * pixdim[0] * pixdim[1] * pixdim[2]
    print("Volume :", round(volumetrie, 3), "cm3")
    return round(volumetrie/1000,3)

def a_calc_volume(mask_data, pixdim):
    """ Calculate the volume of a mask

    Parameters
    ----------
    mask_data : array
        Array of the mask

    pixdim : array
            Array of size 3 of the dimension of the voxels

    """
    volume = np.count_nonzero(mask_data)
    volumetrie = volume * pixdim[0] * pixdim[1] * pixdim[2]
    print("Volume :", round(volumetrie, 3), "cm3")
    return round(volumetrie/1000,3)