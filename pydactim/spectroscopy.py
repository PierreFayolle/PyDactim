import os
import nibabel as nib
import numpy as np
import pydicom
import math

from .utils import load_dicom

def generate_mask(svs_path, nii_path, force=True, output_suffixe="mask"):
    """ Create a 3D mask of the spectroscopy volume of interest

    Parameters
    ----------
    svs_path : str 
        Path of the spectroscopy DICOM file
    
    nii_path : str
        Path of the reference volume

    output_suffixe : str
        Nifti file suffixe for the new generated image
    """
    output_path = nii_path.replace(".nii.gz", "_" + output_suffixe + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Mask already generated for\n\t{nii_path :}")
        return output_path

    ref = nib.load(nii_path)
    ref_array = ref.get_fdata()

    pixdim = ref.header["pixdim"][1:4]

    csa_header = load_dicom(svs_path)

    voi = np.array([
        round(float(csa_header["sSpecPara.sVoI.dReadoutFOV"]) / pixdim[0]),
        round(float(csa_header["sSpecPara.sVoI.dPhaseFOV"])   / pixdim[1]), 
        round(float(csa_header["sSpecPara.sVoI.dThickness"])  / pixdim[2])
    ], dtype=np.int8)

    pcg_centre = [
        float(csa_header["sSpecPara.sVoI.sPosition.dSag"]),
        float(csa_header["sSpecPara.sVoI.sPosition.dCor"]), 
        float(csa_header["sSpecPara.sVoI.sPosition.dTra"])
    ]
    pcg_centre.append(1)
    pcg_centre_index = np.matmul(np.linalg.inv(ref.affine), pcg_centre).squeeze()

    mask = np.zeros((ref_array.shape))
    mask[
        round(pcg_centre_index[0] - voi[0]/2) : round(pcg_centre_index[0] + voi[0]/2),
        round(pcg_centre_index[1] - voi[1]/2) : round(pcg_centre_index[1] + voi[1]/2),
        round(pcg_centre_index[2] - voi[2]/2) : round(pcg_centre_index[2] + voi[2]/2),
    ] = 1

    mask = np.ma.masked_where(mask==1, mask)
    mask = mask[::-1, ::-1, :]
    mask_nii = nib.Nifti1Image(mask, ref.affine)
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(mask_nii, output_path)
    return output_path

def generate_temporal_spectre(svs_path, spectre_path, save=True):
    dcm = pydicom.dcmread(svs_path)
    spectro_data = dcm[0x5600,0x0020].value
    spectre = np.fromstring(spectro_data, np.float32)
    np.save(spectre_path, spectre)
    if save:
        return spectre_path
    else:
        return spectre
    
def generate_frequency_spectre(path):
    if os.path.endswith(".dcm"):
        temporal_spectre = generate_temporal_spectre(path, save=False) 
    else:
        temporal_spectre = np.load(path)

    shape = int(temporal_spectre.shape[0]/2)
    complex_spectre = np.zeros((shape,), dtype=complex)
    cpt = 0 
    for i in range(shape):
        compl = temporal_spectre[cpt] + 1j * temporal_spectre[cpt+1]
        complex_spectre[i] = np.array(compl, dtype=complex)
        cpt += 2

    frequency_spectre = np.fft.fftshift(np.fft.fft(complex_spectre))
    np.save(path.replace(".npy", "_fft.npy").replace(".dcm", "_fft.npy"), frequency_spectre)
    return frequency_spectre