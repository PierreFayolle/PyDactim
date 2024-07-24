import os
import SimpleITK as sitk
import torchio as tio
import nibabel as nib
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from .brain_extraction import run_hd_bet
from dipy.segment.tissue import TissueClassifierHMRF
import itk
from numba import jit
import torchio as tio
import torch
from .models.mpunet import MPUnetPP2CBND

def n4_bias_field_correction(input_path, mask=False, force=True, suffix="corrected"):
    """ Correct the bias field correction.

    Parameters
    ----------
    input_path : str
        Nifti file path

    suffix : str
        Nifti file suffixe for the new generated image

    mask_path : str
        Nifti file path containing the mask of the bias field

    """
    print(f"INFO - Starting bias field correction for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Bias field correction already done for\n\t{input_path :}")
        if mask is True: 
            return output_path, mask_path
        else:
            return output_path

    nifti = sitk.ReadImage(input_path, sitk.sitkFloat32)
    mask_otsu = sitk.OtsuThreshold(nifti,0,1,200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    output = corrector.Execute(nifti, mask_otsu)
    print(f"INFO - Saving generated image at\n\t{output_path :}")

    sitk.WriteImage(output, output_path)

    if mask is True: 
        print(f"INFO - Saving generated mask at\n\t{mask_path :}")
        log_bias_field = corrector.GetLogBiasFieldAsImage(nifti)
        sitk.WriteImage(log_bias_field, mask_path)
        return output_path, mask_path
    else:
        return output_path

def registration(atlas_path, input_path, matrix=True, force=True, suffix="flirt"):
    """ Register a nifti image to an atlas (MNI152).

    Parameters
    ----------
    atlas_path : str
        Nifti file path which is the atlas 

    input_path : str
        Nifti file path to be registered

    suffix : str
        Nifti file suffixe for the new generated image

    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    matrix_path = input_path.replace(".nii.gz", "_transfo.tfm")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Registration already done for\n\t{input_path :}\n\t{atlas_path :}")
        if matrix == True:
            return output_path, matrix_path
        else:
            return output_path

    print(f"INFO - Starting registration for\n\t{input_path :}\n\t{atlas_path :}")
    fixed = sitk.ReadImage(atlas_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(input_path, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsPowell()
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerScalesFromPhysicalShift()

    outTx = R.Execute(fixed, moving)

    resampled = sitk.Resample(moving,
                                fixed,
                                outTx,
                                sitk.sitkLinear, 0.0,
                                moving.GetPixelIDValue())

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    sitk.WriteImage(resampled, output_path)

    if matrix is True: 
        sitk.WriteTransform(outTx, matrix_path)
        return output_path, matrix_path
    else:
        return output_path

def apply_transformation(atlas_path, input_path, matrix_path, force=True, suffix="flirt"):
    """ Apply a transformation to an image according the transformation matrix .

    Parameters
    ----------
    atlas_path : str
        Nifti file path which is the atlas 

    input_path : str
        Nifti file path to be registered

    matrix_path : str
        Matrix path that will be use to transform the input image

    suffix : str
        Nifti file suffixe for the new generated image

    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Registration already done for\n\t{input_path :}\n\t{atlas_path :}")
        return output_path

    print(f"INFO - Starting registration for\n\t{input_path :}\n\t{atlas_path :}")
    fixed = sitk.ReadImage(atlas_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(input_path, sitk.sitkFloat32)

    outTx = sitk.ReadTransform(matrix_path)

    resampled = sitk.Resample(moving,
                                fixed,
                                outTx,
                                sitk.sitkLinear,
                                0.0,
                                moving.GetPixelIDValue())

    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    sitk.WriteImage(resampled, output_path)

    return output_path

def resample(*values, suffix="resampled"):
    """ Resample images in the same space with a resample factor. If no resample factor is given, the first image become the fixed image and all others paths will be resampled on it

    Parameters
    ----------
    input_paths : str 
        All the image paths which need to be resampled on a fixed image or on a resampled value

    resample : int (optional)
        The resample value in which all the image will be transformed.

    
    Example
    -------
    resample(t1_path, 1) => the t1_path is resampled with 1mm voxel size
    resample(t1_path, t2_path) => t1_path is the fixed image, t2_path is the moving image
    resample(t1_path, t2_path, swi_path, 2) => all images are resampled with 2mm voxel size
    resample(t1_path) => Not working because if there is any resample value, 2 paths are mendatory
    """
    nl = "\n\t"
    temp = values
    if temp[-1] is not str:
        temp = temp[0:-1]
    print(f"INFO - Starting resampling for{nl}{nl.join(temp) :}")
    new_paths = []
    paths = []
    transform = None
    for value in values:
        if type(value) == str and os.path.exists(value):
            paths.append(value)

        elif type(value) == int or type(value) == float:
            transform = tio.Resample(value)

    if len(paths) > 0:
        if transform == None:
            if len(paths) >= 2:
                subject = tio.Subject(ref=tio.ScalarImage(paths.pop(0)))

                for path in paths:
                    subject.add_image(tio.ScalarImage(path), path)

                transform = tio.Resample("ref")
                subject = transform(subject)

                for image in subject.get_images_names():
                    if image != "ref":
                        new_path = image.replace(".nii.gz", "_" + suffix + ".nii.gz")
                        subject[image].save(new_path)
                        new_paths.append(new_path)
            else:
                raise ValueError("Need at least 2 valid paths (1 as ref, 1 as resampled) when no resampling value is given")
        else:
            subject = tio.Subject(ref=tio.ScalarImage(paths[0]))
            for path in paths:
                subject.add_image(tio.ScalarImage(path), path)

            subject = transform(subject)

            for image in subject.get_images_names():
                if image != "ref":
                    new_path = image.replace(".nii.gz", "_" + suffix + ".nii.gz")
                    subject[image].save(new_path)
                    new_paths.append(new_path)

    else:
        raise ValueError("All the given path were not valid")
    
    return new_paths

def histogram_matching(input_ref_path, input_path, suffix="hm"):
    """ Match the histogram of two images.

    Parameters
    ----------
    input_ref_path : str
        The fixed image on which the other image will be matched

    input_path : str 
        The moving image that needs to be filtered

    suffix : str
        Nifti file suffixe for the new generated image
    """
    print(f"INFO - Starting histogram matching for\n\t{input_ref_path :}\n\t{input_path :}")
    him = sitk.HistogramMatchingImageFilter()
    him.SetThresholdAtMeanIntensity(True)

    readerI = sitk.ImageFileReader()
    readerI.SetFileName(input_path)
    in_img = readerI.Execute()

    readerII = sitk.ImageFileReader()
    readerII.SetFileName(input_ref_path)
    ref = readerII.Execute()

    out_img = him.Execute(in_img,ref)

    writer = sitk.ImageFileWriter()

    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    print(f"INFO - Saving generated image at\n\t{output_path :}")

    writer.SetFileName(output_path)
    writer.Execute(out_img)

def apply_mask(input_path, mask_path, force=True, suffix="mask"):
    """ Apply a binary mask to an image sharing the same shape.

    Parameters
    ----------
    input_path : str
        The image path on which apply the mask 

    mask_path : str 
        The binary mask path 

    suffix : str
        Nifti file suffixe for the new generated image
    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Mask already applied for\n\t{input_path :}")
        return output_path

    print(f"INFO - Starting to apply mask for\n\t{input_path :}\n\t{mask_path :}")
    img = nib.load(input_path)
    mask = nib.load(mask_path)

    if mask.shape != img.shape:
        raise(f"ERROR - Shape of the mask does not match the main image :\n\t Image 1 : {img.shape} != Image 2 : {mask.shape}")

    final = mask.get_fdata() * img.get_fdata()

    print(f"INFO - Saving generated image at\n\t{output_path :}")

    nib.save(nib.Nifti1Image(final, img.affine), output_path)

    return output_path

def substract(input_ref_path, input_path, is_aligned=True, suffix="sub"):
    """ Match the histogram of two images.

    Parameters
    ----------
    input_ref_path : str
        The fixed image that will be substracted

    input_path : str 
        The image on which the other image will be substracted

    is_aligned : Boolean
        Boolean to check if the images are aligned. If not, it starts registration function before substraction

    suffix : str
        Nifti file suffixe for the new generated image
    """
    print(f"INFO - Starting substraction for\n\t{input_ref_path :}\n\t{input_path :}")
    ref = nib.load(input_ref_path)

    if is_aligned == False:
        input_aligned_path = registration(input_ref_path, input_path, suffix="flirt")
        img = nib.load(input_aligned_path)
    else:
        print(f"INFO - Assuming the images are already aligned. Registration has been skipped")
        img = nib.load(input_path)


    sub = ref.get_fdata() - img.get_fdata()

    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    print(f"INFO - Saving generated image at\n\t{output_path :}")

    nib.save(nib.Nifti1Image(sub, img.affine), output_path)

    return output_path

def susan(input_path, offset=3, force=True, suffix="susan"):
    """ Amplify borders by creating a susan variance map.
    Based on https://users.fmrib.ox.ac.uk/~steve/susan/susan/node2.html

    Parameters
    ----------
    input_path : str
        The image that will be used to create the susan variance map

    offset : int
        Size of the kernel

    suffix : str
        Nifti file suffixe for the new generated image
    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Susan variance map already computed for\n\t{input_path :}")
        return output_path

    print(f"INFO - Starting Susan variance map creation for\n\t{input_path :}")
    img = nib.load(input_path)
    img_data = img.get_fdata()

    @jit
    def susan(img_data):
        susan_data = np.zeros_like(img_data)
        for x in range(img_data.shape[0]):
            for y in range(img_data.shape[1]):
                for z in range(img_data.shape[2]):
                        values = img_data[x:x + offset, y:y + offset, z:z + offset]
                        mean = np.mean(values)
                        var = np.var(values)
                        susan_data[x,y,z] = np.abs(var - mean**2)
        return susan_data
    
    susan_data = susan(img_data)

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(nib.Nifti1Image(susan_data, img.affine), output_path)
    
    return output_path

def normalize(input_path, force=True, suffix="minmax"):
    """ Normalize image between 0 and 1.

    Parameters
    ----------
    input_path : str
        The image that will be normalized

    suffix : str
        Nifti file suffixe for the new generated image
    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Normalization already computed for\n\t{input_path :}")
        return output_path

    print(f"INFO - Starting normalization for\n\t{input_path :}")
    img = nib.load(input_path)
    img_data = img.get_fdata()
    norm_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(nib.Nifti1Image(norm_data, img.affine), output_path)

    return output_path

def crop(input_path, force=True, suffix="cropped"):
    """ Crop the maximum of noise in an image.

    Parameters
    ----------
    input_path : str
        The image that will bet cropped

    suffix : str
        Nifti file suffixe for the new generated image

    Returns
    -------
    cropped_idx : list if int
        List of the indices that was used to crop the input image

    """
    print(f"INFO - Starting automatic crop for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    
    mri = nib.load(input_path)
    mri_image = mri.get_fdata()
    
    z_dim, y_dim, x_dim = mri_image.shape

    # Calculate the threshold for the image intensity to determine the background
    threshold = np.mean(mri_image) + 2 * np.std(mri_image)

    # Find the maximum intensity value within a margin around the center of the image
    z_margin, y_margin, x_margin = z_dim // 4, y_dim // 4, x_dim // 4
    z_center, y_center, x_center = z_dim // 2, y_dim // 2, x_dim // 2
    center_intensity = np.max(mri_image[z_center - z_margin:z_center + z_margin,
                                        y_center - y_margin:y_center + y_margin,
                                        x_center - x_margin:x_center + x_margin])

    # Set a lower threshold to ensure that the cube contains some useful information
    lower_threshold = center_intensity * 0.75

    # Determine the starting and ending indices for each axis
    z_start = np.argmax(np.max(mri_image, axis=(1, 2)) > threshold)
    z_end = z_dim - np.argmax(np.max(mri_image[::-1, :, :], axis=(1, 2)) > threshold)
    y_start = np.argmax(np.max(mri_image, axis=(0, 2)) > threshold)
    y_end = y_dim - np.argmax(np.max(mri_image[:, ::-1, :], axis=(0, 2)) > threshold)
    x_start = np.argmax(np.max(mri_image, axis=(0, 1)) > threshold)
    x_end = x_dim - np.argmax(np.max(mri_image[:, :, ::-1], axis=(0, 1)) > threshold)

    # Make sure the cube contains some useful information
    while (mri_image[z_start:z_end, y_start:y_end, x_start:x_end].max() < lower_threshold):
        z_start += 1
        z_end -= 1
        y_start += 1
        y_end -= 1
        x_start += 1
        x_end -= 1

    cropped_idx = [z_start, z_end, y_start, y_end, x_start, x_end]
    if os.path.exists(output_path) and not force:
        print(f"INFO - Automatic crop already done for\n\t{input_path :}")
        return output_path, cropped_idx
    
    # Print the starting and ending indices for each axis
    # print(f"z_start = {z_start}, z_end = {z_end}")
    # print(f"y_start = {y_start}, y_end = {y_end}")
    # print(f"x_start = {x_start}, x_end = {x_end}")

    # Crop the MRI image to extract the cube
    cropped_mri_image = mri_image[z_start:z_end, y_start:y_end, x_start:x_end]
    nib.save(nib.Nifti1Image(cropped_mri_image, mri.affine), output_path)

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path, cropped_idx

def apply_crop(input_path, crop, force=True, suffix="cropped"):
    """ Crop the image according crop indices.

    Parameters
    ----------
    input_path : str
        The image that will get the new affine from the reference image

    crop : list
        List of 6 integers to determine the cropped indices 

    suffix : str
        Nifti file suffixe for the new generated image
    """
    print(f"INFO - Starting automatic crop for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Automatic crop already done for\n\t{input_path :}")
        return output_path

    mri = nib.load(input_path)
    mri_image = mri.get_fdata()

    cropped_mri_image = mri_image[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
    nib.save(nib.Nifti1Image(cropped_mri_image, mri.affine), output_path)

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def copy_affine(input_ref_path, input_path, force=True, suffix="affined"):
    """ Match the histogram of two images.

    Parameters
    ----------
    input_ref_path : str
        The image used as reference

    input_path : str
        The image that will get the new affine from the reference image

    suffix : str
        Nifti file suffixe for the new generated image
    """
    print(f"INFO - Starting copying affine for\n\t{input_path :}\n\tWith the following image\n\t{input_ref_path :}")

    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Copying affine already done for\n\t{input_path :}")
        return output_path

    subject = tio.Subject(
        ref=tio.ScalarImage(input_ref_path),
        img=tio.ScalarImage(input_path)
    )
    subject = tio.CopyAffine('ref')(subject)

    for image in subject.get_images_names():
        if image == "img":
            subject[image].save(image.replace(".nii.gz", "_" + suffix + ".nii.gz"))

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def skull_stripping(input_path, model_path, mask=False, force=True, suffix="brain"):
    """ HD-BET to extract the brain of any anatomic mri image.
    This script is a slightly modified version of the HD-BET script that can be found at the following link:
    https://github.com/MIC-DKFZ/HD-BET

    Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W,
    Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificia
    neural networks. arXiv preprint arXiv:1901.11341, 2019.
    https://doi.org/10.1002/hbm.24750

    Parameters
    ----------
    input_path : str
        The image path in which the brain will be extracted

    model_path : str
        Path of the directory containing the folds of the model. 
        The files can be downloaded from the following path: 
        https://zenodo.org/record/2540695

    mask : bool
        To get the binary mask of the brain
        
    suffix : str
        Nifti file suffixe for the new generated image

    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")

    print(f"INFO - Starting skull stripping for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Skull stripping already done for\n\t{input_path :}")
        if mask == True:
            return output_path, mask_path
        else:
            return output_path
    
    img = nib.load(input_path)
    data = np.transpose(img.get_fdata(), (2,1,0)) 
    sitk_data = sitk.GetImageFromArray(data)
    img_out, mask_out = run_hd_bet(sitk_data, model_path=model_path)
    img_out = np.transpose(img_out, (2,1,0)) 
    mask_out = np.transpose(mask_out, (2,1,0)) 

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(nib.Nifti1Image(img_out, img.affine), output_path)
    if mask == True:
        print(f"\t{mask_path :}")
        nib.save(nib.Nifti1Image(mask_out, img.affine), mask_path)
        return output_path, mask_path
    else:
        return output_path

def remove_small_object(input_path, min_size, force=False, suffix="filtered"):
    """ Remove small objects above a minimum size

    Parameters
    ----------
    input_path : str
        Nifti file path

    min_size : int
        Minimum size of objects to be removed

    suffix : str
        Nifti file suffixe for the new generated image

    """
    print(f"INFO - Starting to remove small objects of size inferior to {min_size} for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Small object removing already done for\n\t{input_path :}")
        return output_path
    
    img = nib.load(input_path)
    img_data = img.get_fdata()

    img_out = morphology.remove_small_objects(img_data.astype(bool), min_size=min_size)

    nib.save(nib.Nifti1Image(img_out.astype(np.int16), img.affine), output_path)
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def tissue_classifier(input_path, pve=True, force=True, suffix="fast"):
    """ Normalize image between 0 and 1.

    Parameters
    ----------
    input_data : str
        The image path that will be normalized

    suffix : str
        Nifti file suffixe for the new generated image
    """
    print(f"INFO - Starting tissue classification for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    pve_path = output_path.replace(".nii.gz", "_pve.nii.gz")

    if os.path.exists(output_path) and not force:
        print(f"INFO - Tissue classification already done for\n\t{input_path :}")
        if pve == True:
            return output_path, pve_path
        else:
            return output_path

    nclass = 3
    beta = 0.1

    img = nib.load(input_path)
    img_data = img.get_fdata()

    hmrf = TissueClassifierHMRF(verbose=False)
    _, img_out, mask_out = hmrf.classify(img_data, nclass, beta)

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(nib.Nifti1Image(img_out, img.affine), output_path)
    if pve == True:
        print(f"\t{pve_path :}")
        nib.save(nib.Nifti1Image(mask_out, img.affine), pve_path)
        return output_path, pve_path
    else:
        return output_path

def add_tissue_class(input_path, mask_path, num_class, force=True, suffix="masked"):
    print(f"INFO - Starting to add a new class for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - New added class already done for\n\t{input_path :}")
        return output_path
    
    img = nib.load(input_path)
    img_data = img.get_fdata()

    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()

    img_data[mask_data > 0] = num_class

    nib.save(nib.Nifti1Image(img_data, img.affine), output_path)
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def extract_dim(input_path, dim, force=True, suffix=""):
    print(f"INFO - Starting to extract the dimension {dim} for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_dim" + str(dim) + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Extracted dimension already done for\n\t{input_path :}")
        return output_path
    
    img = nib.load(input_path)
    img_data = img.get_fdata()

    img_data = img_data[..., dim]
 
    nib.save(nib.Nifti1Image(img_data, img.affine), output_path)
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def prediction_glioma(input_path, model_path, landmark_path, force=True, suffix="glioma_predicted"):
    print(f"INFO - Starting glioma prediction for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Glioma segmentation already done for\n\t{input_path :}")
        return output_path
    
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNETR
    model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(176, 208, 160),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad((176, 208, 160)),
            tio.HistogramStandardization({"image": np.load(landmark_path)}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

    ds = tio.SubjectsDataset([
        tio.Subject(image = tio.ScalarImage(input_path))], 
        transform=transform
    )[0]

    affine = nib.load(input_path).affine
    with torch.no_grad():
        img = ds["image"]["data"]
        val_inputs = torch.unsqueeze(img, 1)
        val_outputs = sliding_window_inference(val_inputs.cuda(), (176, 208, 160), 4, model, overlap=0.25)
        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0].astype(float)
        pred_map = tio.LabelMap(tensor=np.expand_dims(val_outputs, 0), affine=ds.image.affine)
        ds.add_image(pred_map, "pred")
        ds_inv = ds.apply_inverse_transform(warn=True)
        val_outputs = ds_inv["pred"].data.numpy().squeeze()

        nib.save(nib.Nifti1Image(val_outputs.squeeze(), affine), output_path)
    
    output_path = remove_small_object(output_path, 5000, force=True)
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

def prediction_multiple_sclerosis(input_path, model_path, landmark_path, force=True, suffix="ms_predicted"):
    print(f"INFO - Starting multiple sclerosis prediction for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Multiple sclerosis segmentation already done for\n\t{input_path :}")
        return output_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MPUnetPP2CBND(1, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        # tio.CopyAffine('image'),
        tio.CropOrPad((192, 192, 224)),
        tio.HistogramStandardization({"image": landmark_path}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

    subject = tio.SubjectsDataset([ tio.Subject( image = tio.ScalarImage(input_path) )],  transform=transform )[0]

    patch_overlap = 16
    grid_sampler = tio.inference.GridSampler(
        subject,
        64,
        patch_overlap
    )

    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=16)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")

    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['image'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            batch_pred = model(inputs)
            aggregator.add_batch(batch_pred, locations)
    output = aggregator.get_output_tensor()
    y_pred = output[0].numpy().reshape(subject.image.shape)

    val_inputs  = subject.image.numpy().squeeze()
    # val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0].astype(float)
    val_outputs = y_pred.squeeze()

    affine = nib.load(input_path).affine

    output_path = input_path.replace(".nii", "_predicted.nii")
    input_path = input_path.replace(".nii", "_input.nii")
    nib.save(
        nib.Nifti1Image(val_inputs, affine), 
        input_path
    )

    val_outputs[val_outputs <= 0.01] = 0
    val_outputs[val_outputs > 0.01] = 1
    nib.save(
        nib.Nifti1Image(val_outputs, affine), 
        output_path
    )

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path, input_path

def uncertainty_prediction_glioma(input_path, model_path, force=True, suffix="uncertainty"):
    print(f"INFO - Starting glioma uncertainty prediction for\n\t{input_path :}")
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Uncertainty glioma segmentation already done for\n\t{input_path :}")
        return output_path
    
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNETR
    from tqdm.auto import tqdm, trange
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(176, 208, 160),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad((176, 208, 160)),
            tio.HistogramStandardization({"image": np.load("E:/Leo/script/results/landmarks.npy")}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomFlip(),
            tio.RandomAffine(p=0.5),
    ])

    subject = tio.Subject(image = tio.ScalarImage(input_path)) 
    affine = nib.load(input_path).affine

    results = []
    for _ in trange(20):
        subject = transform(subject)
        inputs = subject.image.data.to(device)
        
        with torch.no_grad():
            inputs = torch.unsqueeze(inputs, 1)
            outputs = sliding_window_inference(inputs, (176, 208, 160), 4, model, overlap=0.25)
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0].astype(float)
        pred_map = tio.LabelMap(tensor=np.expand_dims(outputs, 0), affine=subject.image.affine)
        subject.add_image(pred_map, "pred")
        subject_inv = subject.apply_inverse_transform(warn=True)
        results.append(subject_inv["pred"].data)

    result = torch.stack(results).long()
    tta_result_tensor = result.mode(dim=0).values

    different = torch.stack([
        tensor != tta_result_tensor
        for tensor in results
    ])
    uncertainty = different.float().mean(dim=0)
    uncertainty_img = tio.ScalarImage(tensor=uncertainty, affine=subject.image.affine)
    subject.add_image(uncertainty_img, "uncertainty")

    uncertainty_img = uncertainty_img.data.numpy().squeeze()
    nib.save(nib.Nifti1Image(uncertainty_img, affine), output_path)

    print(f"INFO - Saving generated image at\n\t{output_path :}")
    return output_path

    print(f"INFO - Starting glioma prediction for\n\t{input_path :}")
    
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNETR
    model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(176, 208, 160),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CopyAffine('image'),
            tio.CropOrPad((176, 208, 160)),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

    ds = tio.SubjectsDataset([
        tio.Subject(image = tio.ScalarImage(input_path))], 
        transform=transform
    )[0]

    affine = nib.load(input_path).affine
    with torch.no_grad():
        img = ds["image"]["data"]
        val_inputs = torch.unsqueeze(img, 1)
        val_outputs = sliding_window_inference(val_inputs.cuda(), (176, 208, 160), 4, model, overlap=0.25)

        pred_map = tio.LabelMap(tensor=val_outputs[0], affine=ds.image.affine)
        ds.add_image(pred_map, "pred")
        val_outputs = ds.apply_inverse_transform(warn=True).pred.data

        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0].astype(float)
        val_outputs_path = input_path.replace(".nii", "_predicted_unetr.nii")
        nib.save(nib.Nifti1Image(val_outputs, affine), val_outputs_path)
    
    print(f"INFO - Saving generated image at\n\t{val_outputs_path :}")
    return val_outputs_path

if __name__ == "__main__":
    landmark_path = r"E:\SEP_IA\results_7T\final\landmarks.npy"
    model_path = r"E:\SEP_IA\results_7T\final\patches_epoch_298.pth"
    input_path = r"E:\SEP_IA\data\sub-001\ses-01\anat\sub-001_ses-01_FLAIR.nii.gz"

    prediction_multiple_sclerosis(input_path, model_path, landmark_path)