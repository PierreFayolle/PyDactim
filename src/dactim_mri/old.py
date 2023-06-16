import os
import SimpleITK as sitk
import torchio as tio
import nibabel as nib
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import itk

def skull_stripping(input_path, mask=False, force=True, suffix="brain"):
    """ Extract brain from a Nifti image.

    Parameters
    ----------
    input_path : str
        Nifti file path

    suffix : str
        Nifti file suffixe for the new generated image

    mask : bool
        Nifti file path containing the brain binarized mask

    bse : str
        Base directory path of bse.exe in the Brainsuite directory

    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")

    if os.path.exists(output_path) and not force:
        if mask == True:
            return output_path, mask_path
        else:
            return output_path
        
    # Path of the executable BSE script of Brainsuite
    temp = os.getcwd()
    os.chdir("C:/Program Files/BrainSuite19b/bin/")
    BSE = "bse.exe"

    if mask == False:
        command = f"{BSE} -i \"{input_path}\" -o \"{output_path}\" --auto"
    else:
        command = f"{BSE} -i \"{input_path}\" -o \"{output_path}\" --mask \"{mask_path}\" --auto"

    print(command)
    os.system(command)
    os.chdir(temp)
    if mask == True:
        return output_path, mask_path
    else:
        return output_path

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
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 2000, 0.5, 1e-6)
    R.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving,
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
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

        elif type(value) == int:
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

def variance(input_path, offset=9, force=True, suffix="var"):
    """ Amplify borders by creating a variance map.

    Parameters
    ----------
    input_path : str
        The image that will be used to create the variance map

    offset : int
        offset of the variance, the highest it is, the strongest the borders

    suffix : str
        Nifti file suffixe for the new generated image
    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    if os.path.exists(output_path) and not force:
        print(f"INFO - Variance map already computed for\n\t{input_path :}")
        return output_path

    print(f"INFO - Starting variance map creation for\n\t{input_path :}")
    img = nib.load(input_path)
    img_data = img.get_fdata()
    var_data = np.zeros_like(img_data)
    for x in range(img_data.shape[0]):
        for y in range(img_data.shape[1]):
            for z in range(img_data.shape[2]):
                if img_data[x,y,z] != 0:
                    variance = np.var(img_data[x-offset//2+1:x+offset//2+1,y-offset//2+1:y+offset//2+1,z-offset//2+1:z+offset//2+1])
                    var_data[x,y,z] = variance
    
    print(f"INFO - Saving generated image at\n\t{output_path :}")
    nib.save(nib.Nifti1Image(var_data, img.affine), output_path)
    
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

def crop_mri(input_path, force=True, suffix="cropped"):
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
    if os.path.exists(output_path) and not force:
        print(f"INFO - Automatic crop already done for\n\t{input_path :}")
        return output_path, None
    
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

    # Increase the shape of the cropped image to make it divisible by 8
    z_len, y_len, x_len = z_end - z_start, y_end - y_start, x_end - x_start
    z_mod, y_mod, x_mod = z_len % 8, y_len % 8, x_len % 8
    if z_mod != 0:
        z_extra = 8 - z_mod
        z_start -= z_extra // 2
        z_end += (z_extra + 1) // 2
    if y_mod != 0:
        y_extra = 8 - y_mod
        y_start -= y_extra // 2
        y_end += (y_extra + 1) // 2
    if x_mod != 0:
        x_extra = 8 - x_mod
        x_start -= x_extra // 2
        x_end += (x_extra + 1) // 2

    # Print the starting and ending indices for each axis
    print(f"z_start = {z_start}, z_end = {z_end}")
    print(f"y_start = {y_start}, y_end = {y_end}")
    print(f"x_start = {x_start}, x_end = {x_end}")

    # Crop the MRI image to extract the cube
    cropped_mri_image = mri_image[z_start:z_end, y_start:y_end, x_start:x_end]
    nib.save(nib.Nifti1Image(cropped_mri_image, mri.affine), output_path)
    cropped_idx = [z_start, z_end, y_start, y_end, x_start, x_end]

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

def copy_affine(input_ref_path, input_path, force=True, suffix="aff"):
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

def bet(input_path, mask=False, force=True, suffix="brain"):
    """ Extract brain from a Nifti image

    Parameters
    ----------
    input_path : str
        Nifti file path

    suffix : str
        Nifti file suffixe for the new generated image

    mask : bool
        Nifti file path containing the brain binarized mask

    bse : str
        Base directory path of bse.exe in the Brainsuite directory

    """
    output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")

    if os.path.exists(output_path) and not force:
        if mask == True:
            return output_path, mask_path
        else:
            return output_path
        
    # Path of the executable HD-BET script
    temp = os.getcwd()
    os.chdir("C:/Users/467355/Documents/HD-BET-master/HD_BET")
    HDBET = "python hd-bet"

    if mask == False:
        command = f"{HDBET} -i \"{input_path}\" -o \"{output_path}\""
    else:
        command = f"{HDBET} -i \"{input_path}\" -o \"{output_path}\""

    print(command)
    os.system(command)
    os.chdir(temp)
    if mask == True:
        return output_path, mask_path
    else:
        return output_path

def remove_small_object(input_path, min_size, force=False, suffix="opened"):
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
