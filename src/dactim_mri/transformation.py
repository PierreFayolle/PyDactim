import os
import SimpleITK as sitk
import torchio as tio
import nibabel as nib
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import itk

class Dactim():
    def __init__(self):
        pass

    def init(self, status):
        self.status = status
        self.precise_path = ""
        self.exit = False

        self.nii_path = None
        self.nii = None
        self.nii_data = None
        self.nii_mask = None
        self.nii_mask_data = None
        self.nii_ref = None
        self.nii_ref_data = None

        self.output_path = None
        self.mask_path = None

    def load_data(self, input_data, atlas_data=False, mask_data=False):
        if type(input_data) is str:
            self.nii_path = input_data
            self.precise_path = f"for\n\t{input_data :}"
            self.nii = nib.load(input_data)
            self.nii_data = self.nii.get_fdata()
        else:
            self.nii_data = input_data

        if type(atlas_data) is str:
            self.nii_atlas_path = atlas_data
            self.precise_path = f"\n\t{atlas_data :}"
            self.nii_atlas = nib.load(atlas_data)
            self.nii_atlas_data = self.nii_atlas.get_fdata()
        else:
            self.nii_atlas_data = atlas_data

        if type(mask_data) is str:
            self.nii_mask_path = mask_data
            self.precise_path = f"\n\t{mask_data :}"
            self.nii_mask = nib.load(mask_data)
            self.nii_mask_data = self.nii_mask.get_fdata()
        else:
            self.nii_mask_data = mask_data

    def generate_output_path(self, input_path, suffix, force, other=None, other_format=None):
        self.output_path = input_path.replace(".nii.gz", "_" + suffix + ".nii.gz")
        if other is not None:
            self.other_path = self.output_path.replace(".nii.gz", "_"+other+"."+other_format)
        if os.path.exists(self.output_path) and not force:
            print(f"INFO - {self.status} already done for\n\t{input_path :}")
            if other is not None:
                if os.path.exists(self.other_path):
                    self.exit = True

    def nib_save(self, extra=False, extra_data=None):
        print(f"INFO - Saving generated image at\n\t{self.output_path :}")
        nib.save(nib.Nifti1Image(self.output_data, self.nii.affine), self.output_path)
        if extra is False:
            return self.output_path
        else:
            print(f"INFO - Saving generated image at\n\t{self.output_path :}")
            nib.save(nib.Nifti1Image(extra_data, self.nii.affine), self.other_path)
            return self.output_path, self.other_path

    def sitk_save(self, extra=False, extra_data=None):
        print(f"INFO - Saving generated image at\n\t{self.output_path :}")
        sitk.WriteImage(self.output_data, self.output_path)
        if extra is False:
            return self.output_path
        else:
            print(f"INFO - Saving generated image at\n\t{self.output_path :}")
            sitk.WriteImage(extra_data, self.other_path)
            return self.output_path, self.other_path

    def itk_save(self, extra=False, extra_data=None):
        print(f"INFO - Saving generated image at\n\t{self.output_path :}")
        itk.imwrite(self.output_data, self.output_path)
        if extra is False:
            return self.output_path
        else:
            print(f"INFO - Saving generated image at\n\t{self.output_path :}")
            itk.imwrite(extra_data, self.other_path)
            return self.output_path, self.other_path

    def nib_to_sitk(self, data):
        # sitk is (x y z) while np is (z y x)
        data = np.transpose(data, (2,1,0)) 
        sitk_data = sitk.GetImageFromArray(data)
        # sitk_data.SetOrigin((0, 0, 0))
        return sitk_data

    def plot(self, input_data, pixdim=None, slice=None):
        print(f"INFO - Plotting image")
        if type(input_data) is str:
            img = nib.load(input_data)
            data = img.get_fdata()
            pixdim = img.header["pixdim"][1:4].astype(float)
            title = f"{os.path.basename(input_data)} (shape={data.shape}, pixdim={pixdim})"
        else:
            data = input_data
            if pixdim is None:
                pixdim = [1,1,1] # Default pixdim
                print("WARNING - 'Pixdim' parameter is not defined, hence the pixel dimensions will be set to default (can deform the image)")
            title = f"shape={data.shape}, pixdim={pixdim}"

        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

        indices = np.array(data.shape) // 2
       
        i, j, k = indices
        if slice is not None: k = slice
        sag = np.rot90(np.fliplr(data[i, :, :]), -1)
        cor = np.rot90(np.fliplr(data[:, j, :]), -1)
        tra = np.rot90(np.fliplr(data[:, :, k]), -1)

        sag_aspect = pixdim[2] / pixdim[1]
        ax1.imshow(sag, aspect=sag_aspect, cmap="gray")
        ax1.set_title(f'Sag (slice={str(i)})', y=0.9)
        ax1.axis('off')

        cor_aspect = pixdim[2] / pixdim[0]
        ax2.imshow(cor, aspect=cor_aspect, cmap="gray")
        ax2.set_title(f'Cor (slice={str(j)})', y=0.9)
        ax2.axis('off')

        tra_aspect = pixdim[1] / pixdim[0]
        ax3.imshow(tra, aspect=tra_aspect, cmap="gray")
        ax3.set_title(f'Tra (slice={str(k)})', y=0.9)
        ax3.axis('off')

        fig.text(0.5, 0.95, title, horizontalalignment="center")
        # plt.savefig(r'E:\Leo\data\sub-001\ses-01\anat\myfig.png')
        plt.tight_layout()
        plt.show()

    def n4_bias_field_correction(self, input_data, pixdim=None, mask=False, force=True, suffix="corrected"):
        """ Correct the bias field.

        Parameters
        ----------
        input_data : str/array
            Nifti file path or array that will be corrected

        pixdim : array, optional
            Array of size 3 of the dimension of the voxels

        mask : bool, optional
            Get the mask of the bias field
            
        suffix : str
            Nifti file suffixe for the new generated image

        """
        self.init("Bias field correction")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force, other="bias_field", other_format="nii.gz") 
            if self.exit and mask: return self.output_path, self.other_path
            elif self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting bias field correction {self.precise_path}")

        self.nii_data = self.nib_to_sitk(self.nii_data)
        if pixdim is not None: sitk_data.SetSpacing(pixdim.astype(float))

        # nifti = sitk.ReadImage(input_path, sitk.sitkFloat32)
        mask_otsu = sitk.OtsuThreshold(sitk_data,0,1,200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        self.output_data = corrector.Execute(sitk_data, mask_otsu)
        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_data)
        
        if type(input_data) is str: return self.sitk_save(extra=mask, extra_data=log_bias_field)
        else: 
            if mask: return sitk.GetArrayViewFromImage(self.output_data), sitk.GetArrayFromImage(log_bias_field)
            else: return sitk.GetArrayViewFromImage(self.output_data)

    def registration(self, atlas_data, input_data, matrix=False, elastix=True, force=True, suffix="flirt"):
        """ Register a nifti image to an atlas.

        Parameters
        ----------
        atlas_data : str
            Atlas that will be used as fixed image

        input_data : str
            Image that will be registered

        matrix : bool, optional
            Get the transformation matrix

        elastix : bool, optional
            Make the registration elastix

        suffix : str
            Nifti file suffixe for the new generated image

        """
        self.init("Registeration")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force, other="matrix", other_format="mat") 
            if self.exit and matrix: return self.output_path, self.other_path
            elif self.exit: return self.output_path

        self.load_data(input_data, atlas_data=atlas_data)
        print(f"INFO - Starting registration {self.precise_path}")

        if elastix:
            fixed = itk.imread(atlas_data, itk.F)
            moving = itk.imread(input_data, itk.F)

            self.output_data, transformation_matrix = itk.elastix_registration_method(fixed, moving)
            
            if type(input_data) is str: return self.itk_save(extra=matrix, extra_data=transformation_matrix)
            else: 
                if matrix: return sitk.GetArrayViewFromImage(self.output_data), transformation_matrix
                else: return sitk.GetArrayViewFromImage(self.output_data)
        else:
            fixed = sitk.ReadImage(atlas_data, sitk.sitkFloat32)
            moving = sitk.ReadImage(input_data, sitk.sitkFloat32)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed, 
                moving, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetInterpolator(sitk.sitkLinear)
            R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-8, convergenceWindowSize=10)
            R.SetOptimizerScalesFromPhysicalShift()
            R.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
            R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
            R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            R.SetInitialTransform(initial_transform, inPlace=False)
            transformation_matrix = R.Execute(fixed, moving)

            self.output_data = sitk.Resample(moving, fixed, transformation_matrix, sitk.sitkLinear, 0.0, moving.GetPixelID())

            if type(input_data) is str: return self.sitk_save(extra=matrix, extra_data=transformation_matrix)
            else: 
                if matrix: return sitk.GetArrayViewFromImage(self.output_data), transformation_matrix
                else: return sitk.GetArrayViewFromImage(self.output_data)

    def histogram_matching(self, atlas_data, input_data, force=True, suffix="hm"):
        """ Match the histogram of two images.

        Parameters
        ----------
        atlas_data : str/array
            The fixed image on which the other image will be matched. Can be a path or an array

        input_data : str/array
            The moving image that needs to be filtered. Can be a path or an array

        suffix : str
            Nifti file suffixe for the new generated image
        """
        self.init("Histogram matching")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data, atlas_data=atlas_data)
        print(f"INFO - Starting histogram matching for {self.precise_path}")

        him = sitk.HistogramMatchingImageFilter()
        him.SetThresholdAtMeanIntensity(True)

        self.nii_data = self.nib_to_sitk(self.nii_data)
        self.nii_atlas_data = self.nib_to_sitk(self.nii_atlas_data)

        self.output_data = sitk.GetArrayFromImage(him.Execute(self.nii_data, self.nii_atlas_data))
        self.output_data = np.transpose(self.output_data, (2,1,0))

        if type(input_data) is str: return self.nib_save()
        else: return sitk.GetArrayViewFromImage(self.output_data)

    def normalize(self, input_data, force=True, suffix="minmax"):
        """ Normalize image between 0 and 1.

        Parameters
        ----------
        input_data : str/array
            The image that will be normalized. Can be a path or an array

        suffix : str
            Nifti file suffixe for the new generated image
        """
        self.init("Normalization")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting normalization {self.precise_path}")

        self.output_data = (self.nii_data - self.nii_data.min()) / (self.nii_data.max() - self.nii_data.min())
        
        if type(input_data) is str: return self.nib_save()
        else: return self.output_data
        
    def variance(self, input_data, offset=9, force=True, suffix="var"):
        """ Amplify borders by creating a variance map.

        Parameters
        ----------
        input_data : str/array
            The image that will be used to create the variance map. Can be a path or an array

        offset : int
            offset of the variance, the highest it is, the strongest the borders

        suffix : str
            Nifti file suffixe for the new generated image
        """
        self.init("Variance map")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting variance map {self.precise_path}")

        var_data = np.zeros_like(self.nii_data)
        for x in range(self.nii_data.shape[0]):
            for y in range(self.nii_data.shape[1]):
                for z in range(self.nii_data.shape[2]):
                    if self.nii_data[x,y,z] != 0:
                        variance = np.var(self.nii_data[x-offset//2+1:x+offset//2+1,y-offset//2+1:y+offset//2+1,z-offset//2+1:z+offset//2+1])
                        var_data[x,y,z] = variance

        if type(input_data) is str: return self.nib_save()
        else: return self.output_data
    
    def crop_mri(self, input_data, force=True, suffix="cropped"):
        """ Crop the maximum of noise in an image.

        Parameters
        ----------
        input_data : str/array
            The image that will bet cropped. Can be a path or an array

        suffix : str
            Nifti file suffixe for the new generated image

        Returns
        -------
        cropped_idx : list if int
            List of the indices that was used to crop the input image

        """
        self.init("Automatic crop")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting automatic crop {self.precise_path}")
        
        z_dim, y_dim, x_dim = self.nii_data.shape

        # Calculate the threshold for the image intensity to determine the background
        threshold = np.mean(self.nii_data) + 2 * np.std(self.nii_data)

        # Find the maximum intensity value within a margin around the center of the image
        z_margin, y_margin, x_margin = z_dim // 4, y_dim // 4, x_dim // 4
        z_center, y_center, x_center = z_dim // 2, y_dim // 2, x_dim // 2
        center_intensity = np.max(self.nii_data[z_center - z_margin:z_center + z_margin,
                                            y_center - y_margin:y_center + y_margin,
                                            x_center - x_margin:x_center + x_margin])

        # Set a lower threshold to ensure that the cube contains some useful information
        lower_threshold = center_intensity * 0.75

        # Determine the starting and ending indices for each axis
        z_start = np.argmax(np.max(self.nii_data, axis=(1, 2)) > threshold)
        z_end = z_dim - np.argmax(np.max(self.nii_data[::-1, :, :], axis=(1, 2)) > threshold)
        y_start = np.argmax(np.max(self.nii_data, axis=(0, 2)) > threshold)
        y_end = y_dim - np.argmax(np.max(self.nii_data[:, ::-1, :], axis=(0, 2)) > threshold)
        x_start = np.argmax(np.max(self.nii_data, axis=(0, 1)) > threshold)
        x_end = x_dim - np.argmax(np.max(self.nii_data[:, :, ::-1], axis=(0, 1)) > threshold)

        # Make sure the cube contains some useful information
        while (self.nii_data[z_start:z_end, y_start:y_end, x_start:x_end].max() < lower_threshold):
            z_start += 1
            z_end -= 1
            y_start += 1
            y_end -= 1
            x_start += 1
            x_end -= 1

        # Crop the MRI image to extract the cube
        self.output_data = self.nii_data[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_idx = [z_start, z_end, y_start, y_end, x_start, x_end]

        if type(input_data) is str: return self.nib_save(), cropped_idx
        else: return self.output_data, cropped_idx

    def apply_crop(self, input_data, crop, force=True, suffix="cropped"):
        """ Crop the image according crop indices.

        Parameters
        ----------
        input_data : str/array
            The image that will be cropped. Can be a path or an array

        crop : list
            List of 6 integers to determine the cropped indices 

        suffix : str
            Nifti file suffixe for the new generated image
        """
        self.init("Applying crop")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting crop with shape {crop} {self.precise_path}")

        self.output_data = self.nii_data[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]

        if type(input_data) is str: return self.nib_save()
        else: return self.output_data

    def remove_small_object(self, input_data, min_size, force=False, suffix="opened"):
        """ Remove small objects above a minimum size

        Parameters
        ----------
        input_data : str/array
            Nifti file path of the binary mask in which will be removed all the small objects. Can be a path or an array

        min_size : int
            Minimum size of objects to be removed

        suffix : str
            Nifti file suffixe for the new generated image

        """
        self.init("Remove small objects")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data)
        print(f"INFO - Starting to remove small objects {self.precise_path}")
        
        self.output_data = morphology.remove_small_objects(self.nii_data.astype(bool), min_size=min_size).astype(np.int16)
    
        if type(input_data) is str: return self.nib_save()
        else: return self.output_data

    def substract(self, atlas_data, input_data, force=True, suffix="sub"):
        """ Match the histogram of two images.

        Parameters
        ----------
        atlas_data : str/array
            The image on which the other image will be substracted. Can be a path or an array

        input_data : str/array
            The fixed image that will be substracted. Can be a path or an array

        suffix : str
            Nifti file suffixe for the new generated image
        """
        self.init("Substract")
        if type(input_data) is str:
            self.generate_output_path(input_data, suffix, force) 
            if self.exit: return self.output_path

        self.load_data(input_data, atlas_data=atlas_data)
        print(f"INFO - Starting to substract between {self.precise_path}")

        self.output_data = self.nii_atlas_data - self.nii_data

        if type(input_data) is str: return self.nib_save()
        else: return self.output_data