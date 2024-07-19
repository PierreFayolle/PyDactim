import numpy as np
import nibabel as nib
import os
import pydicom
from copy import deepcopy
import time
import dicom2nifti

def convert_dicom_to_nifti(dicom_dir, output_dir):
    """ Convert Dicom folder to NIfTI format (@author: abrys)

    Parameters
    ----------
    dicom_dir : str
        Dicom directory to convert

    output_dir : str
        Directory of the NIfTI file

    """
    dicom2nifti.convert_directory(dicom_dir, output_dir)
    
class Dicomize():
    """Use a NIfTI file and convert it in DICOM using a folder of DICOM as template.

    The DICOM folder must contains the original DICOM used to create the NIfTI
    """
    def __init__(self, dicom_dir, final_dir, nifti_path, series_description, comment="", rgb=False):
        """ 
        Parameters
        ----------
        dicom_dir : str
            The path of the folder of DICOM used to create the NIfTI file to be converted back to DICOM

        final_dir : str
            The path of the directory in which all the DICOM file generated will be saved

        nifti_path : str
            The path of the NIfTI file to be convertede back to DICOM

        series_description : str
            The series description of the new generated DICOM

        comment : str
            The comment to write in the DICOM tag Image Comment (0020,4000)

        rgb : boolean
            The color palette of the NIfTI file

        """
        self.dicom_dir = dicom_dir
        self.final_dir = final_dir
        self.nifti = nib.load(nifti_path)
        self.array = self.nifti.get_fdata().astype('uint8')[::-1,::-1,::-1]
        self.series_description = series_description
        self.comment = comment
        self.rgb = rgb

    def run(self):
        if not os.path.exists(self.final_dir):
            os.makedirs(self.final_dir)

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        self.array = np.transpose(self.array, (0,2,1,3))

        slices = [pydicom.dcmread(os.path.join(self.dicom_dir, dicom_path)) for dicom_path in os.listdir(self.dicom_dir)]
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        for i, ds in enumerate(slices):
            
            self.final_ds = deepcopy(ds)

            self.final_ds.PixelData = self.array[i,:,:].tobytes()

            self.final_ds.ContentDate = modification_date
            self.final_ds.ContentTime = modification_time

            self.final_ds.file_meta.MediaStorageSOPInstanceUID += "." + str(i)
            # SOP Instance UID
            self.final_ds[0x8,0x18].value = str(self.final_ds.file_meta.MediaStorageSOPInstanceUID)

            self.final_ds[0x8,0x70].value = "DACTIM-MRI"
            if self.rgb is True:
                self.final_ds[0x28,0x2].value = 3
                self.final_ds[0x28,0x4].value = "RGB"
                # self.final_ds[0x28,0x6].value = 0
                self.final_ds[0x28,0x100].value = 8
                self.final_ds[0x28,0x101].value = 8
                self.final_ds[0x28,0x102].value = 7
                self.final_ds[0x28,0x103].value = 0
                self.final_ds[0x28,0x106].value = 0
                self.final_ds[0x28,0x107].value = 1000
            # Series Description
            self.final_ds[0x8,0x103E].value = self.series_description
            # Series Number
            self.final_ds[0x20,0x11].value = 400
            # Series Instance UID
            self.final_ds[0x20,0xe].value = self.final_ds[0x20,0xe].value + "." + str(400)
            # Instance number
            self.final_ds[0x20,0x13].value = str(i + 1)

            self.final_ds.ImageComments = self.comment

            dcm_path = os.path.join(self.final_dir, self.final_ds[0x20,0xe].value + str(i) + ".dcm")
            self.final_ds.save_as(dcm_path)
