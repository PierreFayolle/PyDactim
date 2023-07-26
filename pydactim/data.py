import numpy as np
import nibabel as nib
import torch
import os

class MRIData():
    def __init__(self, path):
        self.path = path
        self.setup()

    def setup(self):
        if os.path.exists(self.path):
            temp = nib.load(self.path)
            self.affine = temp.affine
            self.header = temp.header
            self.pixdim = self.header["pixdim"][1:4]
            self.array = temp.get_fdata()
        else:
            raise ValueError("Could not find a file that matches the following path: %s" % self.path)

    def get_tensor(self):
        return torch.Tensor(self.array).unsqueeze(dim=0)
    
    def save(self, filename=None, suffix=None, affine=None):
        if len(self.array.shape) == 4:
            self.array.squeeze()
        if affine is None:
            affine = self.affine
        if filename is None and suffix is None:
            raise ValueError("At least one of the following parameters have to be specified: filename, suffix")
        if filename is None and suffix is not None:
            if suffix[0] == "_": suffix = suffix[1:]
            filename = self.path.replace(".nii", "_"+suffix+".nii")
        nib.save(
            nib.Nifti1Image(
                self.array,
                affine
            ),
            filename
        )
        print("INFO - Successfully save at\n\t %s" % filename)

    def __str__(self):
        pixdim = self.header["pixdim"][1:4]
        return f"Dactim MRI - Data Object:\n\tShape: {self.array.shape}\n\tVoxel dimension: {pixdim}\n\tAffine:\n{self.affine}"

if __name__ == '__main__':
    data = MRIData(r"D:\Studies\DYN24\data\sub-001\ses-01\anat\sub-001_ses-01_T1w.nii.gz")
    print(data)