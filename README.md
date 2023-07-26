# PyDactim

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://img.shields.io/pypi/v/pydactim?label=PyPI%20version&logo=python&logoColor=white)](https://pypi.org/project/pydactim/)
[![Doc](https://img.shields.io/readthedocs/pydactim?label=Docs&logo=Read%20the%20Docs)](https://dactim-mri.readthedocs.io/en/latest/)
<!-- [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PierreFayolle/Dactim_MRI/blob/master/tests/transformation.ipynb) -->


## Description

> **Warning**
: Currently in development

A MRI preprocessing tool which provides a dozen of functions easy to use made from some of the most popular Python libraries 🚀

It includes :
  - Transformations : skull stripping, resampling, registration, histogram matching, bias field correction, normalization and so on
  - Preprocessing : BIDS compatible, preprocess all of your patient data with only two lines of code
  - Conversion : Dicom to Nifti, Nifti to Dicom
  - Sorting : an elegant way to organize your Dicom files 
  - Anonymization : Dicom files can be anonymized according to your own rules
  - Visualization : 2D and 3D light viewer
  - Spectroscopy : generation of mask of the MRS voxel/slab

## Installation

Requires at least Python 3.7 or above. This library uses a lot of common libraries in image processing as such as [Pytorch](https://github.com/pytorch/pytorch), [Torchio](https://github.com/fepegar/torchio), [Numpy](https://github.com/numpy/numpy), [Nibabel](https://github.com/nipy/nibabel), [SimpleITK](https://github.com/SimpleITK/SimpleITK), [Matplotlib](https://github.com/matplotlib/matplotlib), [Pydicom](https://github.com/pydicom/pydicom), [Dicom2nifti](https://github.com/icometrix/dicom2nifti) and so more !

Can be quickly installed with pip :

```
pip install pydactim
```
