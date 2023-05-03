# Dactim_MRI

[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)
[![PyPI version](https://badge.fury.io/py/dactim_mri.svg)](https://badge.fury.io/py/dactim_mri)

## Description

A MRI preprocessing tool which provides a dozen of functions easy to use made from some of the most popular Python libraries 🚀

It includes :
  - Transformations : skull stripping, resampling, registration, histogram matching, bias field correction, normalization and so on
  - Conversion : dicom to nifti, nifti to dicom
  - Sorting : an elegant way to organize your dicoms 
  - Anonymization : dicom can be anonymized according to hospital compliance
  - Visualization : 2D and 3D light viewer
  - Spectroscopy : generation of mask of the MRS voxel/slab

## Installation

Requires at least Python 3.6 or above. Also requires [Pytorch](https://github.com/pytorch/pytorch), [Torchio](https://github.com/fepegar/torchio), [Numpy](https://github.com/numpy/numpy), [Nibabel](https://github.com/nipy/nibabel), [SimpleITK](https://github.com/SimpleITK/SimpleITK), [Matplotlib](https://github.com/matplotlib/matplotlib), [Pydicom](https://github.com/pydicom/pydicom), [Dicom2nifti](https://github.com/icometrix/dicom2nifti).

Can be quickly installed with pip :

```
pip install dactim-mri
```

