.. Dactim MRI documentation master file, created by
   sphinx-quickstart on Thu May  4 14:14:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dactim MRI's Documentation
======================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. raw:: HTML

   <div style="display: flex; padding-bottom: 20px">
      <img src="_static/I3M-logo.png" width="105px" style="margin-right: 10px">
      <img src="_static/LMA-logo.png" width="173px" style="margin-right: 10px">
      <img src="_static/dactim-logo.png" width="110px" style="margin-right: 10px">
      <img src="_static/Univ-logo.png" width="163px" style="margin-right: 10px">
   </div>

|Windows| |PyPI pyversions| |Open In Collab| |PyPI version fury.io|

.. |Windows| image:: https://svgshare.com/i/ZhY.svg
   :target: https://svgshare.com/i/ZhY.svg

.. |PyPI pyversions| image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://pypi.python.org/pypi/dactim-mri/

.. |Open In Collab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://pypi.python.org/pypi/dactim-mri/

.. |PyPI version fury.io| image:: https://badge.fury.io/py/dactim_mri.svg
   :target: https://pypi.python.org/pypi/dactim-mri/
   
Description
-----------

A MRI preprocessing tool which provides a dozen of functions easy to use made from some of the most popular Python libraries 🚀

It includes :
  - Transformations : skull stripping, resampling, registration, histogram matching, bias field correction, normalization and so on
  - Conversion : dicom to nifti, nifti to dicom
  - Sorting : an elegant way to organize your dicoms 
  - Anonymization : dicom can be anonymized according to hospital compliance
  - Visualization : 2D and 3D light viewer
  - Spectroscopy : generation of mask of the MRS voxel/slab

Installation
------------

Requires at least Python 3.7 or above. Also requires `Pytorch <https://github.com/pytorch/pytorch/>`__, `Torchio <https://github.com/fepegar/torchio/>`__, `Numpy <https://github.com/numpy/numpy/>`__, `Nibabel <https://github.com/nipy/nibabel/>`__, `SimpleITK <https://github.com/SimpleITK/SimpleITK/>`__, `SimpleITK <https://github.com/matplotlib/matplotlib/>`__, `Pydicom <https://github.com/pydicom/pydicom/>`__, `Dicom2nifti <https://github.com/icometrix/dicom2nifti/>`__.


Can be quickly installed with pip :

.. code-block:: python

   pip install dactim-mri

.. NOTE::

   Ceci est une note.

.. WARNING::

   Ceci est un avertissement !

.. IMPORTANT::

   Ceci est important !

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`