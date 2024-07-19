.. Dactim MRI documentation master file, created by
   sphinx-quickstart on Thu May  4 14:14:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyDactim Documentation
======================================

.. raw:: HTML

   <div style="padding-bottom: 20px">
      <img src="_static/pydactim-logo.png" width="120px" style="margin-right: 5px">
      <img src="_static/I3M-logo.png" width="105px" style="margin-right: 5px">
      <img src="_static/LMA-logo.png" width="155px" style="margin-right: 5px">
      <img src="_static/dactim-logo.png" width="110px" style="margin-right: 5px">
      <img src="_static/Univ-logo.png" width="163px" style="margin-right: 5px">
   </div>

|PyPI pyversions| |PyPI version fury.io|

.. |PyPI pyversions| image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://pypi.python.org/pypi/pydactim/

.. |PyPI version fury.io| image:: https://img.shields.io/pypi/v/pydactim
   :target: https://pypi.python.org/pypi/pydactim/

.. .. |Open In Collab| image:: https://colab.research.google.com/assets/colab-badge.svg
..    :target: https://colab.research.google.com/github/PierreFayolle/Dactim_MRI/blob/master/tests/transformation.ipynb

.. _index:

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   tutorial

Description
-----------

A MRI preprocessing tool which provides a dozen of functions easy to use made from some of the most popular Python libraries 🚀

It includes:
   - Transformations: skull stripping, resampling, registration, histogram matching, bias field correction, normalization and so on
   - Conversion: Dicom to Nifti, Nifti to Dicom
   - Sorting: an elegant way to organize your Dicom files
   - Anonymization: Dicom files can be anonymized according to your own rules
   - Visualization: 2D and 3D light viewer
   - Spectroscopy: generation of mask of the MRS voxel/slab


Installation
------------

Requires at least Python 3.7 or above. Also requires `Pytorch <https://github.com/pytorch/pytorch/>`__, `Torchio <https://github.com/fepegar/torchio/>`__, `Numpy <https://github.com/numpy/numpy/>`__, `Nibabel <https://github.com/nipy/nibabel/>`__, `SimpleITK <https://github.com/SimpleITK/SimpleITK/>`__, `Pydicom <https://github.com/pydicom/pydicom/>`__, `Dicom2nifti <https://github.com/icometrix/dicom2nifti/>`__.
Make sure to manually install PyTorch on your own !

Can be quickly installed with pip:

.. code-block:: python

   pip install pydactim

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
