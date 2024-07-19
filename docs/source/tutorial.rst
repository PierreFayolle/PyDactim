
PyDactim Tutorial
======================================

.. _tutorial:

Here is some example of what you can do with PyDactim 🔥

🔧 Pipeline example 
+++++++++++++++++++++

Let's imagine a small pipeline, including:
   - Auto cropping the MRI image
   - Normalizing voxel intensity value by 1
   - Extracting the brain
   - Applying a SUSAN filter

So first at all, let's import PyDactim and define a variable of some kind of data:

.. code-block:: python

   import pydactim as pyd

   data_path = "path/to/your/data.nii.gz"

Now we can call the auto crop function:

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path)

As you can see, all PyDactim function take a path as input and output the generated image as a path. The new image is saved as the same location than the input and will have a suffix added that you can change according to the parameter 'suffix'.
For instance, let's change the suffix for the crop:

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path, suffix='zoom')

That's a bad suffix ! But now, in the same folder than the input image you will have the generated image with this suffix.

If you want to overwrite an already existing output image, you can use the parameter 'force'.

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path, force=True)

Let's see now the rest of the pipeline:

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path) # This will crop the data to remove most of the noise
    normalized_data_path = pyd.normalize(cropped_data_path) # Normalize between 0 and 1 the image
    brain_data_path, mask_data_path = pyd.skull_stripping(normalized_data_path, model_path="path_to_fold", mask=True) # See https://github.com/MIC-DKFZ/HD-BET for the fold of the HD-BET
    susan_data_path = pyd.susan(brain_data_path, offset=3) # SUSAN filtering with a kernel of size 3

💻 Viewer App
+++++++++++++++++++++

PyDactim also have a GUI to visualize and launch the function without using coding. It can be easily launch with the following line:

.. code-block:: python

    app = ViewerApp()

It can handle 3D MRI images as well as 4D Perfusion MRI data with a ROI to observe the signal over time.
This GUI is still in work so you might encounter some unexpected behaviour.

📁 Managing DICOM files
+++++++++++++++++++++++++

You can also sort DICOM by tags such as  Study description, Patient ID, Study Date, Series description:

.. code-block:: python

    dicom_dir = "directory/to/your/dicom/"
    sorted_dir = "directory/for/the/sorted/dicom/"

    sort_dicom(dicom_dir, sorted_dir)

Then, you can convert a directory of DICOM in NIfTI

.. code-block:: python

    dicom_dir = "directory/to/your/dicom/"
    nifti_dir = "directory/for/the/sorted/dicom/"

    convert_dicom_to_nifti(dicom_dir, nifti_dir)