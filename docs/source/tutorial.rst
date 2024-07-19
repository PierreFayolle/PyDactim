
PyDactim Tutorial
======================================

Here is some example of what you can do with PyDactim 🔥

Let's imagine a small pipeline, including:
- Auto cropping the MRI image
- Normalizing voxel intensity value by 1
- Extracting the brain
- Applying a SUSAN filter

So first at all, let's import PyDactim and define so variable of some kind of data:

.. code-block:: python

   import pydactim as pyd

   data_path = "path/to/your/data.nii.gz"

Now we can call the auto crop function:

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path)

As you can see, all PyDactim function take a path as input and output the generated image as a path. The new image is saved as the same location than the input and will have a suffix added that you can change according to the parameter 'suffix'
For instance, let's change the suffix for the crop:

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path, suffix='zoom')

That's a bad suffix ! But now, in the same folder than the input image you will have the generated image with this suffix.
If you want to overwrite an already existing output image, you can use the parameter 'force'.

.. code-block:: python

    cropped_data_path, cropped_idx = pyd.crop(data_path, force=True)