from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

from setuptools import setup

setup(name='Dactim_MRI',
      python_requires='>=3.7',
      install_requires=[
        "torch>=1.13.1",
        "torchio>=0.18.86",
        "SimpleITK",
        "numpy",
        "nibabel",
        "matplotlib",
        "pydicom>=2.3.1",
        "dicom2nifti",
        "itk-elastix==0.17.1"
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Windows :: Windows 10'
      ],
      command_options={
        'build_sphinx': {
            'project': ('setup.py', 'Dactim_MRI'),
            'source_dir': ('setup.py', 'docs')
        }
      }
)
