# from sphinx.setup_command import BuildDoc
# cmdclass = {'build_sphinx': BuildDoc}

from setuptools import setup, find_packages

setup(name='pydactim',
      python_requires='>=3.7',
      version='0.0.26',
      packages=find_packages(),
      install_requires=[
        "torchio>=0.18.86",
        "SimpleITK",
        "numpy",
        "nibabel",
        "matplotlib",
        "pydicom>=2.3.1",
        "dicom2nifti",
        "itk-elastix==0.17.1",
        "dipy>=1.7.0",
        "scikit-image",
        "numba",
        "pyside6",
        "pyqtgraph",
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows :: Windows 10'
      ],
      command_options={
        'build_sphinx': {
            'project': ('setup.py', 'pydactim'),
            'source_dir': ('setup.py', 'docs')
        }
      }
)
