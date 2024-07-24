# from sphinx.setup_command import BuildDoc
# cmdclass = {'build_sphinx': BuildDoc}

from setuptools import setup, find_packages

project_urls = {
  'Documentation' : "https://pydactim.readthedocs.io/en/latest/",
  'Source' : "https://github.com/PierreFayolle/PyDactim",
  'Download' : "https://pypi.org/project/PyDactim/#files",
  'Tracker' : "https://github.com/PierreFayolle/PyDactim/issues"
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='pydactim',
      python_requires='>=3.7',
      version='1.0.11',
      author="Pierre Fayolle",
      author_email="fayollepierre86@gmail.com",
      description="A library to post-process MRI data",
      long_description=long_description,
      long_description_content_type="text/markdown", 
      url="https://github.com/PierreFayolle/PyDactim",
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
      },
      project_urls = project_urls
)
