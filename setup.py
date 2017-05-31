from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy as np

extensions = [
    Extension("rough_front_expansion.cython",
              sources=["rough_front_expansion/cython.pyx"],
              language="c", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()]),
Extension("rough_front_expansion.cython_D2Q9",
              sources=["rough_front_expansion/cython_D2Q9.pyx"],
              language="c", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()])
]

setup(
    name='rough_front_expansion',
    version='0.1',
    packages=['rough_front_expansion'],
    url='',
    license='',
    author='btweinstein',
    author_email='btweinstein@gmail.com',
    description='',
    include_dirs = [cython_gsl.get_include(), np.get_include()],
    ext_modules = cythonize(extensions, annotate=True)
)
