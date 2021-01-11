from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extenstions = [
    Extension("const_decoder", ["src_joint/const_decoder.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("hpsg_decoder", ["src_joint/hpsg_decoder.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    setup_requires=['setuptools_scm'],
    name='lal-parser-server',
    entry_points={
        'console_scripts': [
            'lal-parser-server=src_joint.server.serve:serve'
        ],
    },
    ext_modules = cythonize(extenstions)
)
