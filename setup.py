import numpy
import os
import sys
from distutils.core import setup, Extension

modules = []
modules.append(['multi_array', 'source/python_multi_array.cpp'])

extensions = []
for module in modules:
    mod_name = module[0]
    mod_source = module[1]
    ex_module = Extension(
        mod_name,
        sources = [mod_source],
        include_dirs = ['/usr/local/include', numpy.get_include()],
        libraries = ['boost_python', 'boost_numpy'],
        library_dirs = ['/usr/local/lib'],
        extra_compile_args=['-std=c++14'],
    )
    extensions.append(ex_module)

setup(name = "multi_array", version = "0.9", ext_modules = extensions)
