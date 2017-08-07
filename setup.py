import numpy
import os
import platform
import subprocess
import sys
from distutils.core import setup, Extension

include_dirs = []
library_dirs = []

if (platform.system() == 'Darwin'):
    try:
        subprocess.check_call(['which', 'brew'], stdout = subprocess.PIPE)
        include_dirs = ['/usr/local/include']
        library_dirs = ['/usr/local/lib']
    finally: pass

modules = []
modules.append(['multi_array', 'python_multi_array.cpp'])
modules.append(['demo_module', 'demo_module.cpp'])

extensions = []
for module in modules:
    mod_name = module[0]
    mod_source = module[1]
    ex_module = Extension(
        mod_name,
        sources = [mod_source],
        include_dirs = [numpy.get_include()] + include_dirs,
        libraries = ['boost_python', 'boost_numpy'],
        library_dirs = library_dirs,
        extra_compile_args=['-std=c++14'],
    )
    extensions.append(ex_module)

setup(name = "multi_array", version = "1.0", ext_modules = extensions)
