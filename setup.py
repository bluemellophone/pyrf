#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool.util_cplat import get_dynamic_lib_globstrs
import sys
import subprocess
import pyrf


def build_command():
    """ Build command run by utool.util_setup """
    if sys.platform.startswith('win32'):
        subprocess.call(['mingw_hesaff_build.bat'])
    else:
        subprocess.call(['unix_hesaff_build.sh'])


if __name__ == '__main__':
    setuptools_setup(
        setup_fpath=__file__,
        module=pyrf,
        build_command=build_command,
        description=('Routines for computation of hessian affine keypoints in images.'),
        url='https://github.com/bluemellophone/pyrf',
        author='Jason Parham',
        author_email='bluemellophone@gmail.com',
        packages=['build', 'pyrf'],
        py_modules=['pyrf'],
        package_data={'build': get_dynamic_lib_globstrs()},
    )
