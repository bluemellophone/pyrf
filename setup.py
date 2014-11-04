#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool import util_cplat
from setuptools import setup
from utool import util_setup

#delete_on_clean = ['libpyrf.dll', 'libpyrf.dll.a', 'libpyrf.so', 'libpyrf.dylib', 'build/']
delete_on_clean = [
    'libpyrf.dll',
    'libpyrf.dll.a',
    'libpyrf.so',
    'libpyrf.dylib',
    'pyrf.egg-info',
    'build/',
    '*.pyo',
    '*.pyc'
]


def build_command():
    """ Build command run by utool.util_setup """
    if util_cplat.WIN32:
        util_cplat.shell('mingw_build.bat')
    else:
        util_cplat.shell('./unix_build.sh')


INSTALL_REQUIRES = [
    #'detecttools >= 1.0.0.dev1',
]

if __name__ == '__main__':
    kwargs = util_setup.setuptools_setup(
        name='pyrf',
        packages=['pyrf', 'build'],
        #packages=util_setup.find_packages(),
        version=util_setup.parse_package_for_version('pyrf'),
        licence=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        description=('Detects objects in images using random forests'),
        url='https://github.com/bluemellophone/pyrf',
        author='Jason Parham',
        author_email='bluemellophone@gmail.com',
        install_requires=INSTALL_REQUIRES,
        package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        build_command=build_command,
        setup_fpath=__file__,
    )
    setup(**kwargs)
