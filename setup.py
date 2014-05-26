#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool import util_cplat


def build_command():
    """ Build command run by utool.util_setup """
    if util_cplat.WIN32:
        util_cplat.shell('mingw_build.bat')
    else:
        util_cplat.shell('./unix_build.sh')


INSTALL_REQUIRES = [
    'detecttools >= 1.0.0.dev1'
]

if __name__ == '__main__':
    setuptools_setup(
        name='pyrf',
        build_command=build_command,
        description=('Detects objects in images using random forests'),
        url='https://github.com/bluemellophone/pyrf',
        author='Jason Parham',
        author_email='bluemellophone@gmail.com',
        packages=['pyrf'],
        install_requires=INSTALL_REQUIRES,
        package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        setup_fpath=__file__,
    )
