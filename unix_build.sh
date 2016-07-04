#!/bin/bash
#################################
echo 'Removing old build'
rm -rf build
rm -rf CMakeFiles
rm -rf CMakeCache.txt
rm -rf cmake_install.cmake
#################################
echo 'Creating new build'
mkdir build
cd build
#################################

export PYEXE=$(which python2.7)
if [[ "$VIRTUAL_ENV" == ""  ]]; then
    export LOCAL_PREFIX=/usr/local
    export _SUDO="sudo"
else
    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
    export _SUDO=""
fi

echo 'Configuring with cmake'
if [[ '$OSTYPE' == 'darwin'* ]]; then
    export CONFIG="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++ -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV"
else
    export CONFIG="-DCMAKE_BUILD_TYPE='Release' -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV"
fi
cmake $CONFIG -G 'Unix Makefiles' ..
#################################
echo 'Building with make'
export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -j$NCPUS -w
#################################
if [[ '$OSTYPE' == 'darwin'* ]]; then
    echo 'Fixing OSX libiomp'
    install_name_tool -change libiomp5.dylib ~/code/libomp_oss/exports/mac_32e/lib.thin/libiomp5.dylib lib*
fi
#################################
echo 'Moving the shared library'
cp -v lib* ../pyrf
cd ..
