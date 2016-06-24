#!/bin/bash

#################################
#echo 'Removing old build'
#rm -rf build
#rm -rf CMakeFiles
#rm -rf CMakeCache.txt
#rm -rf cmake_install.cmake
python2.7 -c "import utool as ut; print('keeping build dir' if ut.get_argflag(('--fast', '--no-rmbuild')) else ut.delete('build'))" $@


#################################
echo 'Creating new build'
mkdir -p build
cd build
#################################
echo 'Configuring with cmake'
if [[ '$OSTYPE' == 'darwin'* ]]; then
    export CONFIG="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++"
else
    export CONFIG="-DCMAKE_BUILD_TYPE='Release'"
fi


export PYEXE=$(which python2.7)
export IN_VENV=$($PYEXE -c "import sys; print(hasattr(sys, 'real_prefix'))")
echo "IN_VENV = $IN_VENV"

if [[ "$IN_VENV" -eq "True" ]]; then
    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
    #export _SUDO=""
else
#if [[ "$VIRTUAL_ENV" == ""  ]]; then
    export LOCAL_PREFIX=/usr/local
    #export _SUDO="sudo"
fi


# Use Virtual Env OpenCV if available

if [[ "$IN_VENV" -eq "True" ]]; then
    export OpenCV_Dir="$LOCAL_PREFIX/share/OpenCV"
    if [ -d "$VENV_OpenCV_Dir" ]; then
        export CONFIG=" $CONFIG -DOpenCV_DIR='$OpenCV_Dir'"
    fi
fi

# install_name_tool -change libiomp5.dylib ~/code/libomp_oss/exports/mac_32e/lib.thin/libiomp5.dylib lib*

echo "CONFIG = $CONFIG"

#################################
echo 'Building with make'
export NCPUS=$(grep -c ^processor /proc/cpuinfo)
if [[ "$OSTYPE" == "msys"* ]]; then
    # Handle mingw on windows
    cmake $CONFIG -G 'MSYS Makefiles' ..
    make
else
    cmake $CONFIG -G 'Unix Makefiles' ..
    make -j$NCPUS -w
fi

#################################
export MAKE_EXITCODE=$?
echo "MAKE_EXITCODE=$MAKE_EXITCODE"

if [[ $MAKE_EXITCODE == 0 ]]; then
    #make VERBOSE=1
    echo 'Moving the shared library'
    #cp -v libhesaff* ../pyhesaff
    #cp lib* ../pyrf
    cp -v lib* ../pyrf
else
    export FAILCMD='{ echo "FAILED PYRF BUILD" ; exit 1; }'
    $FAILCMD
fi

cd ..
