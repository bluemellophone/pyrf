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
echo 'Configuring with cmake'
if [[ '$OSTYPE' == 'darwin'* ]]; then
    export CONFIG="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++"
else
    export CONFIG="-DCMAKE_BUILD_TYPE='Release'"
fi

# Virtual Env
virtualenv_settings()
{
    $PYTHON_VENV/venv
    ls $PYTHON_VENV
    ls $PYTHON_VENV/local/share/OpenCV
    export OpenCV_DIR=$PYTHON_VENV/local/share/OpenCV
}
cmake $CONFIG -G 'Unix Makefiles' ..
#################################
echo 'Building with make'
export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -j$NCPUS -w
#################################
echo 'Moving the shared library'
cp lib* ../pyrf
cd ..
