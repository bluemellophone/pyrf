echo "removing old build"
rm -rf build
rm -rf CMakeFiles
rm -rf CMakeCache.txt
rm -rf cmake_install.cmake

mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++ -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
else
    export CMAKE_BUILD_TYPE="Release"
    #export CMAKE_BUILD_TYPE="Debug"
    cmake -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        .. || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
fi

export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -w -j$NCPUS || { echo "FAILED MAKE" ; exit 1; }

cd ..
#python -c "import utool; print(utool.truepath('build/libpyrf'))"
# cp -v build/libpyrf* pyrf/
