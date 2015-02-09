SET ORIGINAL=%CD%
call :build_pyrf
goto :exit

:build_pyrf
:: #################################
:: Removing old build
rmdir build /s
del CMakeFiles
del CMakeCache.txt
del cmake_install.cmake
:: #################################
:: Creating new build
mkdir build
cd build
:: #################################
:: Configuring with cmake
set INSTALL32=C:\Program Files (x86)
cmake -G "MSYS Makefiles" -DOpenCV_DIR="%INSTALL32%\OpenCV" ..
:: -DCMAKE_C_FLAGS="-march=i486" -DCMAKE_CXX_FLAGS="-march=i486"
:: #################################
:: Building with make
make
:: mingw32-make -j7 "MAKE=mingw32-make -j3" -f CMakeFiles\Makefile2 all
:: #################################
:: Moving the shared library
move lib* ..\pyrf
exit /b

:exit
cd %ORIGINAL%
exit /b
