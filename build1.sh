cmake -DCMAKE_INSTALL_PREFIX=/home/igor/SOFTWARE/INSTALL/llvm/9.0.0/gfk -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_COMPILER=`which clang++` -DKokkos_ENABLE_CUDA:BOOL=ON -DCMAKE_PREFIX_PATH="$ADIOS2_DIR;$KOKKOS_DIR" ..