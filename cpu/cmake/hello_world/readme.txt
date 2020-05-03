mkdir -p build
cd build
cmake ..
=============
cmake -H. -Bbuild

--------

cmake --build .
cmake --build . --target help
