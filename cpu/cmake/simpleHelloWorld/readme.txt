//configure and generation 
mkdir -p build
cd build
cmake ..
-------------------
cmake -H. -Bbuild

===================
//build
cmake --build .
cmake --build . --target help
