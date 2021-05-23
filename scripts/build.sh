cd .. && mkdir build
cd build/ && rm -rf *
cd ../build/ && cmake \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release ..
cd ../build && make install
cp ../build/metnum.cpython-36m-x86_64-linux-gnu.so ../scripts
