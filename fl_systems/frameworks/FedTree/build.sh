rm -rf ./build
mkdir build
pushd build

cmake .. -DDISTRIBUTED=ON > /dev/null
make -j4 > /dev/null

popd