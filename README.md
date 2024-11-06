# Installation
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.0+cu124.zip
rm libtorch-cxx11-abi-shared-with-deps-2.5.0+cu124.zip
sudo cp -r libtorch /usr/lib/
rm -rf libtorch
unset LIBTORCH
export LIBTORCH=/usr/lib/libtorch
export LD_LIBRARY_PATH=/urs/lib/libtorch/lib:$LD_LIBRARY_PATH
source ~/.bashrc
cargo t --workspace
