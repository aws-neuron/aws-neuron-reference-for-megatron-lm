#! /bin/bash -v

set -ex

#copy megatron to build dir and then make in data dir

BLD_ROOT=$(brazil-path package-build-root)
PKG_ROOT=$(brazil-path package-src-root)
RUNTIMEFARM_PATH=$(brazil-path ht2-build.runtimefarm)

if [ "$1" == "clean" ]; then
    rm -rf ${PKG_ROOT}/*egg-info ${BUILD_ROOT}/pip ${BUILD_ROOT}/pip/megatron ${PKG_ROOT}/build/pip 
    env TOOL_VERBOSE=true python-3p-tool "$@"
    exit
fi

cp -r megatron-lm/megatron $BLD_ROOT
cd $BLD_ROOT/megatron/data 
BRZ_PATH=$(brazil-path run.runtimefarm)
export PATH=$PATH:$BRZ_PATH/bin

#make helpers.cpp python 3.6 wheel
PYBIND_INCLUDE=$(brazil-path build.libfarm)/include
#PYTHON_INCLUDE=$(brazil-path build.libfarm)/python3.6/include/python3.6m/
#export CXXFLAGS="-I$PYBIND_INCLUDE -I$PYTHON_INCLUDE"
#make
#mv helpers helpers.cpython-36m-x86_64-linux-gnu.so

#generate wheel for python 3.6
SITE_PACKAGES_SUFFIX=lib/python3.7/site-packages
#cd $PKG_ROOT
export PYTHONPATH=$RUNTIMEFARM_PATH/$SITE_PACKAGES_SUFFIX
#python megatron-lm/setup.py bdist_wheel --python-tag=py3.6 --dist-dir $BLD_ROOT/pip/public/megatron

#rm -rf $BLD_ROOT/megatron/
#cp -r megatron-lm/megatron $BLD_ROOT
#only building wheel for python3.7 currently
cd $BLD_ROOT/megatron/data
export CXXFLAGS="-I$PYBIND_INCLUDE -I$RUNTIMEFARM_PATH/python3.7/include/python3.7m"
make
mv helpers helpers.cpython-37m-x86_64-linux-gnu.so

#generate wheel for python3.7
cd $PKG_ROOT
python3.7 megatron-lm/setup.py bdist_wheel --python-tag=py3.7 --dist-dir $BLD_ROOT/pip/private/megatron



