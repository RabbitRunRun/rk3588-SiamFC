#!/bin/bash

set -e


ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

#GCC_COMPILER=${ROOT_PWD}/../../3399pro/deployment/toolchain/toolchain/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
#GCC_COMPILER=/wqy/test/onnx_rkn/tools/vizvision_rk3588_linux_sdk_v0.1.19/vizvision_rk3588_linux_sdk_v0.1.19/gcc-arm-12.3-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu

GCC_COMPILER=/wqy/test/onnx_rkn/tools/vizvision_rk3588_linux_sdk_v0.1.19/vizvision_rk3588_linux_sdk_v0.1.19/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
# build rockx
BUILD_DIR=${ROOT_PWD}/build

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++
make -j4
make install
cd -

cp run_yolov5_demo.sh rknn_yolov5_demo/
cp -r model rknn_yolov5_demo/

