/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        namespace platforms {
            PLATFORM_IMPL(conv2d, ENGINE_CUDA) {
                auto handle = reinterpret_cast<cudnnHandle_t *>(block.launchContext()->getCuDnnHandle());
                auto res = cudnnSetStream(*handle, *block.launchContext()->getCudaStream());
                if (res != 0)
                    throw nd4j::cuda_exception::build("Can't set stream for cuDNN", res);

                auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
                auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] always
                auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

                auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

                NDArray::prepareSpecialUse({output}, {input, weights, bias});

                int sH = INT_ARG(2);                                                        // strides height
                int sW = INT_ARG(3);                                                        // strides width
                int pH = INT_ARG(4);                                                        // paddings height
                int pW = INT_ARG(5);                                                        // paddings width
                int dH = INT_ARG(6);                                                        // dilations height
                int dW = INT_ARG(7);                                                        // dilations width
                int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
                bool isNCHW    = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // INT_ARG(9): 0-NCHW,  1-NHWC

                int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0)); // filter(kernel) height
                int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1)); // filter(kernel) width

                int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
                int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
                ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, isSameMode);

                auto dtype = cudnnDataType(input->dataType());


                cudnnTensorDescriptor_t src;
                cudnnCreateTensorDescriptor(&src);
                res = cudnnSetTensor4dDescriptorEx(src, dtype, input->sizeAt(0), input->sizeAt(1), input->sizeAt(2), input->sizeAt(3), input->strideAt(0), input->strideAt(1), input->strideAt(2), input->strideAt(3));
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnSetTensor4dDescriptorEx src failed", res);

                // TODO: we definitely want NHWC here as well
                cudnnFilterDescriptor_t wght;
                cudnnCreateFilterDescriptor(&wght);
                res = cudnnSetFilter4dDescriptor(wght, dtype, CUDNN_TENSOR_NCHW, oC, iC, kH, kW);
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnSetFilter4dDescriptor failed", res);

                cudnnConvolutionDescriptor_t cdc;
                cudnnCreateConvolutionDescriptor(&cdc);
                res = cudnnSetConvolution2dDescriptor(cdc, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, dtype);
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnSetConvolution2dDescriptor failed", res);

                cudnnTensorDescriptor_t dst;
                cudnnCreateTensorDescriptor(&dst);
                res = cudnnSetTensor4dDescriptorEx(dst, dtype, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3));
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnSetTensor4dDescriptorEx dst failed", res);

                // TODO: workspace algorithms are supposed to be faster, so we should use it here if we have enough memory
                cudnnConvolutionFwdAlgo_t algo;
                res = cudnnGetConvolutionForwardAlgorithm(*handle, src, wght, cdc, dst, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnGetConvolutionForwardAlgorithm failed", res);

                // TODO: should be float if dtype is half/float, and double otherwise
                float alpha = 1.0f;
                float beta = 0.0f;
                res = cudnnConvolutionForward(*handle, &alpha, src, input->specialBuffer(), wght, weights->specialBuffer(), cdc, algo, nullptr, 0, &beta, dst, output->specialBuffer());
                if (res != 0)
                    throw nd4j::cuda_exception::build("cudnnConvolutionForward failed", res);


                if (bias != nullptr) {
                    cudnnTensorDescriptor_t bs;
                    cudnnCreateTensorDescriptor(&bs);
                    if (isNCHW) {
                        res = cudnnSetTensor4dDescriptor(bs, CUDNN_TENSOR_NCHW, dtype, 1, bias->lengthOf(), 1, 1);
                        if (res != 0)
                            throw nd4j::cuda_exception::build("cudnnSetTensor4dDescriptorEx bias NHWC failed", res);
                    } else {
                        res = cudnnSetTensor4dDescriptor(bs, CUDNN_TENSOR_NHWC, dtype, 1, 1, 1, bias->lengthOf());
                        if (res != 0)
                            throw nd4j::cuda_exception::build("cudnnSetTensor4dDescriptorEx bias NHWC failed", res);
                    }

                    res = cudnnAddTensor(*handle, &alpha, bs, bias->specialBuffer(), &alpha, dst, output->specialBuffer());
                    if (res != 0)
                        throw nd4j::cuda_exception::build("cudnnAddTensor failed", res);
                }


                NDArray::registerSpecialUse({output}, {input, weights, bias});

                return Status::OK();
            }

            PLATFORM_CHECK(conv2d, ENGINE_CUDA) {
                return true;
            }
        }
    }
}
