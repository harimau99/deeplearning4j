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




                return Status::OK();
            }

            PLATFORM_CHECK(conv2d, ENGINE_CUDA) {
                return true;
            }
        }
    }
}
