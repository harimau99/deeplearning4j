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

#ifndef SD_CUDNNUTILS_H
#define SD_CUDNNUTILS_H

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <dll.h>

#include <cudnn.h>

namespace nd4j {
    namespace ops {
        namespace platforms {

            DECLARE_PLATFORM(conv2d, ENGINE_CUDA);


            FORCEINLINE cudnnDataType_t cudnnDataType(nd4j::DataType dataType) {
                switch (dataType) {
                    case nd4j::DataType::FLOAT32:
                        return CUDNN_DATA_FLOAT;
                    case nd4j::DataType::DOUBLE:
                        return CUDNN_DATA_DOUBLE;
                    case nd4j::DataType::HALF:
                        return CUDNN_DATA_HALF;
                    case nd4j::DataType::INT32:
                        return CUDNN_DATA_INT32;
                    case nd4j::DataType::INT8:
                        return CUDNN_DATA_INT8;
                    default:
                        throw datatype_exception::build("Unsupported data type", dataType);
                }
            }
        }
    }
}

#endif //SD_CUDNNUTILS_H
