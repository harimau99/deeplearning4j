/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

#include <array/InteropDataBuffer.h>

namespace nd4j {
    InteropDataBuffer::InteropDataBuffer(std::shared_ptr<DataBuffer> databuffer) {
        _dataBuffer = databuffer;
    }

    InteropDataBuffer::InteropDataBuffer(size_t elements, nd4j::DataType dtype, bool allocateBoth) {
        if (elements == 0) {
            _dataBuffer = std::make_shared<DataBuffer>();
        } else {
            _dataBuffer = std::make_shared<DataBuffer>(elements, dtype, nullptr, allocateBoth);
        }
    }

    std::shared_ptr<DataBuffer> InteropDataBuffer::getDataBuffer() const {
        return _dataBuffer;
    }

    std::shared_ptr<DataBuffer> InteropDataBuffer::dataBuffer() {
        return _dataBuffer;
    }

    void* InteropDataBuffer::primary() {
        return _dataBuffer->primary();
    }

    void* InteropDataBuffer::special() {
        return _dataBuffer->special();
    }
}
