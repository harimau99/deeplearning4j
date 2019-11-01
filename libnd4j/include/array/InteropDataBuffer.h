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

#include <dll.h>
#include <array/DataBuffer.h>
#include <array/DataType.h>
#include <memory>

#ifndef LIBND4J_INTEROPDATABUFFER_H
#define LIBND4J_INTEROPDATABUFFER_H

namespace nd4j {
    /**
     * This class is a wrapper for DataBuffer, suitable for sharing DataBuffer between front-end and back-end languages
     */
    class InteropDataBuffer {
    private:
        std::shared_ptr<DataBuffer> _dataBuffer;
        uint64_t _offset;
    public:
        InteropDataBuffer(InteropDataBuffer &dataBuffer, Nd4jLong offset);
        InteropDataBuffer(std::shared_ptr<DataBuffer> databuffer);
        InteropDataBuffer(size_t elements, nd4j::DataType dtype, bool allocateBoth);
        ~InteropDataBuffer() = default;

#ifndef __JAVACPP_HACK__
        std::shared_ptr<DataBuffer> getDataBuffer() const;
        std::shared_ptr<DataBuffer> dataBuffer();
#endif

        void* primary();
        void* special();

        uint64_t offset();
        void setOffset(uint64_t offset);

        void setPrimary(void* ptr, size_t length);
        void setSpecial(void* ptr, size_t length);
    };
}


#endif //LIBND4J_INTEROPDATABUFFER_H
