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
    InteropDataBuffer::InteropDataBuffer(InteropDataBuffer &dataBuffer, Nd4jLong offset) {
        _dataBuffer = dataBuffer.getDataBuffer();
        _offset = offset + dataBuffer.offset();
    }

    InteropDataBuffer::InteropDataBuffer(std::shared_ptr<DataBuffer> databuffer) {
        _dataBuffer = databuffer;
    }

    InteropDataBuffer::InteropDataBuffer(size_t elements, nd4j::DataType dtype, bool allocateBoth) {
        if (elements == 0) {
            _dataBuffer = std::make_shared<DataBuffer>();
            _dataBuffer->setDataType(dtype);
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

    void* InteropDataBuffer::primary() const {
        return _dataBuffer->primary();
    }

    void* InteropDataBuffer::special() const {
        return _dataBuffer->special();
    }

    void InteropDataBuffer::setPrimary(void* ptr, size_t length) {
        _dataBuffer->setPrimaryBuffer(ptr, length);
    }

    void InteropDataBuffer::setSpecial(void* ptr, size_t length) {
        _dataBuffer->setSpecialBuffer(ptr, length);
    }

    uint64_t InteropDataBuffer::offset() const {
        return _offset;
    }

    void InteropDataBuffer::setOffset(uint64_t offset) {
        _offset = offset;
    }


    void InteropDataBuffer::registerSpecialUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList) {
        for (const auto &v:writeList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->writeSpecial();
        }
    }

    void InteropDataBuffer::prepareSpecialUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList, bool synchronizeWritables) {
        for (const auto &v:readList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->syncToSpecial();
            v->getDataBuffer()->readSpecial();
        }
    }

    void InteropDataBuffer::registerPrimaryUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList) {
        for (const auto &v:writeList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->writePrimary();
        }
    }

    void InteropDataBuffer::preparePrimaryUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList, bool synchronizeWritables) {
        for (const auto &v:readList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->syncToPrimary(LaunchContext::defaultContext());
            v->getDataBuffer()->readPrimary();
        }
    }
}
