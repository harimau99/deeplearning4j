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
// Created by raver119 on 20/04/18.
//

#include <helpers/StringUtils.h>

namespace nd4j {
    static FORCEINLINE bool match(uint8_t *haystack, uint8_t *needle, uint64_t length) {
        for (int e = 0; e < length; e++)
            if (haystack[e] != needle[e])
                return false;

        return true;
    }

    uint64_t StringUtils::countSubarrays(void *vhaystack, uint64_t haystackLength, void *vneedle, uint64_t needleLength) {
        auto haystack = reinterpret_cast<uint8_t*>(vhaystack);
        auto needle = reinterpret_cast<uint8_t*>(vneedle);

        uint64_t number = 0;

        for (int e = 0; e < haystackLength - needleLength; e++) {
            if (match(&haystack[e], needle, needleLength))
                number++;
        }

        return number;
    }
}
