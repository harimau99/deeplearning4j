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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_split_string)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(compat_split, 2, 2, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            return Status::OK();
        };

        DECLARE_SHAPE_FN(compat_split) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            return SHAPELIST();
        }

        DECLARE_TYPES(compat_split) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_STRINGS})
                    ->setAllowedOutputTypes({ALL_STRINGS});
        }
    }
}

#endif