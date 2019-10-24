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

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(print_variable, 1, 1, true) {
            // TODO: make this op compatible with ArrayList etc
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // optionally add message to the print out
            if (block.width() == 2) {
                auto message = INPUT_VARIABLE(1);

                if (message->isS()) {
                    auto str = message->e<std::string>(0);

                    input->printIndexedBuffer(str.c_str());
                }
            } else {
                input->printIndexedBuffer();
            }

            if (!block.isInplace())
                output->assign(input);

            return Status::OK();
        }

        DECLARE_TYPES(print_variable) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(1, {ALL_STRINGS})
                    ->setAllowedOutputTypes(0, nd4j::DataType::ANY);
        }
    }
}