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
#include <ops/declarable/helpers/print_variable.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(print_variable, 1, 1, true, 0, 0) {
            // TODO: make this op compatible with ArrayList etc
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);
            std::string str;

            input->printShapeInfo("Input ShapeInfo");
            if (block.width() == 2) {
                auto message = INPUT_VARIABLE(1);
                REQUIRE_TRUE(message->isS(), 0, "print_variable: message variable must be a String");

                str = message->e<std::string>(0);
            }

            bool printSpecial = false;
            if (block.numB() > 0)
                printSpecial = B_ARG(0);

            if (printSpecial && !nd4j::Environment::getInstance()->isCPU()) {
                // only specific backends support special printout. for cpu-based backends it's the same as regular print

                if (block.width() == 2)
                    helpers::print_special(*block.launchContext(), *input, str);
                else
                    helpers::print_special(*block.launchContext(), *input);
            } else {
                // optionally add message to the print out
                if (block.width() == 2) {
                    input->printIndexedBuffer(str.c_str());
                } else {
                    input->printIndexedBuffer();
                }
            }

            return Status::OK();
        }

        DECLARE_TYPES(print_variable) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(1, {ALL_STRINGS})
                    ->setAllowedOutputTypes(0, nd4j::DataType::ANY);
        }

        DECLARE_SHAPE_FN(print_variable) {
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(DataType::INT32));
        }
    }
}