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
// Created by GS <sgazeos@gmail.com> at 12/20/2019
//

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/qr.h>

#if NOT_EXCLUDED(OP_qr)
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(qr, 1, 2, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto outputQ = OUTPUT_VARIABLE(0);
            auto outputR = OUTPUT_VARIABLE(1);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "qr: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(outputQ->sizeAt(-1) == input->sizeAt(-2), 0, "qr: The last dimmensions should be equal with result Q, but %i and %i are given", outputQ->sizeAt(-1), input->sizeAt(-2));
            REQUIRE_TRUE(outputR->sizeAt(-1) == input->sizeAt(-1), 0, "qr: The last dimmensions should be equal with result R, but %i and %i are given", outputR->sizeAt(-1), input->sizeAt(-1));

            helpers::qr(block.launchContext(), input, outputQ, outputR);\
            return Status::OK();
        }

        DECLARE_SHAPE_FN(qr) {
            auto inShape = inputShape->at(0);

            Nd4jLong* shapeQ;
            Nd4jLong* shapeR;
            int targetRank = shape::rank(inShape); // last two dimensions will be reduced to scalar
            auto shape = ShapeUtils::shapeAsVector(inShape);
            auto rank = shape.size();
            shape[rank - 1] = shape::sizeAt(inShape, -1);
            shape[rank - 2] = shape[rank - 1];
            shapeQ = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), targetRank, shape::shapeOf(inShape));
            shapeR = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), shape);

            return SHAPELIST(shapeQ, shapeR);
        }

        DECLARE_TYPES(qr) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif
