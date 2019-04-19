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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_barnes_edge_force)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
namespace ops  {
		
    CUSTOM_OP_IMPL(barnes_edge_forces, 5, 3, false, 0, 1) {
        auto rowP  = INPUT_VARIABLE(0);
        auto colP  = INPUT_VARIABLE(1);
        auto valP  = INPUT_VARIABLE(2);
        auto dataP  = INPUT_VARIABLE(3);
        auto bufP  = INPUT_VARIABLE(4);
        auto N = INT_ARG(0);

        auto output = OUTPUT_VARIABLE(0);
        auto outputData = OUTPUT_VARIABLE(1);
        auto outputBuf = OUTPUT_VARIABLE(2);

            REQUIRE_TRUE(rowP->isVector(), 0, "barnes_edge_force op: row input must be a vector, but its rank is %i instead !", rowP->rankOf());
            REQUIRE_TRUE(colP->isVector(), 0, "barnes_edge_force op: col input must be a vector, but its rank is %i instead !", colP->rankOf());
        outputBuf->assign(bufP);
        outputData->assign(dataP);
        helpers::barnes_edge_forces(rowP, colP, valP, N, output, *outputData, *outputBuf);

        return Status::OK();
    }

    DECLARE_TYPES(barnes_edge_forces) {
        getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_INTS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedInputTypes(2, {ALL_INTS, ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_INTS, ALL_FLOATS})
        ->setAllowedInputTypes(4, {ALL_INTS, ALL_FLOATS})
        ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS})
        ->setAllowedOutputTypes(1, {ALL_INTS, ALL_FLOATS})
        ->setAllowedOutputTypes(2, {ALL_INTS, ALL_FLOATS})
        ->setSameMode(false);
    }

    DECLARE_SHAPE_FN(barnes_edge_forces) {
        Nd4jLong* dataShape;
        Nd4jLong* bufShape;
        Nd4jLong* outShapeInfo;
        COPY_SHAPE(inputShape->at(3), dataShape);
        COPY_SHAPE(inputShape->at(4), bufShape);
        outShapeInfo = ShapeBuilders::copyShapeInfoAndType(inputShape->at(3), inputShape->at(3), false, block.getWorkspace());
        return SHAPELIST(outShapeInfo, dataShape, bufShape);
    }


}
}

#endif