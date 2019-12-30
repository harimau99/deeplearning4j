/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_multinomial)

#include <ops/declarable/CustomOperations.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/helpers/random.h>

namespace nd4j {
    namespace ops {
        ///////////////////////
        /**
         * multinomial (categorical) random generator
         * takes 2D ndarray with logits with shape [batch_size (N), num_classes (K)]
         * represents the unnormalized log-probabilities for all classes.
         * Int arguments: 0 - scalar value of samples number, number of independent samples to draw for each experiment 1,N.
         * Int arguments: 1 - optional argument, corresponds to dimension with batch_size
         * Int arguments: 2 - optional argument, integer type to use for the output. Default int64.
         * Int arguments: 3 - optional argument, integer used to create a random seed for the distribution
         */
        CUSTOM_OP_IMPL(random_multinomial, 1, 1, false, 0, 4) {
            
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);
            auto nSamples = INT_ARG(0);
            const int argSize = block.getIArguments()->size();

            REQUIRE_TRUE(argSize >= 1, 0, "Have to be specified atleast number of samples,"
                " number of specified arguments %i ", argSize);

            // just skip op number of samples = 0
            if (0 == nSamples) 
                return Status::OK();

            REQUIRE_TRUE(!input->isEmpty(), 0, "Number of classes should be positive, got 0. ");

            const int rank = input->rankOf();
            REQUIRE_TRUE(rank == 2, 0, "Logits should be a matrix, with requirement rank: %i == 2 ", rank);
            const int dimC = argSize > 1 ? (INT_ARG(1) >= 0 ? INT_ARG(1) : INT_ARG(1) + rank) : rank - 1;

            auto dimA = (0 == dimC) ? 1 : 0;
            if (0 == input->sizeAt(dimA)) {
                *output = 0;
                return Status::OK();
            }

            auto rng = block.randomGenerator();
            if (argSize > 3) {
                rng.setSeed(static_cast<int>(INT_ARG(3)));
            }

            helpers::fillRandomMultiNomial(block.launchContext(), rng, *input, *output, dimC);
            return Status::OK();
        }


        DECLARE_SHAPE_FN(random_multinomial) {

            const int argSize = block.getIArguments()->size();
            REQUIRE_TRUE(argSize >= 1, 0, "Have to be specified atleast number of samples,"
                " number of specified arguments %i ", argSize);

            auto nSamples = INT_ARG(0);

            auto input = INPUT_VARIABLE(0);
            REQUIRE_TRUE(!input->isEmpty(), 0, "Number of classes should be positive, got 0. ");
            const int rank = input->rankOf();
            REQUIRE_TRUE(rank == 2, 0, "Logits should be a matrix, with requirement rank: %i == 2 ", rank);
            const int dimC = argSize > 1 ? (INT_ARG(1) >= 0 ? INT_ARG(1) : INT_ARG(1) + rank) : rank - 1;
           

            auto nShape = input->getShapeAsVector();
            auto dimA = (0 == dimC) ? 1 : 0;
            nShape[dimA] = nSamples;

            DataType nType = (argSize > 2) ? ( INT_ARG(2) >= 0 ? static_cast<DataType>(INT_ARG(2)) : nd4j::DataType::INT64) : nd4j::DataType::INT64;
            return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(nType, input->ordering(), nShape));
        }
        
        DECLARE_TYPES(random_multinomial) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, { ALL_INDICES });
        }
    }
}

#endif