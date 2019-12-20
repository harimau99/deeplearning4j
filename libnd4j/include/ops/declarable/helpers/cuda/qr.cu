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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/qr.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void qr_(NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        Nd4jLong M = input->sizeAt(-2);
        Nd4jLong N = input->sizeAt(-1);
        Nd4jLong lastDim = input->rankOf() - 1;
        Nd4jLong preLastDim = input->rankOf() - 2;
//        std::unique_ptr<ResultSet> listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
//        std::unique_ptr<ResultSet> listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
//        std::unique_ptr<ResultSet> listDiag(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));

//        for (auto batch = 0; batch < listOutQ->size(); ++batch) {
            //qr here
//        }

    }

    void qr(nd4j::LaunchContext* context, NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (input, outputQ, outputR, fullMatricies), FLOAT_TYPES);
    }

}
}
}

