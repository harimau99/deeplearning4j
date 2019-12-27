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

#include "testlayers.h"
#include <initializer_list>
#include <ops/declarable/PlatformHelper.h>

#ifdef HAVE_CUDNN

#include <ops/declarable/platform/cudnn/cudnnUtils.h>

#endif

class CuDnnTests : public testing::Test {
public:

};

static void printer(std::initializer_list<nd4j::ops::platforms::PlatformHelper*> helpers) {

    for (auto v:helpers) {
        nd4j_printf("Initialized [%s]\n", v->name().c_str());
    }
}


TEST_F(CuDnnTests, helpers_includer) {
    // we need this block, to make sure all helpers are still available within binary, and not optimized out by linker
#ifdef HAVE_CUDNN
    nd4j::ops::platforms::PLATFORM_conv2d_ENGINE_CUDA conv2d;


    printer({&conv2d});
#endif
}