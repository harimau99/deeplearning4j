/*******************************************************************************
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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/random.h>
//#include <vector>
#include <memory>
//#include <graph/Context.h>
#include <ShapeUtils.h>
#include <helpers/RandomLauncher.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {

        Nd4jLong* broadcasted = nullptr;
        if (beta != nullptr)
            ShapeUtils::evalBroadcastShapeInfo(*alpha, *beta, true, broadcasted, context->getWorkspace());
        else
            broadcasted = alpha->shapeInfo();
        auto step = shape::length(broadcasted);
        auto shift = output->lengthOf() / step;

        auto copyAlpha = alpha;
        auto copyBeta = beta;
        if (beta != nullptr) {
            NDArray alphaBroadcasted(broadcasted, alpha->dataType(), false, context);
            NDArray betaBroadcasted(broadcasted, beta->dataType(), false, context);

            copyAlpha = new NDArray(alphaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *alpha));
            copyBeta  = new NDArray(betaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *beta));

        }
//        bool directAlpha = alpha->ews() == 1 && alpha->ordering() == 'c';
        bool directOutput = output->ews() == 1 && output->ordering() == 'c';
        T* outputBuf = output->dataBuffer()->primaryAsT<T>();

        PRAGMA_OMP_PARALLEL_FOR
        for (auto k = 0; k < shift; k++) {
            auto pos = k * step;
            auto u = rng.relativeT<T>(k, 0., 1.);
            for (auto e = 0; e < step; e++)
                    if (directOutput) {
                        outputBuf[pos + e] = math::nd4j_igamma<T, T, T>(copyAlpha->t<T>(e),
                                                                        beta != nullptr ? copyBeta->t<T>(e) * u : u);
                    }
                    else {
                        output->t<T>(pos + e) = math::nd4j_igamma<T, T, T>(copyAlpha->t<T>(e),
                                                                        beta != nullptr ? copyBeta->t<T>(e) * u : u);
                    }
        }

        if (beta != nullptr) {
            delete copyAlpha;
            delete copyBeta;
            //delete broadcasted;
        }
    }

    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), FLOAT_NATIVE);
    }
    BUILD_SINGLE_TEMPLATE(template void fillRandomGamma_, (LaunchContext* context,
            graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output), FLOAT_NATIVE);

    /*
     * algorithm Poisson generator based upon the inversion by sequential search:[48]:505
    init:
         Let x ← 0, p ← e−λ, s ← p.
         Generate uniform random number u in [0,1].
    while u > s do:
         x ← x + 1.
         p ← p * λ / x.
         s ← s + p.
    return x.
     * */
    template <typename T>
    void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        auto shift = output->lengthOf() / lambda->lengthOf();
        auto step = lambda->lengthOf();
        T* lambdaBuf = lambda->dataBuffer()->primaryAsT<T>();
        T* outputBuf = output->dataBuffer()->primaryAsT<T>();
        bool directLa = lambda->ews() == 1 && lambda->ordering() == 'c';
        bool directOut = output->ews() == 1 && output->ordering() == 'c';
        PRAGMA_OMP_PARALLEL_FOR
        for (auto k = 0; k < shift; k++) {
            auto pos = k * step;
            auto u = rng.relativeT<T>(k, 0., 1.);
            for (auto e = 0; e < step; e++) {
                auto p = math::nd4j_exp<T, T>(-lambda->t<T>(e));
                auto s = p;
                auto x = T(0.f);
                while (u > s) {
                    x += 1.f;
                    p *= directLa?lambdaBuf[e]/x:lambda->t<T>(e) / x;
                    s += p;
                }
                if (directOut)
                    outputBuf[pos + e] = x;
                else
                    output->t<T>(pos + e) = x;
            }
        }
    }

    void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), FLOAT_NATIVE);
    }
    BUILD_SINGLE_TEMPLATE(template void fillRandomPoisson_, (LaunchContext* context,
            graph::RandomGenerator& rng, NDArray* lambda, NDArray* output), FLOAT_TYPES);

    template <typename T>
    void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        T minVal = T(0);
        T maxVal = DataTypeUtils::max<T>();
        if (min)
            minVal = min->t<T>(0);
        if (max)
            maxVal = max->t<T>(0);

        if (output->isR())
            RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
        else {
            PRAGMA_OMP_PARALLEL_FOR
            for (auto i = 0; i < output->lengthOf(); i++) {
                output->t<T>(i) = rng.relativeT<T>(i, minVal, maxVal);
            }
        }
    }

    void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomUniform_, (context, rng, min, max, output), NUMERIC_TYPES);
    }
}
}
}