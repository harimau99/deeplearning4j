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
#include <helpers/MmulHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    NDArray matrixMinor(NDArray& in, Nd4jLong col) {
        NDArray m = in.ulike();

        for (auto i = 0; i < col; i++)
            m.t<T>(i,i) = T(1.f);
        for (int i = col; i < m.rows(); i++)
            for (int j = col; j < m.columns(); j++)
                m.t<T>(i,j) = in.t<T>(i,j);
        return m;
    }

/* m = I - v v^T */
    template <typename T>
    NDArray vmul(NDArray const& v, int n)
    {
        NDArray res('c', {n,n}, v.dataType()); // x = matrix_new(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                res.t<T>(i, j) = -2 *  v.t<T>(i) * v.t<T>(j) + (i == j?T(1):T(0));

        return res;
    }

    template <typename T>
    void qrSingle(NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatricies) {
        Nd4jLong M = matrix->sizeAt(-2);
        Nd4jLong N = matrix->sizeAt(-1);
        NDArray q[M];
        NDArray z = *matrix;
        for (auto k = 0; k < N && k < M - 1; k++) { // loop for columns, but not further then row number
            NDArray e('c', {M}, DataTypeUtils::fromT<T>()); // two internal buffers and scalar for squared norm
            q[k] = Q->ulike();
            q[k].nullify();
            z = matrixMinor<T>(z, k); // minor computing for current column with given matrix z (initally is a input matrix)
            z.printIndexedBuffer("Z!!!");
            auto currentColumn = z({0, 0, k, k+1}); // retrieve k column from z to x buffer
            auto norm = currentColumn.reduceAlongDims(reduce::Norm2, {0});
            nd4j_printf("%f\n", norm.t<T>(0));
            if (matrix->t<T>(k,k) > T(0.f)) // negate on positive matrix diagonal element
                norm.t<T>(0) = -norm.t<T>(0);
            e.t<T>(k) = T(1.f); // e - is filled by 0 vector except diagonal element (filled by 1)
            e.printIndexedBuffer("E0");
            e = currentColumn + norm * e; // e[i] = x[i] + a * e[i] for each i from 0 to n - 1
            e.printIndexedBuffer("E+");
            auto normE = e.reduceAlongDims(reduce::Norm2, {0});
            e /= normE;
            e.printIndexedBuffer("E!!!");
            q[k] = vmul<T>(e, M);
            q[k].printIndexedBuffer("And Qk");
//            z = q[k] * z;
            z.printIndexedBuffer("a_Step1");
            auto qQ = z.ulike();
            MmulHelper::matmul(&q[k], &z, &qQ, false, false);
            q[k].printIndexedBuffer("q1");
            qQ.printIndexedBuffer("Step1");
            z = qQ;
        }
        *Q = q[0]; //
        MmulHelper::matmul(&q[0], &z, R, false, false);
        //*R = q[0] * *matrix;
        for (int i = 1; i < N && i < M - 1; i++) {
//            z = q[i] * *Q;
            z = Q->ulike();
            MmulHelper::matmul(&q[i], Q, &z, false, false);
            *Q = z;
        }
//        *R = *Q * *matrix;
        MmulHelper::matmul(Q, matrix, R, false, false);
        Q->transposei();// transpose of matrix Q - now it in non-refined square state (MxM)
    }

    template <typename T>
    void qr_(NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        Nd4jLong lastDim = input->rankOf() - 1;
        Nd4jLong preLastDim = input->rankOf() - 2;
        std::unique_ptr<ResultSet> listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        std::unique_ptr<ResultSet> listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        std::unique_ptr<ResultSet> listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));

        for (auto batch = 0; batch < listOutQ->size(); ++batch) {
            //qr here
            qrSingle<T>(listInput->at(batch), listOutQ->at(batch), listOutR->at(batch), fullMatricies);
        }

    }

    void qr(nd4j::LaunchContext* context, NDArray* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (input, outputQ, outputR, fullMatricies), FLOAT_TYPES);
    }

}
}
}

