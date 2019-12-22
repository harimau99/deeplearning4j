/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <memory>

using namespace nd4j;


class DeclarableOpsTests13 : public testing::Test {
public:

    DeclarableOpsTests13() {
        //printf("\n");
        //fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests13, test_pow_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {2.f, 2.f, 2.f, 2.f});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 3});
    auto e = NDArrayFactory::create<float>('c', {2, 2}, {8.f, 8.f, 8.f, 8.f});

    nd4j::ops::Pow op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_1) {
    auto start = NDArrayFactory::create<int>(0);
    auto limit = NDArrayFactory::create<int>(0);

    nd4j::ops::range op;
    auto result = op.execute({&start, &limit}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_2) {

    nd4j::ops::range op;
    auto result = op.execute({}, {1.0, 1.0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_3) {

    nd4j::ops::range op;
    auto result = op.execute({}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_argmax_edge_1) {
    auto ctx = new Context(1);
    auto arr = NDArrayFactory::create_<float>('c', {1024,1});

    ctx->setInputArray(0, arr, true);
    ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong >('c', {1}), true);
    ctx->setInputArray(1, NDArrayFactory::create_<Nd4jLong >(0), true);   //Axis 0


    nd4j::ops::argmax op;
    auto result = op.execute(ctx);
    ASSERT_EQ(Status::OK(), result);

    //nd4j_printf("Done\n","");
    delete ctx;
}

TEST_F(DeclarableOpsTests13, test_add_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 768});
    auto y = NDArrayFactory::create<float>('c', {768});
    auto e = NDArrayFactory::create<float>('c', {1, 768});;
    y. assign(1.0f);
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(DeclarableOpsTests13, test_listdiff_1) {
    auto x = NDArrayFactory::create<int>('c', {4}, {0, 1, 2, 3});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 1});

    auto od = NDArrayFactory::create<int>('c', {2});
    auto oi = NDArrayFactory::create<int>('c', {2});

    nd4j::ops::listdiff op;
    auto result = op.execute({&x, &y}, {&od, &oi}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests13, test_greater_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 1});
    auto y = NDArrayFactory::create<float>('c', {1, 4});

    nd4j::ops::greater op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_eval_reduction_shape_1) {
    Nd4jLong axis = 0L;
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {4, 2});
    auto y = NDArrayFactory::create<Nd4jLong>('c', {1}, {axis});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {2}, {1, 2});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.execute({&x, &y}, {}, {}, {true});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, test_or_1) {

    NDArray x('c', {4}, {false, true, false, true}, nd4j::DataType::BOOL);
    NDArray y('c', {4}, {false, false, true, true}, nd4j::DataType::BOOL);
    NDArray e('c', {4}, {false, true, true, true}, nd4j::DataType::BOOL);

    NDArray z('c', {4}, nd4j::DataType::BOOL);

    x.applyPairwiseTransform(pairwise::Or, y, z);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_and_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, false, false, true});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::And, y, z);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_xor_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, true, true, false});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::Xor, y, z);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_1) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {1,2,3, 4, 5, 6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {1,-2,3, -4, 5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1.2,2.2,3.2,4.2,5.2,6.2});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_2) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {1, -2, 3, -4, 5, -6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {1, -2, 3, -4, 5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1.2, 0.01, 3.2, 0.01, 5.2, 0.01});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    //ASSERT_EQ(e, z);
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_GainsTest_3) {
    auto x = NDArrayFactory::create<double>('c', {2,3}, {-1, 2, -3, 4, -5, 6});
    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    auto exp = NDArrayFactory::create<double>('c', {2,3}, {0.01, 2.2, 0.01, 4.2, 0.01, 6.2});
    nd4j::ops::barnes_gains op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Gains out");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_1) {
    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {2}, {2, 3});
    auto cols = NDArrayFactory::create<int>('c', {5}, {0, 2, 1, 4, 3});
    auto vals = NDArrayFactory::create<double>('c', {5}, {10., 20., 30., 40., 50.});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp1 = NDArrayFactory::create<double>('c', {5,4}, {-1.846154, -1.846154, -1.846154, -1.846154, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {1});


    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    ASSERT_TRUE(exp1.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_2) {
    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {3}, {1,2,3});
    auto cols = NDArrayFactory::create<int>('c', {5}, {1, 2, 0, 4, 3});
    auto vals = NDArrayFactory::create<double>('c', {5}, {10., 20., 30., 40., 50.});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('c', {5,4}, {-0.622568, -0.622568, -0.622568, -0.622568, 1.846154, 1.846154, 1.846154, 1.846154, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {2});


    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_EdgeForceTest_3) {
    auto data = NDArrayFactory::create<double>('c', {11, 5}, {0.3, 0.2625, 0.2674, 0.8604, 0.4803, 0.1096, 0.795, 0.5918, 0.2738, 0.952, 0.969, 0.8586, 0.8088, 0.5338, 0.5961, 0.7187, 0.463, 0.0867, 0.7748, 0.4802, 0.2493, 0.3227, 0.3064, 0.698, 0.7977, 0.7674, 0.168, 0.3107, 0.0217, 0.138, 0.8619, 0.8413, 0.5285, 0.9703, 0.6774, 0.2624, 0.4374, 0.1569, 0.1107, 0.0601, 0.4094, 0.9564, 0.5994, 0.8279, 0.3859, 0.6202, 0.7604, 0.0788, 0.0865, 0.7445, 0.6548, 0.3385, 0.0582, 0.6249, 0.7432});
    auto rows = NDArrayFactory::create<int>({0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99});
    auto cols = NDArrayFactory::create<int>({4, 3, 10, 8, 6, 7, 1, 5, 9, 4, 9, 8, 10, 2, 0, 6, 7, 3, 6, 8, 3, 9, 10, 1, 4, 0, 5, 10, 0, 4, 6, 8, 9, 2, 5, 7, 0, 10, 3, 1, 8, 9, 6, 7, 2, 7, 9, 3, 10, 0, 4, 2, 8, 1, 2, 8, 3, 10, 0, 4, 9, 1, 5, 5, 9, 0, 3, 10, 4, 8, 1, 2, 6, 2, 0, 3, 4, 1, 10, 9, 7, 10, 1, 3, 7, 4, 5, 2, 8, 6, 3, 4, 0, 9, 6, 5, 8, 7, 1});
    auto vals = NDArrayFactory::create<double>({0.6199614579042966, 0.19644097697184246, 0.13824979367331638, 0.01949900138247239, 0.008923198738222747, 0.008392793826291798, 0.0033348224714784204, 0.0026246189757042166, 0.0025733360563748838, 0.5877136110798608, 0.28250257562439585, 0.08098135424273815, 0.014862718272075049, 0.01219187321450782, 0.01152346362368888, 0.004243137936786281, 0.0034626999030188577, 0.0025185661029283168, 0.6777005651521399, 0.18321248222489303, 0.04018202465629351, 0.02941935889988646, 0.02164146250842832, 0.019898422145651618, 0.011683461395713935, 0.008439076090480863, 0.007823146926512332, 0.6770900431883232, 0.16617511239723026, 0.06039349887686468, 0.04650913399744179, 0.016886531410284355, 0.014591049666869658, 0.006407638669806174, 0.006074413005122801, 0.0058725787880570205, 0.6278185083409108, 0.235127797795446, 0.07023700015217448, 0.030885483448633774, 0.01229522088606573, 0.009238279699136107, 0.008219511168822047, 0.004303744819835723, 0.0018744536889749907, 0.7122603898978483, 0.07862620103245824, 0.07061257369349086, 0.06721483653169834, 0.028957853952131768, 0.01778978123182596, 0.01481713955181034, 0.005492728917348627, 0.0042284951913875955, 0.5266844101016999, 0.3304104787383107, 0.10930017433210941, 0.018514917515240075, 0.006969360999637938, 0.0063776901975396, 0.0010590388116165708, 6.526830884629785E-4, 3.1246215383067865E-5, 0.7176179284835663, 0.08741734015883978, 0.05927699083866909, 0.04663169573956976, 0.03287576269194147, 0.02993912340339554, 0.013365238657916641, 0.010616858763291145, 0.002259061262810172, 0.6891905160321706, 0.1397658294110526, 0.05438284759722162, 0.05437184733708826, 0.028683289714498808, 0.020986120697576355, 0.007218358114741088, 0.0032834770669826364, 0.002117714028667893, 0.6823873496503976, 0.1345267083671607, 0.08712863515505885, 0.04286621088946242, 0.02544804597749639, 0.01689343932533317, 0.007219134659004873, 0.0019232929717404616, 0.0016071830043453991, 0.6425809622897437, 0.18474464886441516, 0.10897036475298316, 0.03466939253836615, 0.013288054277817787, 0.005149178177380355, 0.0037974063158903518, 0.0037851733015991287, 0.0030148194818042273});
    //auto buf = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('c', {11, 5}, {-0.080205, -0.085862, 0.024045, 0.133551, -0.199896, -0.170597, 0.187301, 0.205824, -0.165268, 0.131228, 0.155135, 0.021446, 0.217583, -0.262873, -0.021075, 0.114537, 0.088023, -0.039205, 0.087984, -0.179565, -0.132683, 0.003677, 0.072081, -0.068737, 0.204481, 0.287223, -0.193989, 0.104569, -0.123401, -0.036368, 0.086745, 0.002961, -0.091327, 0.234853, 0.120270, -0.304006, 0.128305, -0.084867, -0.017550, -0.130837, -0.288569, 0.124679, 0.054078, -0.034187, -0.192599, 0.033196, 0.228182, -0.044972, -0.314217, 0.020287, 0.054427, -0.078887, -0.078246, -0.104543, 0.169803});
    //auto exp2 = NDArrayFactory::create<double>({-4., -4., -4., -4.
    //std::vector<NDArray*> exp({&exp1, &exp2});
    //data.assign(1.0); //linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_edge_forces op;
    auto result = op.execute({&rows, &cols, &vals, &data}, {}, {11});

    //nd4j_printf("rows %lld, cols %lld, vals %lld, res full %lld\n", rows.lengthOf(), cols.lengthOf(), vals.lengthOf(), exp1.lengthOf());
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(0)->printBuffer("Output");
    //exp.printBuffer("Expect");
    //result->at(0)->printShapeInfo("Shape output");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_1) {
//    auto data = NDArrayFactory::create<double>('c', {5,4});
    auto rows = NDArrayFactory::create<int>('c', {2}, {0, 1});
    auto cols = NDArrayFactory::create<int>('c', {4}, {0, 1, 1, 0});
    auto vals = NDArrayFactory::create<double>('c', {4}, {20., 30., 40., 50.});
    auto exp = NDArrayFactory::create<double>('c', {1,1}, {20.});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {1});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized1");
    ASSERT_TRUE(exp.equalsTo(result->at(2)));

    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_2) {
    auto rows = NDArrayFactory::create<int>('c', {4}, {0, 2, 2, 3});
    auto cols = NDArrayFactory::create<int>('c', {8}, {0, 1, 1, 0, 0, 1, 1, 1});
    auto vals = NDArrayFactory::create<double>('c', {8}, {20., 30., 40., 50., 120., 130., 140., 150.});
    auto exp = NDArrayFactory::create<double>('c', {1,5}, {20., 15., 15., 20., 20.});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {3});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized2");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    ASSERT_TRUE(exp.equalsTo(result->at(2)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_3) {
    auto rows = NDArrayFactory::create<int>('c', {12}, {0, 2, 3, 5, 7, 8, 9, 11, 12, 14, 18, 21});
    auto cols = NDArrayFactory::create<int>('c', {24}, {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 0, 2, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5});
    auto vals = NDArrayFactory::create<double>('c', {24}, {20., 30., 40., 50., 120., 130., 140., 150.,220., 230., 240., 250., 2120., 2130., 2140., 2150., 320., 330., 340., 350., 3120., 3130., 3140., 3150.});
    auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
//    data.linspace(1);

//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {11});
    ASSERT_EQ(result->status(), Status::OK());
    //result->at(2)->printBuffer("Symmetrized3");
    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    //ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests13, BarnesHutTsne_symmetrized_4) {
    auto rows = NDArrayFactory::create<int>({0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
    auto cols = NDArrayFactory::create<int>({4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
    auto vals = NDArrayFactory::create<double>( {0.6200,    0.1964,    0.1382,    0.0195,    0.0089,    0.0084,    0.0033,    0.0026,    0.0026,    0.5877,    0.2825,    0.0810,    0.0149,    0.0122,    0.0115,    0.0042,    0.0035,    0.0025,    0.6777,    0.1832,    0.0402,    0.0294,    0.0216,    0.0199,    0.0117,    0.0084,    0.0078,    0.6771,    0.1662,    0.0604,    0.0465,    0.0169,    0.0146,    0.0064,    0.0061,    0.0059,    0.6278,    0.2351,    0.0702,    0.0309,    0.0123,    0.0092,    0.0082,    0.0043,    0.0019,    0.7123,    0.0786,    0.0706,    0.0672,    0.0290,    0.0178,    0.0148,    0.0055,    0.0042,    0.5267,    0.3304,    0.1093,    0.0185,    0.0070,    0.0064,    0.0011,    0.0007, 3.1246e-5,    0.7176,    0.0874,    0.0593,    0.0466,    0.0329,    0.0299,    0.0134,    0.0106,    0.0023,    0.6892,    0.1398,    0.0544,    0.0544,    0.0287,    0.0210,    0.0072,    0.0033,    0.0021,    0.6824,    0.1345,    0.0871,    0.0429,    0.0254,    0.0169,    0.0072,    0.0019,    0.0016,    0.6426,    0.1847,    0.1090,    0.0347,    0.0133,    0.0051,    0.0038,    0.0038,    0.0030});
    //auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
//    data.linspace(1);
    auto exp4 = NDArrayFactory::create<double>('c', {1, 108}, {0.6239, 0.1813, 0.1236, 0.03695, 0.00795, 0.03385, 0.0074, 0.0158, 0.0013, 0.0042, 0.0074, 0.3093, 0.2085, 0.051, 0.00895, 0.01605, 0.00245, 0.00705, 0.00125, 0.0021, 0.01605, 0.6022, 0.1615, 0.0233,
                                                0.0183, 0.0108, 0.0068, 0.0042, 0.0113, 0.00115, 0.1813, 0.00125, 0.0233, 0.65985, 0.0653, 0.0779, 0.03565, 0.05085, 0.03835, 0.02625, 0.6239, 0.3093, 0.0068, 0.0653, 0.2099, 0.0205, 0.0173, 0.0073,
                                                0.0171, 0.0089, 0.0158, 0.0113, 0.03835, 0.71495, 0.04775, 0.03615, 0.0089, 0.00275, 0.0021, 1.5623E-5, 0.00795, 0.00245, 0.6022, 0.0779, 0.0073, 0.5098, 0.0159, 0.00135, 1.5623E-5, 0.03385, 0.00705,
                                                0.02625, 0.0171, 0.71495, 0.06515, 0.01835, 0.00775, 0.00115, 0.03695, 0.051, 0.1615, 0.03565, 0.0205, 0.00275, 0.5098, 0.00775, 0.0055, 0.0026, 0.0013, 0.2085, 0.0183, 0.05085, 0.0173, 0.04775,
                                                0.00135, 0.06515, 0.0026, 0.35855, 0.1236, 0.00895, 0.0108, 0.65985, 0.2099, 0.03615, 0.0159, 0.01835, 0.0055, 0.35855});
//    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
//    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
//    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::barnes_symmetrized op;
    auto result = op.execute({&rows, &cols, &vals}, {}, {11});
    ASSERT_EQ(result->status(), Status::OK());
    auto res = result->at(2);
  //  res->printBuffer("Symmetrized4");
  //  exp4.printBuffer("Expected sym");
  //  nd4j_printf("Total res is {1, %lld}\n", res->lengthOf());
  //  nd4j_printf("Expected is {1, %lld}\n", exp4.lengthOf());

    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    ASSERT_TRUE(exp4.equalsTo(res));
    delete result;
}

TEST_F(DeclarableOpsTests13, CellContains_test_1) {

    auto corners = NDArrayFactory::create<double>( {0.5384,    0.5640,    0.3449,    0.5257,    0.5505});
    auto width = NDArrayFactory::create<double>({0.4306,    0.3960,    0.4639,    0.5040,    0.4904});
    auto point = NDArrayFactory::create<double>({0.3000,    0.2625,    0.2674,    0.8604,    0.4803});
    //auto exp = NDArrayFactory::create<double>('c', {1, 39}, {15.000000, 0.000000, 0.000000, 65.000000, 60.000000, 145.000000, 20.000000, 25.000000, 65.000000, 145.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000});
    //    data.linspace(1);

    //    auto y = NDArrayFactory::create<double>('c', {2,3}, {-0.1,-2,3, -4, -0.5, -6});
    //    auto eps = NDArrayFactory::create<double>('c', {2,3}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6});
    //    auto exp = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 1, 2, 2, 2});
    nd4j::ops::cell_contains op;
    auto result = op.execute({&corners, &width, &point}, {}, {5});
    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(result->at(0)->e<bool>(0));
    //result->at(2)->printBuffer("Symmetrized3");
    //exp.printBuffer("EXPect symm3");
    //    ASSERT_TRUE(exp[i]->equalsTo(result->at(i)));
    //ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_1) {

    NDArray input('c', {2,2,3}, {0,100,56, 17,220,5,  150,97,230, 255,2,13}, nd4j::DataType::FLOAT32);
    NDArray factor = NDArrayFactory::create<float>(0.5);
    NDArray exp  ('c', {2,2,3}, {100,0,44, 208,5,220, 177,230,97,  2,255,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    std::unique_ptr<nd4j::ResultSet> results (op.execute({&input, &factor}, {}, {2}));

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_2) {

    NDArray input('c', { 2,2,3 }, { 0.f,100.f / 255.f,56.f / 255.f, 17.f / 255.f,220.f / 255.f,5.f / 255.f,  150.f / 255.f,97.f / 255.f,230.f / 255.f, 255.f / 255.f,2.f / 255.f,13.f / 255.f }, nd4j::DataType::FLOAT32);
    NDArray exp('c', { 2,2,3 }, { 4.f / 255.f,100.f / 255.f,0.f,  146.f / 255.f,220.f / 255.f,5.f / 255.f, 97.f / 255.f,123.8f / 255.f,230.f / 255.f, 255.f / 255.f,2.f / 255.f,164.8f / 255.f }, nd4j::DataType::FLOAT32);


    nd4j::ops::adjust_hue op;
    std::unique_ptr<nd4j::ResultSet> results(op.execute({&input}, {0.9}, {2}));

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_3) {

    NDArray input('c', {2,2,3}, {0,100,56,    17,220,5,          150,97,230,     255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {0.,84.,100., 5.,220.,122.0001,  229.8,97.,230., 255.,142.8002,2.}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    std::unique_ptr<nd4j::ResultSet> results(op.execute({&input}, {-0.9}, {2}));

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_4) {

    NDArray input('c', {2,3,2}, {0,17,   100,220, 56,5,   150,255, 97,2,   230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,3,2}, {100,208, 0,5,   44,220,  177,2,   230,255, 97,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    std::unique_ptr<nd4j::ResultSet> results(op.execute({&input}, {0.5}, {1}));

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustHue_5) {

    NDArray input('c', {3,2,2}, {0,17, 150,255,   100,220, 97,2,  56,5, 230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {3,2,2}, {100,208, 177,2,  0,5, 230,255,   44,220, 97,244}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_hue op;
    std::unique_ptr<nd4j::ResultSet> results(op.execute({&input}, {0.5}, {0}));

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_1) {

    NDArray input('c', {2,2,3}, {0,100,56,  17,220,5,         150,97,230,    255,2,13}, nd4j::DataType::FLOAT32);
    NDArray factor = NDArrayFactory::create<float>(0.5);
    NDArray exp  ('c', {2,2,3}, {50,100,78, 118.5,220,112.5,  190,163.5,230, 255,128.5,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input, &factor}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_2) {

    NDArray input('c', {2,2,3}, {0,100,56,    17,220,5,          150,97,230,        255,2,13}, nd4j::DataType::DOUBLE);
    NDArray exp  ('c', {2,2,3}, {0.,100.,56., 12.279087,220.,0., 91.654228,0.,230., 255.,0.,11.087015}, nd4j::DataType::DOUBLE);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {10}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printIndexedBuffer("Result2");
//    exp.printIndexedBuffer("Expect2");

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_3) {

    NDArray input('c', {2,2,3}, {0,100,56,       17,220,5,       150,97,230,     255,2,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,2,3}, {100.,100.,100., 220.,220.,220., 230.,230.,230., 255., 255., 255.}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {-10}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_4) {

    NDArray input('c', {2,3,2}, {0,17,   100,220,  56,5,   150,255,  97,2,   230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {2,3,2}, {50,118.5, 100,220, 78,112.5,  190,255, 163.5,128.5, 230,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {0.5}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, adjustSaturation_5) {

    NDArray input('c', {3,2,2}, {0,17,     150,255,  100,220,  97,2,        56,5,     230,13}, nd4j::DataType::FLOAT32);
    NDArray exp  ('c', {3,2,2}, {50,118.5, 190,255,  100,220,  163.5,128.5, 78,112.5, 230,134}, nd4j::DataType::FLOAT32);

    nd4j::ops::adjust_saturation op;
    auto results = op.execute({&input}, {0.5}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}


TEST_F(DeclarableOpsTests13, shift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(32);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, rshift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(512);
    e.assign(32);

    nd4j::ops::rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_shift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(32);
    e.assign(512);

    nd4j::ops::cyclic_shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_rshift_bits_1) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>(4);
    auto e = x.ulike();
    x.assign(512);
    e.assign(32);

    nd4j::ops::cyclic_rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, shift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, rshift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(512);
    y.assign(4);
    e.assign(32);

    nd4j::ops::rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_shift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::cyclic_shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, cyclic_rshift_bits_2) {
    auto x = NDArrayFactory::create<int>('c', {5});
    auto y = NDArrayFactory::create<int>('c', {5});
    auto e = x.ulike();
    x.assign(512);
    y.assign(4);
    e.assign(32);

    nd4j::ops::cyclic_rshift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}
TEST_F(DeclarableOpsTests13, shift_bits_3) {
    auto x = NDArrayFactory::create<int>('c', {5, 5});
    auto y = NDArrayFactory::create<int>('c', {1, 5});
    auto e = x.ulike();
    x.assign(32);
    y.assign(4);
    e.assign(512);

    nd4j::ops::shift_bits op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_1) {

    NDArray x('c', {1, 2, 2, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 2} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {0, 0, 0, 0, 0, 0} , nd4j::DataType::INT32);

    NDArray exp('c', {8, 1, 1, 1, 3}, nd4j::DataType::FLOAT32);

    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_2) {

    NDArray x('c', {2,  2,4,3,  1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {0,0,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {24, 1,3,2, 1}, { 0, 2, 0, 8, 0, 0, 0, 26, 0, 32, 0, 0, 0, 3, 0, 9, 0, 0, 0, 27, 0, 33, 0, 0, 1,
                                        0, 7, 0, 0, 0, 25, 0, 31, 0, 0, 0, 0, 5, 0, 11, 0, 0, 0, 29, 0, 35, 0, 0, 0, 6,
                                        0, 12, 0, 0, 0, 30, 0, 36, 0, 0, 4, 0, 10, 0, 0, 0, 28, 0, 34, 0, 0, 0, 0, 14,
                                        0, 20, 0, 0, 0, 38, 0, 44, 0, 0, 0, 15, 0, 21, 0, 0, 0, 39, 0, 45, 0, 0, 13, 0,
                                        19, 0, 0, 0, 37, 0, 43, 0, 0, 0, 0, 17, 0, 23, 0, 0, 0, 41, 0, 47, 0, 0, 0, 18,
                                        0, 24, 0, 0, 0, 42, 0, 48, 0, 0, 16, 0, 22, 0, 0, 0, 40, 0, 46, 0, 0, 0}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, space_to_batch_nd_3) {

    NDArray x('c', {2,  2,4,3,  1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray paddings('c', {3, 2}, {1,1,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {24, 2,3,2, 1}, { 0, 0, 0, 0, 0, 0, 0, 14, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15,
                                        0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 19, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 37, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 47, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 18, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0,
                                        22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 46, 0, 0, 0, 0, 2, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 32,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                        0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 11, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 29, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 36, 0, 0,
                                        0, 0, 0, 0, 0, 0, 4, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::space_to_batch_nd op;
    auto result = op.execute({&x, &blockShape, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_1) {

    NDArray x('c', {8, 1, 1, 1, 3}, nd4j::DataType::FLOAT32);

    NDArray blockShape('c', {3}, {2, 2, 2} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {0, 0, 0, 0, 0, 0} , nd4j::DataType::INT32);

    NDArray exp('c', {1, 2, 2, 2, 3}, nd4j::DataType::FLOAT32);

    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_2) {

    NDArray x('c', {24, 1,3,2, 1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {0,0,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {2,  2,4,3,  1}, {25, 2, 14, 61, 38, 50, 27, 4, 16, 63, 40, 52, 97, 74, 86, 133, 110, 122, 99, 76, 88, 135, 112, 124,
                                      31, 8, 20, 67, 44, 56, 33, 10, 22, 69, 46, 58, 103, 80, 92, 139, 116, 128, 105, 82, 94, 141, 118, 130}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, batch_to_space_nd_3) {

    NDArray x('c', {24, 2,3,2, 1}, nd4j::DataType::FLOAT32);
    NDArray blockShape('c', {3}, {2, 2, 3} , nd4j::DataType::INT32);    // three spatial dimensions
    NDArray crop('c', {3, 2}, {1,1,  0,2,  2,1} , nd4j::DataType::INT32);

    NDArray exp('c', {2,  2,4,3,  1}, {193, 146, 170, 265, 218, 242, 195, 148, 172, 267, 220, 244, 55, 8, 32, 127, 80, 104, 57, 10, 34, 129, 82,
                                    106, 205, 158, 182, 277, 230, 254, 207, 160, 184, 279, 232, 256, 67, 20, 44, 139, 92, 116, 69, 22, 46, 141, 94, 118}, nd4j::DataType::FLOAT32);
    x.linspace(1);

    nd4j::ops::batch_to_space_nd op;
    auto result = op.execute({&x, &blockShape, &crop}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, mergemax_1) {

    NDArray x1('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray x3('c', {5, 5}, nd4j::DataType::FLOAT32);
    NDArray e('c', {5, 5}, nd4j::DataType::FLOAT32);
    x1.assign(3);
    x2.assign(1);
    x3.assign(2);
    e.assign(3);


    nd4j::ops::mergemax op;
    auto result = op.execute({&x1, &x2, &x3}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, mergemax_2) {

    NDArray x1('c', {1, 3}, {0., 1, 2}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {1, 1}, {1.}, nd4j::DataType::FLOAT32);
    NDArray out('c', {1, 3}, {-1., -1, -1}, nd4j::DataType::FLOAT32);

    nd4j::ops::mergemax op;
    auto status = op.execute({&x1, &x2}, {&out}, {}, {}, {});

    ASSERT_EQ(20, status);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_1) {

    const int sL   = 5;
    const int bS   = 3;
    const int nIn  = 3;
    const int nOut = 3;

    // input arguments

    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    auto expH = NDArrayFactory::create<float>('c', {sL, bS, nOut}, {0.57574f, 0.57574f, 0.57574f, 0.58006f, 0.58006f, 0.58006f, 0.58434f, 0.58434f, 0.58434f,
                                                           0.55114f, 0.55114f, 0.55114f, 0.55732f, 0.55732f, 0.55732f, 0.56338f, 0.56338f, 0.56338f,
                                                           0.53763f, 0.53763f, 0.53763f, 0.54534f, 0.54534f, 0.54534f, 0.55287f, 0.55287f, 0.55287f,
                                                           0.53626f, 0.53626f, 0.53626f, 0.54487f, 0.54487f, 0.54487f, 0.55327f, 0.55327f, 0.55327f,
                                                           0.54484f, 0.54484f, 0.54484f, 0.55379f, 0.55379f, 0.55379f, 0.5625f, 0.5625f, 0.5625f});

    auto expClast = NDArrayFactory::create<float>('c', {bS, nOut}, {1.1589154f, 1.1589154f, 1.1589154f, 1.1892855f, 1.1892855f, 1.1892855f, 1.219861f, 1.219861f, 1.219861f});

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *h  = results->at(0);
    auto *cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expClast.isSameShape(cL));
    ASSERT_TRUE(expClast.equalsTo(cL));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_2) {

    const int sL   = 5;
    const int bS   = 3;
    const int nIn  = 3;
    const int nOut = 3;

    // input arguments

    const int dataFormat = 1;       // [bS,sL,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {bS, sL, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    auto expH = NDArrayFactory::create<float>('c', {bS, sL, nOut}, {0.575735f, 0.575735f, 0.575735f, 0.541562f, 0.541562f, 0.541562f, 0.514003f, 0.514003f, 0.514003f, 0.495597f, 0.495597f, 0.495597f, 0.485999f, 0.485999f, 0.485999f,
                                            0.596965f, 0.596965f, 0.596965f, 0.571978f, 0.571978f, 0.571978f, 0.552888f, 0.552888f, 0.552888f, 0.540606f, 0.540606f, 0.540606f, 0.534764f, 0.534764f, 0.534764f,
                                            0.61725f, 0.61725f, 0.61725f, 0.599828f, 0.599828f, 0.599828f, 0.587627f, 0.587627f, 0.587627f, 0.580408f, 0.580408f, 0.580408f, 0.577735f, 0.577735f, 0.577735f});

    auto expClast = NDArrayFactory::create<float>('c', {bS, nOut}, {0.996965f, 0.996965f, 0.996965f, 1.146756f, 1.146756f, 1.146756f, 1.301922f, 1.301922f, 1.301922f});

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *h  = results->at(0);
    auto *cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expClast.isSameShape(cL));
    ASSERT_TRUE(expClast.equalsTo(cL));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_3) {

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments

    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 1;    // backward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL,bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {0.493883f, 0.493883f, 0.493883f, 0.510990f, 0.510990f, 0.510990f, 0.534701f, 0.534701f, 0.534701f, 0.549139f,
                                        0.549139f, 0.549139f, 0.571900f, 0.571900f, 0.571900f, 0.583561f, 0.583561f, 0.583561f, 0.605106f, 0.605106f,
                                        0.605106f, 0.614114f, 0.614114f, 0.614114f, 0.635354f, 0.635354f, 0.635354f, 0.642045f, 0.642045f, 0.642045f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {bS, nOut}, {0.493883f, 0.493883f, 0.493883f, 0.510990f, 0.510990f, 0.510990f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {bS, nOut}, {1.061274f, 1.061274f, 1.061274f, 1.115888f, 1.115888f, 1.115888f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_4) {

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 3;    // bidirectional concat
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;   // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {2,nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {2,nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {2,4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx({0,1, 0,0, 0,0}) = 0.003f;
    Wx({1,2, 0,0, 0,0}) = -0.003f;
    Wr({0,1, 0,0, 0,0}) = 0.006f;
    Wr({1,2, 0,0, 0,0}) = -0.006f;
    b({0,1, 0,0})       = 0.5f;
    b({1,2, 0,0})       = -0.5f;
    hI({0,1, 0,0, 0,0}) = 1;
    hI({1,2, 0,0, 0,0}) = -1;
    cI({0,1, 0,0, 0,0}) = 2;
    cI({1,2, 0,0, 0,0}) = -2;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, 2 * nOut}, {
                                         0.577661f,  0.577661f,  0.577661f, -0.107642f, -0.107642f, -0.107642f,  0.585289f,  0.585289f,  0.585289f,
                                        -0.106937f, -0.106937f, -0.106937f,  0.556517f,  0.556517f,  0.556517f, -0.111647f, -0.111647f, -0.111647f,
                                         0.567274f,  0.567274f,  0.567274f, -0.110214f, -0.110214f, -0.110214f,  0.547395f,  0.547395f,  0.547395f,
                                        -0.123305f, -0.123305f, -0.123305f,  0.560640f,  0.560640f,  0.560640f, -0.120862f, -0.120862f, -0.120862f,
                                         0.550714f,  0.550714f,  0.550714f, -0.156223f, -0.156223f, -0.156223f,  0.565308f,  0.565308f,  0.565308f,
                                        -0.152313f, -0.152313f, -0.152313f,  0.563741f,  0.563741f,  0.563741f, -0.234128f, -0.234128f, -0.234128f,
                                         0.578676f,  0.578676f,  0.578676f, -0.228917f, -0.228917f, -0.228917f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {2,bS, nOut}, {0.563741f, 0.563741f, 0.563741f, 0.578676f, 0.578676f, 0.578676f, -0.107642f,
                                    -0.107642f, -0.107642f, -0.106937f, -0.106937f, -0.106937f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {2,bS, nOut}, {1.217757f, 1.217757f, 1.217757f, 1.272398f, 1.272398f, 1.272398f, -0.295768f,
                                    -0.295768f, -0.295768f, -0.298453f, -0.298453f, -0.298453f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_5) {

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 1;       // [bS,sL,nIn]
    const int directionMode = 3;    // bidirectional concat
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;   // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {bS, sL, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {2,nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {2,nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {2,4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx({0,1, 0,0, 0,0}) = 0.003;
    Wx({1,2, 0,0, 0,0}) = -0.003;
    Wr({0,1, 0,0, 0,0}) = 0.006;
    Wr({1,2, 0,0, 0,0}) = -0.006;
    b({0,1, 0,0})       = 0.5;
    b({1,2, 0,0})       = -0.5;
    hI({0,1, 0,0, 0,0}) = 1;
    hI({1,2, 0,0, 0,0}) = -1;
    cI({0,1, 0,0, 0,0}) = 2;
    cI({1,2, 0,0, 0,0}) = -2;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {bS, sL, 2*nOut}, {
        0.577661f,  0.577661f,  0.577661f, -0.107659f, -0.107659f, -0.107659f, 0.548099f,  0.548099f,  0.548099f, -0.113406f, -0.113406f, -0.113406f,
        0.526881f,  0.526881f,  0.526881f,  -0.12883f,  -0.12883f,  -0.12883f, 0.515882f,  0.515882f,  0.515882f,  -0.16868f,  -0.16868f,  -0.16868f,
         0.51409f,   0.51409f,   0.51409f, -0.255185f, -0.255185f, -0.255185f, 0.614599f,  0.614599f,  0.614599f, -0.102739f, -0.102739f, -0.102739f,
        0.599572f,  0.599572f,  0.599572f, -0.105802f, -0.105802f, -0.105802f, 0.591089f,  0.591089f,  0.591089f, -0.116681f, -0.116681f, -0.116681f,
        0.588694f,  0.588694f,  0.588694f, -0.149201f, -0.149201f, -0.149201f, 0.591492f,  0.591492f,  0.591492f, -0.228917f, -0.228917f, -0.228917f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {2,bS, nOut}, {0.51409f,  0.51409f,  0.51409f,   0.591492f,  0.591492f,  0.591492f,
                                     -0.107659f, -0.107659f, -0.107659f,  -0.102739f, -0.102739f, -0.102739f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {2,bS, nOut}, {1.07293f ,  1.07293f ,  1.07293f, 1.346609f,  1.346609f,  1.346609f,
                                    -0.295811f, -0.295811f, -0.295811f, -0.305394f, -0.305394f, -0.305394f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    // h->printBuffer();
    // hL->printBuffer();
    // cL->printBuffer();

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_6) {

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 2;    // bidirectional sum
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = false;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;   // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {2,nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {2,nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {2,4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx({0,1, 0,0, 0,0}) = 0.003f;
    Wx({1,2, 0,0, 0,0}) = -0.003f;
    Wr({0,1, 0,0, 0,0}) = 0.006f;
    Wr({1,2, 0,0, 0,0}) = -0.006f;
    b({0,1, 0,0})       = 0.5f;
    b({1,2, 0,0})       = -0.5f;
    hI({0,1, 0,0, 0,0}) = 1;
    hI({1,2, 0,0, 0,0}) = -1;
    cI({0,1, 0,0, 0,0}) = 2;
    cI({1,2, 0,0, 0,0}) = -2;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {
        0.470019f, 0.470019f, 0.470019f, 0.478352f, 0.478352f, 0.478352f, 0.444871f, 0.444871f, 0.444871f, 0.457060f,
        0.457060f, 0.457060f, 0.424090f, 0.424090f, 0.424090f, 0.439778f, 0.439778f, 0.439778f, 0.394491f, 0.394491f,
        0.394491f, 0.412995f, 0.412995f, 0.412995f, 0.329613f, 0.329613f, 0.329613f, 0.349760f, 0.349760f, 0.349760f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {2,bS, nOut}, {0.563741f, 0.563741f, 0.563741f, 0.578676f, 0.578676f, 0.578676f,
                                      -0.107642f, -0.107642f, -0.107642f, -0.106937f, -0.106937f, -0.106937f},
                                      nd4j::DataType::FLOAT32);
    NDArray expCL('c', {2,bS, nOut}, {1.217757f, 1.217757f, 1.217757f, 1.272398f, 1.272398f, 1.272398f,
                                      -0.295768f, -0.295768f, -0.295768f, -0.298453f, -0.298453f, -0.298453f},
                                      nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_7) {
    #ifndef HAVE_MKLDNN

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments

    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;
    Wp = -0.05;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {0.55533 , 0.55533 , 0.55533 , 0.562925, 0.562925, 0.562925, 0.531795, 0.531795, 0.531795, 0.542556,
                                        0.542556, 0.542556, 0.521466, 0.521466, 0.521466, 0.534638, 0.534638, 0.534638, 0.524805, 0.524805,
                                        0.524805, 0.539187, 0.539187, 0.539187, 0.538309, 0.538309, 0.538309, 0.552923, 0.552923, 0.552923}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {bS, nOut}, {0.538309, 0.538309, 0.538309,0.552923, 0.552923, 0.552923}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {bS, nOut}, {1.147089, 1.147089, 1.147089,1.197228, 1.197228, 1.197228}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_8) {
    #ifndef HAVE_MKLDNN

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments

    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 1;    // backward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 1.;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;
    Wp = -0.05;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {
        0.436221f, 0.436221f, 0.436221f, 0.450573f, 0.450573f, 0.450573f, 0.463602f, 0.463602f, 0.463602f, 0.474674f, 0.474674f, 0.474674f,
        0.484039f, 0.484039f, 0.484039f, 0.490679f, 0.490679f, 0.490679f, 0.494871f, 0.494871f, 0.494871f, 0.499028f, 0.499028f, 0.499028f,
        0.504649f, 0.504649f, 0.504649f, 0.508719f, 0.508719f, 0.508719f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {bS, nOut}, {0.436221f, 0.436221f, 0.436221f, 0.450573f, 0.450573f, 0.450573f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {bS, nOut}, {0.879804f, 0.879804f, 0.879804f, 0.914666f, 0.914666f, 0.914666f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_9) {
    #ifndef HAVE_MKLDNN

    const int sL   = 5;
    const int bS   = 2;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 3;    // bidirectional concat
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = false;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;   // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {2,nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {2,nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {2,4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {2,3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx({0,1, 0,0, 0,0}) = 0.003;
    Wx({1,2, 0,0, 0,0}) = -0.003;
    Wr({0,1, 0,0, 0,0}) = 0.006;
    Wr({1,2, 0,0, 0,0}) = -0.006;
    b({0,1, 0,0})       = 0.5;
    b({1,2, 0,0})       = -0.5;
    hI({0,1, 0,0, 0,0}) = 1;
    hI({1,2, 0,0, 0,0}) = -1;
    cI({0,1, 0,0, 0,0}) = 2;
    cI({1,2, 0,0, 0,0}) = -2;
    Wp({0,1, 0,0}) = -0.05;
    Wp({1,2, 0,0}) = 0.05;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, 2*nOut}, {
         0.55533f,   0.55533f,   0.55533f, -0.104502f, -0.104502f, -0.104502f, 0.562925f,  0.562925f,  0.562925f, -0.103843f, -0.103843f, -0.103843f,
        0.531795f,  0.531795f,  0.531795f, -0.107456f, -0.107456f, -0.107456f, 0.542556f,  0.542556f,  0.542556f, -0.106139f, -0.106139f, -0.106139f,
        0.521466f,  0.521466f,  0.521466f,  -0.11681f,  -0.11681f,  -0.11681f, 0.534638f,  0.534638f,  0.534638f,  -0.11458f,  -0.11458f,  -0.11458f,
        0.524805f,  0.524805f,  0.524805f, -0.145177f, -0.145177f, -0.145177f, 0.539187f,  0.539187f,  0.539187f,  -0.14157f,  -0.14157f,  -0.14157f,
        0.538309f,  0.538309f,  0.538309f, -0.218056f, -0.218056f, -0.218056f, 0.552923f,  0.552923f,  0.552923f, -0.213068f, -0.213068f, -0.213068f}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {2,bS, nOut}, {0.538309f,  0.538309f,  0.538309f, 0.552923f,  0.552923f,  0.552923f, -0.104502f, -0.104502f, -0.104502f,
                                     -0.103843f, -0.103843f, -0.103843f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {2,bS, nOut}, {1.147089f,  1.147089f,  1.147089f, 1.197228f,  1.197228f,  1.197228f, -0.289425f, -0.289425f, -0.289425f,
                                     -0.292174f, -0.292174f, -0.292174f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_10) {
    #ifndef HAVE_MKLDNN

    const int sL   = 6;
    const int bS   = 5;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 0;    // forward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;   // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray seqLen('c', {bS}, {0,1,2,3,5}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003;
    Wr = 0.006;
    b = 0.5;
    hI = 1.;
    cI = 2.;
    Wp = -0.05;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {
              0.f,       0.f,       0.f, 0.562925f, 0.562925f, 0.562925f, 0.570404f, 0.570404f, 0.570404f,  0.57777f,
         0.57777f,  0.57777f, 0.585023f, 0.585023f, 0.585023f,       0.f,       0.f,       0.f,       0.f,       0.f,
              0.f, 0.576568f, 0.576568f, 0.576568f, 0.586163f, 0.586163f, 0.586163f, 0.595462f, 0.595462f, 0.595462f,
              0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f, 0.611224f,
        0.611224f, 0.611224f, 0.621298f, 0.621298f, 0.621298f,       0.f,       0.f,       0.f,       0.f,       0.f,
              0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f, 0.655858f, 0.655858f, 0.655858f,
              0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,        0.f,
              0.f,       0.f, 0.692315f, 0.692315f, 0.692315f,       0.f,       0.f,       0.f,       0.f,        0.f,
              0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f,        0.f},
        nd4j::DataType::FLOAT32);

    NDArray expHL('c', {bS, nOut}, {0.f, 0.f, 0.f, 0.562925f, 0.562925f, 0.562925f, 0.576568f, 0.576568f, 0.576568f, 0.611224f, 0.611224f, 0.611224f, 0.692315f, 0.692315f, 0.692315f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {bS, nOut}, {0.f, 0.f, 0.f, 1.534275f, 1.534275f, 1.534275f,  1.40183f,  1.40183f,  1.40183f, 1.449675f, 1.449675f, 1.449675f, 1.767702f, 1.767702f, 1.767702f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_11) {
    #ifndef HAVE_MKLDNN

    const int sL   = 6;
    const int bS   = 5;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 1;    // backward
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;   // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;   // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;  // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray seqLen('c', {bS}, {0,1,2,3,5}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx = 0.003f;
    Wr = 0.006f;
    b = 0.5f;
    hI = 1.f;
    cI = 2.f;
    Wp = -0.05f;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, nOut}, {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.61209f,
        0.61209f, 0.61209f,0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.652042f, 0.652042f, 0.652042f,  0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.677708f, 0.677708f, 0.677708f, 0.684177f, 0.684177f, 0.684177f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.699627f, 0.699627f,
        0.699627f, 0.705371f, 0.705371f, 0.705371f, 0.710989f, 0.710989f, 0.710989f, 0., 0., 0., 0.719014, 0.719014, 0.719014, 0.724087,
        0.724087f, 0.724087f, 0.729084f, 0.729084f, 0.729084f, 0.734004f, 0.734004f, 0.734004f }, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {bS, nOut}, {0.f, 0.f, 0.f, 0.719014f, 0.719014f, 0.719014f, 0.699627f, 0.699627f, 0.699627f, 0.677708f, 0.677708f, 0.677708f,  0.61209f,  0.61209f,  0.61209f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {bS, nOut}, {0.f, 0.f, 0.f, 2.092814f, 2.092814f, 2.092814f,  2.08832f,  2.08832f,  2.08832f, 2.009851f, 2.009851f, 2.009851f, 1.646034f, 1.646034f, 1.646034f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests13, lstmLayer_12) {
    #ifndef HAVE_MKLDNN

    const int sL   = 6;
    const int bS   = 5;
    const int nIn  = 4;
    const int nOut = 3;

    // input arguments
    const int dataFormat = 0;       // [sL,bS,nIn]
    const int directionMode = 3;    // bidirectional concat
    const int gateAct = 2;          // sigmoid activation for input (i), forget (f) and output (o) gates
    const int cellAct = 0;          // tanh activation for cell state
    const int outAct = 0;           // tanh activation for output

    const bool hasBiases  = true;   // biases array is provided
    const bool hasSeqLen  = true;  // seqLen array is not provided
    const auto hasInitH   = true;   // initial output is provided
    const auto hasInitC   = true;   // initial cell state is provided
    const auto hasPH      = true;  // peephole connections are absent
    const auto retFullSeq = true;   // return whole h {h_0, h_1, ... , h_sL-1}, [sL,bS,nOut]
    const auto retLastH   = true;   // do not return output at last time step
    const auto retLastC   = true;   // return cells state at last time step

    const double cellClip = 0;       // do not apply clipping

    NDArray x('c', {sL, bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {2,nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {2,nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b('c', {2,4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {2,bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray seqLen('c', {bS}, {0,1,2,3,5}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {2,3*nOut}, nd4j::DataType::FLOAT32);

    x.linspace(0.5, 0.5);
    Wx({0,1, 0,0, 0,0}) = 0.003f;
    Wx({1,2, 0,0, 0,0}) = -0.003f;
    Wr({0,1, 0,0, 0,0}) = 0.006f;
    Wr({1,2, 0,0, 0,0}) = -0.006f;
    b({0,1, 0,0})       = 0.5f;
    b({1,2, 0,0})       = -0.5f;
    hI({0,1, 0,0, 0,0}) = 1;
    hI({1,2, 0,0, 0,0}) = -1;
    cI({0,1, 0,0, 0,0}) = 2;
    cI({1,2, 0,0, 0,0}) = -2;
    Wp({0,1, 0,0}) = -0.05f;
    Wp({1,2, 0,0}) = 0.05f;

    std::initializer_list<double>   tArgs = {cellClip};
    std::initializer_list<Nd4jLong> iArgs = {dataFormat, directionMode, gateAct, cellAct, outAct};
    std::initializer_list<bool>     bArgs = {hasBiases, hasSeqLen, hasInitH, hasInitC, hasPH, retFullSeq, retLastH, retLastC};

    NDArray expH('c', {sL, bS, 2*nOut}, {0., 0., 0., 0., 0., 0.,  0.562925, 0.562925, 0.562925, -0.25361 , -0.25361 , -0.25361 ,   0.570404, 0.570404, 0.570404, -0.157103,
                                        -0.157103, -0.157103, 0.57777 , 0.57777 , 0.57777 , -0.116502, -0.116502, -0.116502,0.585023, 0.585023, 0.585023, -0.100025,
                                        -0.100025, -0.100025, 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,   0.576568, 0.576568, 0.576568, -0.223072, -0.223072, -0.223072,
                                        0.586163, 0.586163, 0.586163, -0.135714, -0.135714, -0.135714,0.595462, 0.595462, 0.595462, -0.094438, -0.094438, -0.094438,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.611224, 0.611224, 0.611224, -0.193473, -0.193473, -0.193473,
                                        0.621298, 0.621298, 0.621298, -0.090626, -0.090626, -0.090626, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0.655858, 0.655858, 0.655858, -0.098015, -0.098015, -0.098015, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.692315, 0.692315, 0.692315, -0.143704, -0.143704, -0.143704, 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, nd4j::DataType::FLOAT32);

    NDArray expHL('c', {2,bS, nOut}, {0.f, 0.f, 0.f, 0.562925f, 0.562925f, 0.562925f,   0.576568f,  0.576568f,  0.576568f,  0.611224f,  0.611224f,  0.611224f,   0.692315f,  0.692315f,  0.692315f,
                                      0.f, 0.f, 0.f, -0.25361f, -0.25361f, -0.25361f,  -0.157103f, -0.157103f, -0.157103f, -0.116502f, -0.116502f, -0.116502f,  -0.100025f, -0.100025f, -0.100025f}, nd4j::DataType::FLOAT32);
    NDArray expCL('c', {2,bS, nOut}, {0.f, 0.f, 0.f, 1.534275f, 1.534275f, 1.534275f,    1.40183f,   1.40183f,   1.40183f,  1.449675f,  1.449675f,  1.449675f,   1.767702f,  1.767702f,  1.767702f,
                                      0.f, 0.f, 0.f, -0.86636f, -0.86636f, -0.86636f,  -0.470245f, -0.470245f, -0.470245f, -0.341856f, -0.341856f, -0.341856f,  -0.294986f, -0.294986f, -0.294986f}, nd4j::DataType::FLOAT32);

    nd4j::ops::lstmLayer op;
    auto results = op.execute({&x, &Wx, &Wr, &b, &seqLen, &hI, &cI, &Wp}, tArgs, iArgs, bArgs);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h  = results->at(0);
    auto hL = results->at(1);
    auto cL = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expHL.isSameShape(hL));
    ASSERT_TRUE(expHL.equalsTo(hL));

    ASSERT_TRUE(expCL.isSameShape(cL));
    ASSERT_TRUE(expCL.equalsTo(cL));

    delete results;
    #endif
}


