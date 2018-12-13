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

#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NDArrayFactory.h"
#include "NativeOpExecutioner.h"
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform_same.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>
#include <MmulHelper.h>
#include <helpers/threshold.h>
#include <graph/exceptions/datatype_exception.h>

#include <NDArray.hpp>

namespace nd4j {

    void* NDArray::operator new(size_t i) {
        if (nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            nd4j::memory::Workspace* ws = nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace();

            return ws->allocateBytes((Nd4jLong) i);
        } else {
            auto p = malloc(i);
            
            CHECK_ALLOC(p, "Failed to allocate new NDArray");

            return p;
        }
    }

    void NDArray::operator delete(void* p) {
        if (!nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            free(p);
        }
    }


////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {

    _length = other._length;
    _workspace = other._workspace;
    _dataType = other._dataType;

    ALLOCATE(_buffer, other._workspace, _length * other.sizeOfT(), int8_t);
    _shapeInfo = ShapeBuilders::copyShapeInfo(other._shapeInfo, false, _workspace);

    _isBuffAlloc = true;
    _isShapeAlloc = true;

    this->assign(&other);
}

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, nd4j::memory::Workspace* workspace, const bool isBuffAlloc, const bool isShapeAlloc) {
    _buffer    = reinterpret_cast<int8_t *>(buffer);
    _shapeInfo = shapeInfo;
    _isBuffAlloc = isBuffAlloc;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = isShapeAlloc;
    _workspace = workspace;

    if (shapeInfo != nullptr) {
        _length = shape::length(shapeInfo);
        _dataType = ArrayOptions::dataType(shapeInfo);
    } else
        throw std::runtime_error("NDArray can't be initalized without shapeinfo");
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));
    ALLOCATE(_buffer, workspace, _length * DataTypeUtils::sizeOf(dtype), int8_t);
    memset(_buffer, 0, _length * DataTypeUtils::sizeOf(dtype));
    _workspace = workspace;
    triggerAllocationFlag(true, true);
}


//////////////////////////////////////////////////////////////////////////
// perform array transformation
    // void NDArray::applyTransform(nd4j::transform::FloatOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::AnyOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::SameOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::BoolOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::StrictOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // perform array transformation

/*
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyRandom(nd4j::random::RandomBuffer *buffer, NDArray<T>* y, NDArray<T>* z, T* extraArgs) {
        Nd4jPointer state = (Nd4jPointer) buffer;
        if (y == nullptr && z == nullptr) {
            // we're executing indexed z here
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), extraArgs);
        } else if (y == nullptr && z != nullptr) {
            // XZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        } else if (y != nullptr && z != nullptr) {
            // XYZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        }
    }
    */

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast bool method: target or other = nullptr !");
        
        if (isScalar()) {
            NDArray temp(target->_shapeInfo, _dataType, false, _workspace);
            temp.assign(this);
            temp.applyPairwiseTransform(op.p, other, target,  extraArgs);
            return;
        }
        if (other->isScalar()) {
            this->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = this;
            min = other;
        }
        else {
            max = other;
            min = this;
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsSoft(target->_shapeInfo, newShapeInfo) || target->_dataType != DataType::BOOL)
                throw std::runtime_error("NDArray::applyTrueBroadcast bool method: the shape or type of target array is wrong !");
            if(_dataType != other->_dataType)
                throw std::invalid_argument("NDArray::applyTrueBroadcast bool method: this and other arrays must have the same type !");

            // if workspace is not null - do not call delete.
            if (_workspace == nullptr)
                delete[] newShapeInfo;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_workspace);
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);

        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));


        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);

#pragma omp parallel for schedule(guided)
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                NDArray targetSubArr = (*target)(i, dimsToExclude);
                if (pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    NDArray pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
        if(pTarget != target)
            delete pTarget;
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, void *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast method: target or other = nullptr !");
        if(((op.s == scalar::Divide || op.s == scalar::FloorDiv || op.s == scalar::FloorMod) && other->isB()) || (op.s == scalar::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyTrueBroadcast method: you can't divide by bool array !");

        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op.p, *other, extraArgs);
            return;
        }
        if (other->isScalar()) {
            const_cast<NDArray*>(this)->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = this;
            min = other;
        }
        else {
            max = other;
            min = this;
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsTypesAndShapesSoft(target->getShapeInfo(), newShapeInfo))
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape or type of target array is wrong !");

            // if workspace is not null - do not call delete.
            if (_workspace == nullptr)
                delete[] newShapeInfo;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_workspace);
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);


        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);
        
#pragma omp parallel for schedule(guided)
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto targetSubArr = (*target)(i, dimsToExclude);
                if(pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    auto pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
         if(pTarget != target)
            delete pTarget;
    }

    //////////////////////////////////////////////////////////////////////////
    // return array which is broadcasted from this and argument array
    NDArray* NDArray::broadcast(const NDArray& other) {
	    // the orders must be the same
	    char order = ordering();
	    if(order != other.ordering())
		    throw std::runtime_error("Broadcast method: arrays have different orders!");

	    // recognize shapes with smaller and bigger rank
	    Nd4jLong* biggerShapeInfo = nullptr;
	    Nd4jLong* smallerShapeInfo = nullptr;
	    int smallerRank, biggerRank;
	    if (rankOf() > other.rankOf()) {
		    biggerShapeInfo = _shapeInfo;
		    biggerRank = shape::rank(_shapeInfo);
		    smallerShapeInfo = other._shapeInfo;
		    smallerRank = shape::rank(other._shapeInfo);
	    }
	    else {
		    biggerShapeInfo = other._shapeInfo;
		    biggerRank = shape::rank(other._shapeInfo);
		    smallerShapeInfo = _shapeInfo;
		    smallerRank = shape::rank(_shapeInfo);
	    }

	    // check shapes on consistency
	    int diff = biggerRank - smallerRank;
	    for (int i = smallerRank; i<=1; --i)
		    if(biggerShapeInfo[diff+i] != smallerShapeInfo[i] && biggerShapeInfo[i] != 1 && smallerShapeInfo[i] != 1)
			    throw std::runtime_error("Broadcast method: arrays have incompatible shapes !");

		// create and fill ret shapeInfo
	    auto shapeInfoNew = new Nd4jLong[shape::shapeInfoLength(biggerRank)];
	    memcpy(shapeInfoNew, biggerShapeInfo, shape::shapeInfoByteLength(biggerRank));
	    for (int i = smallerRank; i>=1; --i)
		    if(shapeInfoNew[diff+i] == 1 || smallerShapeInfo[i] == 1)
			    shapeInfoNew[diff+i] *= smallerShapeInfo[i];

	    auto ret = new NDArray(shapeInfoNew, true, _workspace);
        ShapeUtils::updateStridesAndType(ret->getShapeInfo(), DataTypeUtils::pickPairwiseResultType(_dataType, other._dataType), order);
	    delete []shapeInfoNew;

    	return ret;
    }


    //////////////////////////////////////////////////////////////////////////
    // check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
    bool NDArray::hasOrthonormalBasis(const int arg) {
        if (isS())
            throw std::runtime_error("NDArray::hasOrthonormalBasis: you can't use this method on String array!");
	    if(rankOf() !=2 )
		    throw std::runtime_error("NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !");

	    if(arg!=0  && arg!=1)
		    throw std::runtime_error("NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !");

	    const double eps = 1e-5;
        double dot = 0.f;

        if(arg) {					// check whether columns create orthogonal basis
		    for(int j=0; j<columns()-1; ++j)
			    for(int k=j+1; k<columns(); ++k) {
				    for(int i=0; i<rows(); ++i)
					    dot += e<double>(i,j)*e<double>(i,k);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.f;
			    }

			    for(int j=0; j<columns(); ++j)	{	// check whether norm of column vector = 1
			        for(int i=0; i<rows(); ++i)
				        dot += e<double>(i,j)*e<double>(i,j);
			    if(dot != 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.f) > eps)
				    return false;

			    dot = 0.f;
		    }
	    }
	    else {						// check whether rows create orthogonal basis
		    for(int i=0; i<rows()-1; ++i)
			    for(int k=i+1; k<rows(); ++k) {
				    for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(k,j);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.;
			    }

		        for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			        for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(i,j);

			        if(dot!= 0. && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.) > eps)
				        return false;
			        dot = 0.;
		        }
	        }
	    return true;
    }

    template <typename T>
    std::vector<T> NDArray::asVectorT() {
        std::vector<T> result(this->lengthOf());

#pragma omp parallel for simd
        for (int e = 0; e < this->lengthOf(); e++)
            result[e] = this->e<T>(e);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::asVectorT(), LIBND4J_TYPES);


    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
        if (isS())
            throw std::runtime_error("NDArray::setValueInDiagMatrix: you can't use this method on String array!");
        if(rankOf() != 2)
           throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");

        const auto rows = sizeAt(0);
        const auto cols = sizeAt(1);

        switch(direction) {

            case 'u':                           // fill upper triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
                for(Nd4jLong i = 0; i < rows; ++i)
                    for(Nd4jLong j = 0; j < cols; ++j)
                        if (i + diag <= j)
                            p<T>(i, j, value);
                break;

            case 'l':                           // fill lower triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse (2)
                for(Nd4jLong i = 0; i < rows; ++i)
                    for(Nd4jLong j = 0; j < cols; ++j)
                        if (i + diag >= j)
                            p<T>(i, j, value);
                break;
            default:
                throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
        }
    }
    template void NDArray::setValueInDiagMatrix(const double& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bfloat16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const Nd4jLong& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int16_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const uint8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bool& value, const int diag, const char direction);

    ////////////////////////////////////////////////////////////////////////
    // default destructor
    NDArray::~NDArray() noexcept {
        if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr) {
            if (!isS()) {
                delete[] _buffer;
            } else {
                for (int e = 0; e < lengthOf(); e++) {
                    auto t = reinterpret_cast<utf8string**>(_buffer);
                    delete t[e];
                };

                delete[] _buffer;
            }
        }

        if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr)
            delete[] _shapeInfo;
    }


    //////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length
    bool NDArray::reshapei(const char order, const std::vector<Nd4jLong>& cshape) {

        // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary
        if(order == ordering() && rankOf() == cshape.size()) {
            bool areShapesSame = true;
            for(int i = 0; i < cshape.size(); ++i)
                if(cshape[i] != sizeAt(i)) {
                    areShapesSame = false;
                    break;
                }
            if(areShapesSame)
                return areShapesSame;
        }

        std::vector<Nd4jLong> shape(cshape);
        int rank = shape.size();

        // looking for negative in shape

        int numberNegativesOnes = 0;

        Nd4jLong* shape_ = shape.data();
        for (int i = 0; i < (int) shape.size(); i++) {
            if (shape[i] < 0) {
                if (numberNegativesOnes >= 1)
                    throw std::runtime_error("Only one dimension can be negative at once");

                numberNegativesOnes++;

                int shapeLength = 1;
                for (int j = 0; j < (int) shape.size(); j++)
                    if (i != j)
                        shapeLength *= shape_[j];

                Nd4jLong realShape = nd4j::math::nd4j_abs<int>(lengthOf() / shapeLength);
                auto thisNewShape = new Nd4jLong[shape.size()];

                for (int j = 0; j < (int) shape.size(); j++)
                    if (i != j)
                        thisNewShape[j] = shape_[j];
                    else
                        thisNewShape[j] = realShape;

                shape_ = thisNewShape;
            }
        }

        for (int e = 0; e < (int) shape.size(); e++)
            shape[e] = shape_[e];

        if (numberNegativesOnes > 0)
            delete[] shape_;

        int arrLength = 1;
        for(const auto& item : shape)
            arrLength *= item;

        if(_buffer==nullptr || arrLength != this->lengthOf()) {
            this->printShapeInfo("Mismatched shape");
            nd4j::Logger::printv("Shape requested: ", shape);
            nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
            throw std::runtime_error("Bad shape!");
        }

        int shapeLength = shape::shapeInfoLength(rank);
        // remember old values

        // we can do this only if there was no permute applied, or there are no weird strides
        if (shape::canReshape(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f')) {
            Nd4jLong *shapeInfoNew;
            ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), Nd4jLong);

            shape::reshapeCF(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f', shapeInfoNew);

            if (_isShapeAlloc)
                RELEASE(_shapeInfo, _workspace);

            ArrayOptions::setDataType(shapeInfoNew, this->dataType());
            _shapeInfo = shapeInfoNew;
            _isShapeAlloc = true;
        } else {
            Nd4jLong *shapeInfoNew;
            ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), Nd4jLong);

            if (order == 'c')
                shape::shapeBuffer(shape.size(), dataType(), shape.data(), shapeInfoNew);
            else
                shape::shapeBufferFortran(shape.size(), dataType(), shape.data(), shapeInfoNew);

            int8_t *newBuffer;
            ALLOCATE(newBuffer, _workspace, this->lengthOf() * sizeOfT(), int8_t);

            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, _bufferD, _shapeInfoD, newBuffer, shapeInfoNew, nullptr, nullptr, nullptr, nullptr, nullptr);

            if (_isBuffAlloc)
                RELEASE(_buffer, _workspace);


            if (_isShapeAlloc)
                RELEASE(_shapeInfo, _workspace);

            _buffer = newBuffer;
            _shapeInfo = shapeInfoNew;
            _isShapeAlloc = true;
            _isBuffAlloc = true;
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::setIdentity() {
        if (isS())
            throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

        this->assign(0.);

        int  rank    = rankOf();
        auto shape   = shapeOf();
        auto strides = stridesOf();
        int  minDim  = 100000000;
        Nd4jLong indices[MAX_RANK];
        for(int j = 0; j < rank; ++j)
            indices[j] = 1;

        Nd4jLong offset = shape::getOffset(0, shape, strides, indices, rank);

        for(int i = 0; i < rank; ++i)
            if(minDim > shape[i])
                minDim = shape[i];

        float v = 1.0f;
#pragma omp parallel for if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i = 0; i < minDim; ++i)
            templatedSet<float>(_buffer, i*offset, this->dataType(), &v);
    }

    template <typename T>
    void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value), LIBND4J_TYPES);



    template <typename T>
    void NDArray::templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
        auto x = reinterpret_cast<T *>(xBuffer);
        auto y = reinterpret_cast<T *>(yBuffer);

#pragma omp parallel for simd schedule(static)
        for (int i = 0; i < length; ++i) {
            auto temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSwap, (void *xBuffer, void *yBuffer, Nd4jLong length), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void NDArray::swapUnsafe(NDArray& other) {
        auto xType = this->dataType();

        if (xType != other.dataType())
            throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

        if(_buffer == nullptr || other._buffer == nullptr)
            throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

        // if(_buffer == other._buffer)
        //     throw std::runtime_error("NDArray::swapUnsafe method: the buffers of input arrays should not point on the same address!");

        if(lengthOf() != other.lengthOf())
            throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

        BUILD_SINGLE_SELECTOR(xType, templatedSwap, (this->_buffer, other.buffer(), this->lengthOf()), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::diagonal(const char type) const {

        if (isS())
            throw std::runtime_error("NDArray::diagonal: you can't use this method on String array!");

        const char order = ordering();
        const int  rank  = rankOf();
        Nd4jLong *outShapeInfo;
        ALLOCATE(outShapeInfo, _workspace, 8, Nd4jLong);
        outShapeInfo[0] = 2;
        outShapeInfo[5] = 0;

        if(isVector() || isScalar()) {

            outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
            outShapeInfo[6] = 1;
            outShapeInfo[7] = (int)order;
        }
        else {

            int diagSize  = 100000000;
            Nd4jLong indices[MAX_RANK];

            for(int i = 0; i < rank; ++i) {
                if(diagSize > shapeOf()[i])
                    diagSize = shapeOf()[i];
                indices[i] = 1;
            }

            auto step = shape::getOffset(0, shapeOf(), stridesOf(), indices, rank);

            if(type == 'c') {
                outShapeInfo[1] = diagSize;
                outShapeInfo[2] = 1;
            }
            else {
                outShapeInfo[1] = 1;
                outShapeInfo[2] = diagSize;
            }
            shape::updateStrides(outShapeInfo, order);

            outShapeInfo[3] *= step;
            outShapeInfo[4] *= step;
            outShapeInfo[6] =  -1;
        }

        ArrayOptions::setDataType(outShapeInfo, this->dataType());

        auto result = new NDArray(this->_buffer, outShapeInfo, this->_workspace);
        result->_isShapeAlloc = true;
        return result;
    }

    void NDArray::streamline(char o) {
        char order = o == 'a' ? this->ordering() : o;

        Nd4jLong *newShape;
        ALLOCATE(newShape, this->_workspace, shape::shapeInfoLength(this->rankOf()), Nd4jLong);

        int8_t *newBuffer;
        ALLOCATE(newBuffer, this->_workspace, this->lengthOf() * sizeOfT(), int8_t);

        std::vector<Nd4jLong> shape(this->rankOf());
        for (int e = 0; e < this->rankOf(); e++)
            shape[e] = this->sizeAt(e);

        if (order == 'c')
            shape::shapeBuffer(this->rankOf(),dataType(),  shape.data(), newShape);
        else
            shape::shapeBufferFortran(this->rankOf(), dataType(), shape.data(), newShape);

        if (!isView()) {
            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, nullptr, nullptr, newBuffer, newShape, nullptr, nullptr, nullptr, nullptr, nullptr);
            memcpy(_buffer, newBuffer, this->lengthOf() * sizeOfT());

            //if (_isBuffAlloc)
            //    RELEASE(this->_buffer, this->_workspace);
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_workspace);

            //this->_buffer = newBuffer;
            //this->_isBuffAlloc = true;

            RELEASE(newBuffer, this->_workspace);

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        } else {
            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, nullptr, nullptr, newBuffer, newShape, nullptr, nullptr, nullptr, nullptr, nullptr);

            if (_isBuffAlloc)
                RELEASE(this->_buffer, this->_workspace);
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_workspace);

            this->_buffer = newBuffer;
            this->_isBuffAlloc = true;

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        }
    }

    //BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, Y value), LIBND4J_TYPES, LIBND4J_TYPES);
/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}

#endif

