/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.nativeblas;

import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * This class is a opaque pointer to InteropDataBuffer, used for Java/C++ interop related to INDArray DataBuffer
 *
 * @author saudet
 */
public class OpaqueDataBuffer extends Pointer {
    public OpaqueDataBuffer(Pointer p) { super(p); }

    /**
     * This method allocates new InteropDataBuffer and returns pointer to it
     * @param numElements
     * @param dataType
     * @param allocateBoth
     * @return
     */
    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType, boolean allocateBoth) {
        // TODO: add OOM handling right here
        val buffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

        return buffer;
    }

    /**
     * This method expands buffer, and copies content to the new buffer
     *
     * PLEASE NOTE: if InteropDataBuffer doesn't own actual buffers - original pointers won't be released
     * @param numElements
     */
    public void expand(long numElements) {
        // TODO: add OOM handling right here
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbExpand(this, numElements);
    }

    /**
     * This method returns pointer to linear buffer, primary one.
     * @return
     */
    public Pointer primaryBuffer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(this);
    }

    /**
     * This method returns pointer to special buffer, device one, if any.
     * @return
     */
    public Pointer specialBuffer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbSpecialBuffer(this);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setPrimaryBuffer(Pointer ptr, long numElements) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(this, ptr, numElements);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setSpecialBuffer(Pointer ptr, long numElements) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetSpecialBuffer(this, ptr, numElements);
    }

    /**
     * This method creates a view out of this InteropDataBuffer
     *
     * @param bytesLength
     * @param bytesOffset
     * @return
     */
    public OpaqueDataBuffer createView(long bytesLength, long bytesOffset) {
        // TODO: add OOM handling right here
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbCreateView(this, bytesLength, bytesOffset);
    }
}
