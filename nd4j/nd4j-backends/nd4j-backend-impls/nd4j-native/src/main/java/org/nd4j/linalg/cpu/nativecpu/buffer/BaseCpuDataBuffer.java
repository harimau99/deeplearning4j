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

package org.nd4j.linalg.cpu.nativecpu.buffer;

import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.deallocation.DeallocatorService;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.nio.ByteBuffer;

import static org.nd4j.linalg.api.buffer.DataType.INT16;
import static org.nd4j.linalg.api.buffer.DataType.INT8;

/**
 * Base implementation for DataBuffer for CPU-like backend
 *
 * @author raver119@gmail.com
 */
public abstract class BaseCpuDataBuffer extends BaseDataBuffer implements Deallocatable {

    private OpaqueDataBuffer ptrDataBuffer;

    private final long instanceId = Nd4j.getDeallocatorService().nextValue();

    protected BaseCpuDataBuffer() {

    }


    @Override
    public String getUniqueId() {
        return "BCDB_" + instanceId;
    }

    @Override
    public Deallocator deallocator() {
        return new CpuDeallocator(this);
    }

    public OpaqueDataBuffer getOpaqueDataBuffer() {
        return ptrDataBuffer;
    }

    @Override
    public int targetDevice() {
        // TODO: once we add NUMA support this might change. Or might not.
        return 0;
    }


    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseCpuDataBuffer(long length, int elementSize) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = (byte) elementSize;

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(length, dataType().toInt(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asDoublePointer();

            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asFloatPointer();

            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (dataType() == DataType.INT32) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asIntPointer();

            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asLongPointer();

            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asShortPointer();

            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asBytePointer();

            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            // FIXME: probably wrong
            ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(length, DataType.INT64.toInt(), false);
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asLongPointer();

            setIndexer(LongIndexer.create((LongPointer) pointer));
        }

        Nd4j.getDeallocatorService().pickObject(this);
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseCpuDataBuffer(int length, int elementSize, long offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }


    protected BaseCpuDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);

        // for vew we need "externally managed" pointer and deallocator registration
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().dbCreateView(((BaseCpuDataBuffer) underlyingBuffer).ptrDataBuffer, offset * underlyingBuffer.getElementSize());
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, new PagedPointer(this.addressPointer()), length);
        Nd4j.getDeallocatorService().pickObject(this);
    }

    @Override
    public void pointerIndexerByCurrentType(DataType currentType) {

        type = currentType;

        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(length(), type.toInt(), false);
        actualizePointerAndIndexer();
        Nd4j.getDeallocatorService().pickObject(this);
    }

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseCpuDataBuffer(long length) {
        this(length, true);
    }

    protected BaseCpuDataBuffer(long length, boolean initialize) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(length, dataType().toInt(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asDoublePointer();

            indexer = DoubleIndexer.create((DoublePointer) pointer);

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asFloatPointer();

            setIndexer(FloatIndexer.create((FloatPointer) pointer));

            if (initialize)
                fillPointerWithZero();

        } else if (dataType() == DataType.HALF) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asShortPointer();

            setIndexer(HalfIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BFLOAT16) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asShortPointer();

            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.INT) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asIntPointer();

            setIndexer(IntIndexer.create((IntPointer) pointer));
            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asLongPointer();

            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asShortPointer();

            setIndexer(ShortIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asBytePointer();

            setIndexer(UByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT16) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asShortPointer();

            setIndexer(UShortIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT32) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asIntPointer();

            // FIXME: we need unsigned indexer here
            setIndexer(IntIndexer.create((IntPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT64) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asLongPointer();

            // FIXME: we need unsigned indexer here
            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BOOL) {
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length).asBoolPointer();

            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UTF8) {
            // we are allocating buffer as INT8 intentionally
            ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(length(), INT8.toInt(), false);
            pointer = new PagedPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer), length()).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        }

        Nd4j.getDeallocatorService().pickObject(this);
    }

    public void actualizePointerAndIndexer() {
        val cptr = NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer);

        // skip update if pointers are equal
        if (cptr != null && pointer != null && cptr.address() == pointer.address())
            return;

        val t = dataType();
        if (t == DataType.BOOL) {
            pointer = new PagedPointer(cptr, length).asBoolPointer();
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));
        } else if (t == DataType.UBYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.BYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.UINT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(UShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.SHORT) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.UINT32) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.INT) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.UINT64) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.LONG) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.BFLOAT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));
        } else if (t == DataType.HALF) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(HalfIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.FLOAT) {
            pointer = new PagedPointer(cptr, length).asFloatPointer();
            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (t == DataType.DOUBLE) {
            pointer = new PagedPointer(cptr, length).asDoublePointer();
            setIndexer(DoubleIndexer.create((DoublePointer) pointer));
        } else if (t == DataType.UTF8) {
            pointer = new PagedPointer(cptr, length()).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else
            throw new IllegalArgumentException("Unknown datatype: " + dataType());
    }

    protected BaseCpuDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();



        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        // creating empty native DataBuffer
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(0, dataType().toInt(), false);

        if (dataType() == DataType.DOUBLE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asDoublePointer(); //new DoublePointer(length());
            indexer = DoubleIndexer.create((DoublePointer) pointer);

        } else if (dataType() == DataType.FLOAT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asFloatPointer(); //new FloatPointer(length());
            setIndexer(FloatIndexer.create((FloatPointer) pointer));

        } else if (dataType() == DataType.HALF) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(HalfIndexer.create((ShortPointer) pointer));

        } else if (dataType() == DataType.BFLOAT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.INT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));

        } else if (dataType() == DataType.UINT32) {
            attached = true;
            parentWorkspace = workspace;

            // FIXME: need unsigned indexer here
            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));

        } else if (dataType() == DataType.UINT64) {
            attached = true;
            parentWorkspace = workspace;

            // FIXME: need unsigned indexer here
            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new IntPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));

        } else if (dataType() == DataType.LONG) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UINT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new IntPointer(length());
            setIndexer(UShortIndexer.create((ShortPointer) pointer));

        } else if (dataType() == DataType.SHORT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new LongPointer(length());
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.BOOL) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBoolPointer(); //new LongPointer(length());
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));
        }

        // storing pointer into native DataBuffer
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, pointer, length * getElementSize());

        // adding deallocator reference
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
    }

    public BaseCpuDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);

        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(0, type.toInt(), false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, this.pointer, length * elementSize);
        Nd4j.getDeallocatorService().pickObject(this);;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(float[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;

    }

    public BaseCpuDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new FloatPointer(data);

        // creating & registering native DataBuffer
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(data.length, DataType.FLOAT.toInt(), false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, pointer, data.length * getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);

        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();

        length = data.length;
        underlyingLength = data.length;
    }

    public BaseCpuDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asFloatPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseCpuDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asDoublePointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = DoubleIndexer.create((DoublePointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    public BaseCpuDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asIntPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = IntIndexer.create((IntPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseCpuDataBuffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asLongPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = LongIndexer.create((LongPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(double[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    public BaseCpuDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(double[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new DoublePointer(data);
        indexer = DoubleIndexer.create((DoublePointer) pointer);

        // creating & registering native DataBuffer
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(data.length, DataType.DOUBLE.toInt(), false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, pointer, data.length * getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(int[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(int[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new IntPointer(data);
        setIndexer(IntIndexer.create((IntPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(data.length, DataType.INT32.toInt(), false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, pointer, data.length * getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(long[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new LongPointer(data);
        setIndexer(LongIndexer.create((LongPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(data.length, DataType.INT64.toInt(), false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, pointer, data.length * getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(double[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(int[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(float[] data) {
        this(data, true);
    }

    public BaseCpuDataBuffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }


}
