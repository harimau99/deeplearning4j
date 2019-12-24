package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * FrameIter: Identifies a frame + iteration (but not a specific op or variable).<br>
 * Note that frames can be nested - which generally represents nested loop situations.
 */
@Data
@AllArgsConstructor
public class FrameIter {
    private String frame;
    private int iteration;
    private FrameIter parentFrame;

    @Override
    public String toString() {
        return "(\"" + frame + "\"," + iteration + (parentFrame == null ? "" : ",parent=" + parentFrame.toString()) + ")";
    }

    @Override
    public FrameIter clone() {
        return new FrameIter(frame, iteration, (parentFrame == null ? null : parentFrame.clone()));
    }

    public AbstractSession.VarId toVarId(String name) {
        return new AbstractSession.VarId(name, frame, iteration, parentFrame);
    }
}
