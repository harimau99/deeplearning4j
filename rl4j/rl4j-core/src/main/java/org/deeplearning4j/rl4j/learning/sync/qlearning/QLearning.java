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

package org.deeplearning4j.rl4j.learning.sync.qlearning;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.SyncLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 * <p>
 * Mother class for QLearning in the Discrete domain and
 * hopefully one day for the  Continuous domain.
 */
@Slf4j
public abstract class QLearning<O extends Encodable, A, AS extends ActionSpace<A>>
                extends SyncLearning<O, A, AS, IDQN> implements TargetQNetworkSource {

    // FIXME Changed for refac
    // @Getter
    // final private IExpReplay<A> expReplay;
    @Getter
    @Setter(AccessLevel.PROTECTED)
    protected IExpReplay<A> expReplay;

    protected abstract LegacyMDPWrapper<O, A, AS> getLegacyMDPWrapper();

    public QLearning(QLConfiguration conf) {
        this(conf, getSeededRandom(conf.getSeed()));
    }

    public QLearning(QLConfiguration conf, Random random) {
        expReplay = new ExpReplay<>(conf.getExpRepMaxSize(), conf.getBatchSize(), random);
    }

    private static Random getSeededRandom(Integer seed) {
        Random rnd = Nd4j.getRandom();
        if(seed != null) {
            rnd.setSeed(seed);
        }

        return rnd;
    }

    protected abstract EpsGreedy<O, A, AS> getEgPolicy();

    public abstract MDP<O, A, AS> getMdp();

    public abstract IDQN getQNetwork();

    public abstract IDQN getTargetQNetwork();

    protected abstract void setTargetQNetwork(IDQN dqn);

    protected void updateTargetNetwork() {
        log.info("Update target network");
        setTargetQNetwork(getQNetwork().clone());
    }

    public IDQN getNeuralNet() {
        return getQNetwork();
    }

    public abstract QLConfiguration getConfiguration();

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract QLStepReturn<Observation> trainStep(Observation obs);

    protected StatEntry trainEpoch() {
        InitMdp<Observation> initMdp = refacInitMdp();
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();
        int step = initMdp.getSteps();

        Double startQ = Double.NaN;
        double meanQ = 0;
        int numQ = 0;
        List<Double> scores = new ArrayList<>();
        while (step < getConfiguration().getMaxEpochStep() && !getMdp().isDone()) {

            if (getStepCounter() % getConfiguration().getTargetDqnUpdateFreq() == 0) {
                updateTargetNetwork();
            }

            QLStepReturn<Observation> stepR = trainStep(obs);

            if (!stepR.getMaxQ().isNaN()) {
                if (startQ.isNaN())
                    startQ = stepR.getMaxQ();
                numQ++;
                meanQ += stepR.getMaxQ();
            }

            if (stepR.getScore() != 0)
                scores.add(stepR.getScore());

            reward += stepR.getStepReply().getReward();
            obs = stepR.getStepReply().getObservation();
            incrementStep();
            step++;
        }

        meanQ /= (numQ + 0.001); //avoid div zero


        StatEntry statEntry = new QLStatEntry(getStepCounter(), getEpochCounter(), reward, step, scores,
                        getEgPolicy().getEpsilon(), startQ, meanQ);

        return statEntry;

    }

    private InitMdp<Observation> refacInitMdp() {
        LegacyMDPWrapper<O, A, AS> mdp = getLegacyMDPWrapper();
        IHistoryProcessor hp = getHistoryProcessor();

        Observation observation = mdp.reset();

        int step = 0;
        double reward = 0;

        boolean isHistoryProcessor = hp != null;

        int skipFrame = isHistoryProcessor ? hp.getConf().getSkipFrame() : 1;
        int requiredFrame = isHistoryProcessor ? skipFrame * (hp.getConf().getHistoryLength() - 1) : 0;

        while (step < requiredFrame && !mdp.isDone()) {

            A action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP

            StepReply<Observation> stepReply = mdp.step(action);
            reward += stepReply.getReward();
            observation = stepReply.getObservation();

            step++;

        }

        return new InitMdp(step, observation, reward);

    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStatEntry implements StatEntry {
        int stepCounter;
        int epochCounter;
        double reward;
        int episodeLength;
        List<Double> scores;
        float epsilon;
        double startQ;
        double meanQ;
    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class QLStepReturn<O> {
        Double maxQ;
        double score;
        StepReply<O> stepReply;

    }

    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    @JsonDeserialize(builder = QLConfiguration.QLConfigurationBuilder.class)
    public static class QLConfiguration implements LConfiguration {

        Integer seed;
        int maxEpochStep;
        int maxStep;
        int expRepMaxSize;
        int batchSize;
        int targetDqnUpdateFreq;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;
        boolean doubleDQN;

        @JsonPOJOBuilder(withPrefix = "")
        public static final class QLConfigurationBuilder {
        }
    }


}
