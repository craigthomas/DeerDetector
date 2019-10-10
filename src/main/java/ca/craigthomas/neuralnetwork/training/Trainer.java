/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.training;

import java.util.ArrayList;
import java.util.List;

import ca.craigthomas.neuralnetwork.activationfunctions.IActivationFunction;
import ca.craigthomas.neuralnetwork.components.NeuralNetwork;
import org.jblas.DoubleMatrix;

import ca.craigthomas.neuralnetwork.components.DataSet;

/**
 * The trainer class is responsible for training a neural network. Will
 * return the trained neural network.
 * 
 * @author thomas
 */
public class Trainer
{
    // The default number of iterations
    public static final int DEFAULT_MAX_ITERATIONS = 500;
    // The default number of iterations at which to display a heartbeat
    public static final int DEFAULT_HEARTBEAT = 100;
    // The default learning rate of the network
    public static final double DEFAULT_LEARNING_RATE = 0.01;
    
    private NeuralNetwork neuralNetwork;
    private double mLearningRate;
    private boolean mRecordCosts;
    private int mMaxIterations;
    private int mHeartBeat;
    private List<Double> mCosts;

    public static class Builder {
        
        private List<Integer> mLayerSizes;
        private double mLambda;
        private double mLearningRate;
        private DoubleMatrix mInputs;
        private DoubleMatrix mOutputs;
        private IActivationFunction mActivationFunction;
        private int mMaxIterations;
        private boolean mRecordCosts;
        private int mHeartBeat;
        
        /**
         * Initialize a builder object which will be used to build a neural
         * network. 
         * 
         * @param layerSizes the sizes of each of the network layers
         * @param inputs the inputs to use for training
         * @param outputs the outputs to use for training
         */
        public Builder(List<Integer> layerSizes, DoubleMatrix inputs, DoubleMatrix outputs) {
            mRecordCosts = false;
            mLambda = 0.0;
            mMaxIterations = DEFAULT_MAX_ITERATIONS;
            mHeartBeat = DEFAULT_HEARTBEAT;
            mLayerSizes = layerSizes;
            mInputs = inputs;
            mOutputs = outputs;
            mLearningRate = DEFAULT_LEARNING_RATE;
        }
        
        /**
         * Alternate constructor to build a neural network with a DataSet 
         * instead of inputs and outputs.
         * 
         * @param layerSizes the sizes of each of the network layers
         * @param trainingData the training data set to use
         */
        public Builder(List<Integer> layerSizes, DataSet trainingData) {
            this(layerSizes, trainingData.getSamples(), trainingData.getTruth());
        }
        
        /**
         * The regularization parameter to use. Defaults to 0 - no lambda. 
         * 
         * @param lambda the lambda to use for regularization
         * @return the builder for the trainer
         */
        public Builder lambda(double lambda) {
            mLambda = lambda;
            return this;
        }
        
        /**
         * Sets the learning rate at which the algorithm will adjust the
         * theta parameters for learning. Default 0.1.
         * 
         * @param learningRate the learning rate to use
         * @return the builder for the trainer
         */
        public Builder learningRate(double learningRate) {
            mLearningRate = learningRate;
            return this;
        }
        
        /**
         * Sets the activation function for the network.
         * 
         * @param activationFunction the activation function to use
         * @return the builder for the trainer
         */
        public Builder activationFunction(IActivationFunction activationFunction) {
            mActivationFunction = activationFunction;
            return this;
        }
        
        /**
         * Sets the maximum number of iterations to use during the training
         * process. Defaults to 500.
         * 
         * @param maxIterations the maximum number of iterations for training
         * @return the builder for the trainer
         */
        public Builder maxIterations(int maxIterations) {
            mMaxIterations = maxIterations;
            return this;
        }
        
        /**
         * Sets whether or not the trainer should record the cost of the 
         * the neural network for each iteration. Defaults to false.
         * 
         * @return the builder for the trainer
         */
        public Builder recordCosts() {
            mRecordCosts = !mRecordCosts;
            return this;
        }
        
        /**
         * Sets how often informational output is output to stdout during
         * training of the network. Default is to show information every 100
         * iterations.
         * 
         * @param beatIterations the number of iterations to use
         * @return the builder for the trainer
         */
        public Builder heartBeat(int beatIterations) {
            mHeartBeat = beatIterations;
            return this;
        }
        
        /**
         * Builds the trainer for the neural network and trains the neural
         * network.
         * 
         * @return the new Trainer object
         */
        public Trainer build() {
            return new Trainer(this);
        }
    }

    /**
     * Builds the neural network required for the trainer.
     * 
     * @param builder the trainer builder
     */
    private Trainer(Builder builder) {
        NeuralNetwork.Builder nnBuilder = new NeuralNetwork
                .Builder(builder.mLayerSizes)
                .inputs(builder.mInputs)
                .expectedValues(builder.mOutputs)
                .lambda(builder.mLambda);
        
        if (builder.mActivationFunction != null) {
            nnBuilder = nnBuilder.activationFunction(builder.mActivationFunction);
        }
        
        neuralNetwork = nnBuilder.build();
        mLearningRate = builder.mLearningRate;
        mRecordCosts = builder.mRecordCosts;
        mMaxIterations = builder.mMaxIterations;
        mHeartBeat = builder.mHeartBeat;
        mCosts = new ArrayList<>();
    }
    
    /**
     * Trains the neural network.
     */
    public void train() {
        int beat = 0;
        
        for (int iteration = 0; iteration < mMaxIterations; iteration++) {
            neuralNetwork.forwardPropagation();
            neuralNetwork.backPropagation();
            
            if (mRecordCosts) {
                mCosts.add(neuralNetwork.getCost());
            }
            
            beat++;

            if (beat == mHeartBeat && mHeartBeat != 0) {
                System.out.println("Iteration: " + (iteration + 1) + ", Cost: " + neuralNetwork.getCost());
                beat = 0;
            }
            
            adjustThetas();
        }
    }
    
    /**
     * Loop through all of the thetas in the neural network and adjust them so 
     * that they are always approaching zero.
     */
    private void adjustThetas() {
        List<DoubleMatrix> newThetas = new ArrayList<DoubleMatrix>();
        List<DoubleMatrix> thetas = neuralNetwork.getThetas();
        
        for (int index = 0; index < thetas.size(); index++) {
            DoubleMatrix gradients = neuralNetwork.getThetaGradient(index);
            DoubleMatrix theta = neuralNetwork.getTheta(index);
            DoubleMatrix newTheta = new DoubleMatrix(theta.rows, theta.columns);
            
            for (int row = 0; row < theta.rows; row++) {
                for (int col = 0; col < theta.columns; col++) {
                    double gradient = gradients.get(row, col);
                    double value = theta.get(row, col);
                    if (gradient > 0) {
                        value -= mLearningRate;
                    } else {
                        value += mLearningRate;
                    }
                    newTheta.put(row, col, value);
                }
            }  
            newThetas.add(newTheta);
        }
        neuralNetwork.setThetas(newThetas);
    }
    
    /**
     * Returns the neural network.
     * 
     * @return the neural network
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }
    
    /**
     * Returns the costs associated with each iteration of training.
     * 
     * @return the list of training costs
     */
    public List<Double> getCosts() {
        return mCosts;
    }
    
}
