/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.components;

import org.jblas.DoubleMatrix;

/**
 * Given a NeuralNetwork and a set of data samples to process (with their 
 * associated truth values), makes predictions and measures the correctness
 * of those predictions. Used to keep track of the precision, recall, and F-1 
 * score of a neural network on a particular components.
 */
public class Prediction {
    
    protected static final int TRUE_POS = 1;
    protected static final int TRUE_NEG = 2;
    protected static final int FALSE_POS = 3;
    protected static final int FALSE_NEG = 4;

    private NeuralNetwork mNeuralNetwork;
    private DoubleMatrix mSamples;
    private DoubleMatrix mTruth;
    private double mTruePositives = 0.0;
    private double mTrueNegatives = 0.0;
    private double mFalsePositives = 0.0;
    private double mFalseNegatives = 0.0;
    private double mPredictionThreshold;
    private double mPrecision = 0.0;
    private double mRecall = 0.0;
    private double mF1 = 0.0;
    private DoubleMatrix mTruePositiveSamples;
    private DoubleMatrix mTrueNegativeSamples;
    private DoubleMatrix mFalsePositiveSamples;
    private DoubleMatrix mFalseNegativeSamples;
    
    /**
     * Constructor method for the prediction class. Needs a trained NeuralNetwork
     * and a threshold parameter for predictions. Values above the 
     * predictionThreshold will cause the classifier to prediction a 1 (positive)
     * while below the predictionThreshold will cause the classifier to 
     * predict 0 (negative).
     * 
     * @param model a trained neural network
     * @param predictionThreshold the prediction threshold
     */
    public Prediction(NeuralNetwork model, double predictionThreshold) {
        mNeuralNetwork = model;
        mPredictionThreshold = predictionThreshold;
    }
    
    /**
     * Saves the sample to one of four different DoubleMatrix, depending on the
     * type of sample it is (True Positive, True Negative, etc).
     * 
     * @param sample the sample to save
     * @param type the type of sample it is
     */
    protected void saveSampleToClass(DoubleMatrix sample, int type) {
        switch (type) {
        case TRUE_POS:
            mTruePositiveSamples = (mTruePositiveSamples == null) ? sample : DoubleMatrix.concatVertically(mTruePositiveSamples, sample);
            break;
            
        case TRUE_NEG:
            mTrueNegativeSamples = (mTrueNegativeSamples == null) ? sample : DoubleMatrix.concatVertically(mTrueNegativeSamples, sample);
            break;

        case FALSE_POS:
            mFalsePositiveSamples = (mFalsePositiveSamples == null) ? sample : DoubleMatrix.concatVertically(mFalsePositiveSamples, sample);
            break;

        case FALSE_NEG:
            mFalseNegativeSamples = (mFalseNegativeSamples == null) ? sample : DoubleMatrix.concatVertically(mFalseNegativeSamples, sample);
            break;
        }
    }
    
    /**
     * Given a set of data samples, make predictions. The DataSet must have
     * been previously split into a training and testing pair using
     * splitData. It is also good to have randomized the data with 
     * randomize().
     * 
     * @param samples the set of samples to predict
     */
    public void predict(DataSet samples) {
        mSamples = samples.getSamples().dup();
        mTruth = samples.getTruth().dup();
        DoubleMatrix predictions = mNeuralNetwork.predict(mSamples);
                
        for (int index = 0; index < predictions.rows; index++) {
            int prediction = (predictions.get(index, 0) > mPredictionThreshold) ? 1 : 0;
            int actual = (mTruth.get(index, 0) > mPredictionThreshold) ? 1 : 0;
            DoubleMatrix sample = mSamples.getRow(index);
            if (actual == 1) {
                if (prediction == actual) {
                    mTruePositives += 1.0;
                    saveSampleToClass(sample, TRUE_POS);
                } else {
                    mFalseNegatives += 1.0;
                    saveSampleToClass(sample, FALSE_NEG);
                }
            } else {
                if (prediction == actual) {
                    mTrueNegatives += 1.0;
                    saveSampleToClass(sample, TRUE_NEG);
                } else {
                    mFalsePositives += 1.0;
                    saveSampleToClass(sample, FALSE_POS);
                }
            }
        }
        mPrecision = mTruePositives / (mTruePositives + mFalsePositives);
        mRecall = mTruePositives / (mTruePositives + mFalseNegatives);
        mF1 = 2 * (mPrecision * mRecall) / (mPrecision + mRecall);
    }
    
    /**
     * Gets the overall precision.
     * 
     * @return the precision
     */
    public double getPrecision() {
        return mPrecision;
    }
    
    /**
     * Gets the overall recall.
     * 
     * @return the recall
     */
    public double getRecall() {
        return mRecall;
    }
    
    /**
     * Gets the overall F-1 score.
     * 
     * @return the F1 score
     */
    public double getF1() {
        return mF1;
    }
    
    /**
     * Gets the number of true positives.
     * 
     * @return the number of true positives
     */
    public double getTruePositives() {
        return mTruePositives;
    }
    
    /**
     * Gets the number of true negatives.
     * 
     * @return the number of true negatives
     */
    public double getTrueNegatives() {
        return mTrueNegatives;
    }
    
    /**
     * Gets the number of false positives
     * 
     * @return the number of false positives
     */
    public double getFalsePositives() {
        return mFalsePositives;
    }
    
    /**
     * Gets the number of false negatives.
     * 
     * @return the number of false negatives
     */
    public double getFalseNegatives() {
        return mFalseNegatives;
    }
    
    /**
     * Gets the samples that were true positives.
     * 
     * @return the samples that were true positives
     */
    public DoubleMatrix getTruePositiveSamples() {
        return mTruePositiveSamples;
    }
    
    /**
     * Gets the sample that were true negatives.
     * 
     * @return the samples that were true negatives
     */
    public DoubleMatrix getTrueNegativeSamples() {
        return mTrueNegativeSamples;
    }
    
    /**
     * Gets the samples that were false positives.
     * 
     * @return the samples that were false positives
     */
    public DoubleMatrix getFalsePositiveSamples() {
        return mFalsePositiveSamples;
    }
    
    /**
     * Gets the samples that were false negatives.
     * 
     * @return the samples that were false negatives
     */
    public DoubleMatrix getFalseNegativeSamples() {
        return mFalseNegativeSamples;
    }
}
