/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.components;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import ca.craigthomas.neuralnetwork.common.CsvIO;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import ca.craigthomas.neuralnetwork.commandline.Runner;

/**
 * The DataSet class is used to read data from various sources. The DataSet
 * class keeps track of two types of data: the actual example inputs called
 * Samples, and their optional output labels called Truth. 
 */
public class DataSet
{
    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    // The samples in the data set
    private DoubleMatrix samples;
    // The truth values for the samples
    private DoubleMatrix truth;
    // Whether truth values are present
    private boolean hasTruth;
    // A variable for generating random values
    private Random random;
    
    /**
     * Generates a new DataSet based upon current internal values.
     * 
     * @param hasTruth whether the true values are known for the class
     * @param samples the samples for the DataSet
     * @param truth the truth values for the DataSet
     */
    public DataSet(boolean hasTruth, DoubleMatrix samples, DoubleMatrix truth)
    {
        this.samples = samples;
        this.truth = truth;
        this.hasTruth = hasTruth;
        random = new Random();
    }
    
    /**
     * The DataSet constructor. If hasTruth is set to true, then the
     * class will expect ground truth to be passed in with the data.
     *
     * @param hasTruth whether the Samples have ground truth information
     */
    public DataSet(boolean hasTruth) {
        this.hasTruth = hasTruth;
        random = new Random();
    }
    
    /**
     * Returns the Samples.
     * 
     * @return the Samples
     */
    public DoubleMatrix getSamples() {
        return samples;
    }
    
    /**
     * Returns the ground truth.
     * 
     * @return the ground truth
     */
    public DoubleMatrix getTruth() {
        return truth;
    }
    
    /**
     * Get the number of columns in the Samples.
     * 
     * @return the number of columns in the Samples
     */
    public int getNumColsSamples() {
        return (samples == null) ? 0 : samples.columns;
    }
    
    /**
     * Returns the number of columns in the Truth data.
     * 
     * @return return the number of columns in the Truth data
     */
    public int getNumColsTruth() {
        return (truth == null) ? 0 : truth.columns;
    }
    
    /**
     * Returns the number of samples (rows) in the DataSet.
     * 
     * @return the number of samples in the DataSet
     */
    public int getNumSamples() {
        return (samples == null) ? 0 : samples.rows;
    }
    
    /**
     * Returns true if the DataSet has ground truth associated with it,
     * false otherwise.
     * 
     * @return true if the DataSet has ground truth
     */
    public boolean hasTruth() {
        return hasTruth;
    }
    
    /**
     * Reads samples from a CSV file, and adds them to the DataSet. If header
     * is set (true), will ignore the first line of the file.
     * 
     * @param filename the name of the file to read from
     */
    public void addFromCSVFile(String filename) {
        List<List<Double>> data = CsvIO.readCSVFile(filename);
        addSamples(data);
    }
    
    /**
     * Adds a list of samples to the DataSet. Each element in the list contains
     * a list of Doubles, which are assumed to be the samples to add. If the
     * DataSet has ground truth, the last element in each of the list of samples
     * is assumed to be the ground truth label.
     * 
     * @param samples the list of samples to add
     */
    public void addSamples(List<List<Double>> samples) {
        if (samples == null) {
            LOGGER.warning("no samples supplied to dataset");
            return;
        }

        for (List<Double> row : samples) {
            Double [] temp = row.toArray(new Double[row.size()]);
            double [] sampleRow = ArrayUtils.toPrimitive(temp);
            double [][] matrix = new double[1][sampleRow.length];
            if (hasTruth) {
                matrix[0] = ArrayUtils.subarray(sampleRow, 0, sampleRow.length-1);
                addSampleRow(new DoubleMatrix(matrix));
                matrix[0] = ArrayUtils.subarray(sampleRow, sampleRow.length-1, sampleRow.length);
                addTruthRow(new DoubleMatrix(matrix));
            } else {
                matrix[0] = sampleRow;
                addSampleRow(new DoubleMatrix(matrix));
            }
        } 
    }
    
    /**
     * Adds a single row DoubleMatrix set of values to the list of samples.
     *  
     * @param sample the DoubleMatrix column vector to add
     */
    public void addSample(DoubleMatrix sample) {
        List<List<Double>> newSampleList = new ArrayList<>();
        double [] values = new double [sample.columns];
        for (int index = 0; index < sample.columns; index++) {
            values[index] = sample.get(0, index);
        }
        newSampleList.add(Arrays.asList(ArrayUtils.toObject(values)));
        addSamples(newSampleList);
    }

    /**
     * Internal helper function that will add a row of samples to the DataSet.
     * 
     * @param samples the matrix of samples to add
     */
    private void addSampleRow(DoubleMatrix samples) {
        this.samples = (this.samples == null) ? samples : DoubleMatrix.concatVertically(this.samples, samples);
    }
    
    /**
     * Internal helper function that will add a row of ground truth labels
     * to the DataSet.
     * 
     * @param truth the matrix of truth values to add
     */
    private void addTruthRow(DoubleMatrix truth) {
        this.truth = (this.truth == null) ? truth : DoubleMatrix.concatVertically(this.truth, truth);
    }
    
    /**
     * Randomizes the data points within the DataSet.
     */
    public void randomize() {
        for (int counter = 0; counter < samples.rows * 5; counter++) {
            int firstIndex = random.nextInt(samples.rows);
            int secondIndex = random.nextInt(samples.rows);
            DoubleMatrix tempRow = samples.getRow(firstIndex);
            DoubleMatrix tempRow1 = samples.getRow(secondIndex);
            samples.putRow(firstIndex, tempRow1);
            samples.putRow(secondIndex, tempRow);
            
            if (hasTruth) {
                tempRow = truth.getRow(firstIndex);
                tempRow1 = truth.getRow(secondIndex);
                truth.putRow(firstIndex, tempRow1);
                truth.putRow(secondIndex, tempRow);
            }
        }
    }
    
    /**
     * A workaround for the JBlas library IntervalRange class - specifying 
     * ranges that don't start at zero have a problem. Copy the given rows
     * instead.
     * 
     * @param a the start of the range (inclusive)
     * @param b the end of the range (exclusive)
     * @param matrix the matrix to copy from
     * @return a new matrix with the rows from a to b
     */
    protected static DoubleMatrix copyRows(int a, int b, DoubleMatrix matrix) {
        DoubleMatrix result = null;
        for (int index = a; index < b; index++) {
            if (result == null) {
                result = matrix.getRow(index).dup();
            } else {
                result = DoubleMatrix.concatVertically(result, matrix.getRow(index).dup());
            }
        }
        return result;
    }
    
    /**
     * Splits a DataSet into two sets - a training and a testing set - based
     * upon the percentage. For example, a percentage of 60 would allocate 
     * 60% to the training set and 40% to the testing set. Returns a pair of 
     * DataSets - the first pair is the training set, the second pair is the
     * testing set. To build the training set, will take the first 60% of the
     * data starting at row 0. The remainder will be shunted to teh testing set.
     * Use splitEqually if you wish to maintain an equal balance between the
     * positive and negative classes when constructing a training data set.
     * 
     * @param percentage the percentage to put into the training set
     * @return a pair of DataSets - left is training, right is testing
     */
    public Pair<DataSet, DataSet> splitSequentially(int percentage) {
        int trainStart = 0; 
        int trainEnd = (int)Math.ceil(((percentage / 100.0) * (float) samples.rows));
        int testEnd = samples.rows;
        DoubleMatrix trainingSamples = copyRows(trainStart, trainEnd, samples);
        DoubleMatrix testingSamples = copyRows(trainEnd, testEnd, samples);
        DoubleMatrix trainingTruth = null;
        DoubleMatrix testingTruth = null;
        if (hasTruth) {
            trainingTruth = copyRows(trainStart, trainEnd, truth);
            testingTruth = copyRows(trainEnd, testEnd, truth);
        }
        DataSet trainingSet = new DataSet(hasTruth, trainingSamples, trainingTruth);
        DataSet testingSet = new DataSet(hasTruth, testingSamples, testingTruth);
        return Pair.of(trainingSet, testingSet);
    }
    
    /**
     * Splits a DataSet into two sets - a training and a testing set - based
     * upon the percentage. For example, a percentage of 60 would allocate 
     * 60% to the training set and 40% to the testing set. Returns a pair of 
     * DataSets - the first pair is the training set, the second pair is the
     * testing set. Ensures that half of the examples in the training set are
     * positive cases, and half of the examples in the training set are negative
     * cases. If there are not enough positive or negative samples to build
     * an equal training DataSet, then fall back to splitSequentially. 
     *  
     * @param percentage the percentage split to make
     * @return the training DataSet, and the testing DataSet
     */
    public Pair<DataSet, DataSet> splitEqually(int percentage) {
        boolean selectedRows [] = new boolean [samples.rows];
        int half = (int)Math.ceil(((percentage / 100.0) * (float) samples.rows) / 2);
        int negCounter = 0;
        int posCounter = 0;
        
        // First, make sure that the data set has at least 'half' number of
        // negative and positive samples - if we don't have it, default to 
        // splitSequentially.
        for (int index = 0; index < truth.rows; index++) {
            if (truth.get(index, 0) == 1.0) {
                posCounter++;
            } else {
                negCounter++;
            }
        }
        
        if (negCounter < half || posCounter < half) {
            LOGGER.warning("cannot split DataSet equally (" + posCounter + " pos, " + negCounter + " neg, want " + half + " each)");
            return splitSequentially(percentage);
        }
        
        posCounter = 0;
        negCounter = 0;
        DataSet trainingData = new DataSet(hasTruth);
        DataSet testingData = new DataSet(hasTruth);
        
        // Select an index at random and see if we have already added it to
        // the training DataSet. Loop until we have the desired number of
        // positive and negative cases
        while (negCounter < half || posCounter < half) {
            int nextIndex = random.nextInt(samples.rows);
            if (!selectedRows[nextIndex]) {
                DoubleMatrix row = DoubleMatrix.concatHorizontally(samples.getRow(nextIndex), truth.getRow(nextIndex));
                if (truth.get(nextIndex, 0) == 1.0 && posCounter < half) {
                    selectedRows[nextIndex] = true;
                    posCounter++;
                    trainingData.addSample(row);
                }
                
                if (truth.get(nextIndex, 0) == 0.0 && negCounter < half) {
                    selectedRows[nextIndex] = true;
                    negCounter++;
                    trainingData.addSample(row);
                }
            }
        }
        
        // Take all the remaining unused samples, and include them in the
        // testing DataSet.
        for (int index = 0; index < selectedRows.length; index++) {
            if (!selectedRows[index]) {
                DoubleMatrix row = DoubleMatrix.concatHorizontally(samples.getRow(index), truth.getRow(index));
                testingData.addSample(row);
            }
        }
        return Pair.of(trainingData, testingData);
    }
    
    /**
     * Duplicate this DataSet.
     * 
     * @return a duplicate of this DataSet
     */
    public DataSet dup() {
        return new DataSet(hasTruth, samples.dup(), truth.dup());
    }

    public boolean isEmpty() {
        return samples == null;
    }
}
