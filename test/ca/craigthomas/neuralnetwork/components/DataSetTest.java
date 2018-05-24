/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.components;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class DataSetTest
{
    private static final String BAD_FILENAME = "/this_file_does_not_exist.csv";
    private static final String SAMPLE_FILE = "resources/small_dataset_example.csv";

    private DataSet dataSet;
    
    @Test
    public void testReadFromNonExistentCSVReturnsNull() {
        dataSet = new DataSet(true);
        dataSet.addFromCSVFile(BAD_FILENAME);
        assertTrue(dataSet.isEmpty());
    }
    
    @Test
    public void newDataSetIsEmpty() {
        dataSet = new DataSet(true);
        assertTrue(dataSet.getSamples() == null);
        assertTrue(dataSet.getTruth() == null);
        assertEquals(0, dataSet.getNumColsSamples());
        assertEquals(0, dataSet.getNumColsTruth());
        assertEquals(0, dataSet.getNumSamples());
    }
    
    @Test
    public void testReadFromCSVFileNoTruthWorksCorrectly() throws IOException {
        dataSet = new DataSet(false);
        dataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0}
        });

        assertFalse(dataSet.hasTruth());
        assertEquals(0, dataSet.getNumColsTruth());
        assertEquals(3, dataSet.getNumColsSamples());
        assertEquals(4, dataSet.getNumSamples());
        assertTrue(dataSet.getTruth() == null);
        Assert.assertArrayEquals(expected.toArray(), dataSet.getSamples().toArray(), 0.0001);
    }

    @Test
    public void testReadFromCSVFileSeparatesTruthCorrectly() throws IOException {
        dataSet = new DataSet(true);
        dataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0},
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 0.0}
        });

        DoubleMatrix expectedTruth = new DoubleMatrix(new double [][] {
                {1.0},
                {0.0},
                {0.0},
                {0.0}
        });

        assertTrue(dataSet.hasTruth());
        assertEquals(1, dataSet.getNumColsTruth());
        assertEquals(2, dataSet.getNumColsSamples());
        assertEquals(4, dataSet.getNumSamples());
        Assert.assertArrayEquals(expectedTruth.toArray(), dataSet.getTruth().toArray(), 0.0001);
        Assert.assertArrayEquals(expectedSamples.toArray(), dataSet.getSamples().toArray(), 0.0001);
    }

    @Test
    public void testGenerateRangeZeroRange() {
        DoubleMatrix initMatrix = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
        });
        int start = 0;
        int end = 3;

        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
        });
        
        DoubleMatrix result = DataSet.copyRows(start, end, initMatrix);
        
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.001);
    }

    @Test
    public void testGenerateRangeNonZeroBasedRange() {
        DoubleMatrix initMatrix = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
        });
        int start = 3;
        int end = 4;
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {10.0, 11.0, 12.0}
        });
        
        DoubleMatrix result = DataSet.copyRows(start, end, initMatrix);
        
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.001);
    }
    
    @Test
    public void testSplitWorksCorrectlyWhenTruthExists() throws IOException {
        dataSet = new DataSet(true);
        dataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedTrainingSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0},
                {1.0, 0.0},
                {0.0, 1.0}
        });

        DoubleMatrix expectedTestingSamples = new DoubleMatrix(new double [][] {
                {0.0, 0.0}
        });

        DoubleMatrix expectedTrainingTruth = new DoubleMatrix(new double [][] {
                {1.0},
                {0.0},
                {0.0}
        });
        
        DoubleMatrix expectedTestingTruth = new DoubleMatrix(new double [][] {
                {0.0}
        });     
        Pair<DataSet, DataSet> result = dataSet.splitSequentially(75);
        DoubleMatrix trainingSamples = result.getLeft().getSamples();
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingSamples = result.getRight().getSamples();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertTrue(result.getLeft().hasTruth());
        assertTrue(result.getRight().hasTruth());
        Assert.assertArrayEquals(expectedTrainingSamples.toArray(), trainingSamples.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTrainingTruth.toArray(), trainingTruth.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingSamples.toArray(), testingSamples.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingTruth.toArray(), testingTruth.toArray(), 0.0001);
    }
    
    @Test
    public void testSplitWorksCorrectlyWhenTruthDoesNotExists() throws IOException {
        dataSet = new DataSet(false);
        dataSet.addFromCSVFile(SAMPLE_FILE);
        DoubleMatrix expectedTrainingSamples = new DoubleMatrix(new double [][] {
                {1.0, 1.0, 1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
        });

        DoubleMatrix expectedTestingSamples = new DoubleMatrix(new double [][] {
                {0.0, 0.0, 0.0}
        });

        Pair<DataSet, DataSet> result = dataSet.splitSequentially(75);
        DoubleMatrix trainingSamples = result.getLeft().getSamples();
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingSamples = result.getRight().getSamples();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertFalse(result.getLeft().hasTruth());
        assertFalse(result.getRight().hasTruth());
        Assert.assertArrayEquals(expectedTrainingSamples.toArray(), trainingSamples.toArray(), 0.0001);
        assertTrue(trainingTruth == null);
        Assert.assertArrayEquals(expectedTestingSamples.toArray(), testingSamples.toArray(), 0.0001);
        assertTrue(testingTruth == null);
    }
    
    @Test
    public void testRandomizeWorksCorrectly() throws IOException {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 13.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 23.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 33.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 43.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 53.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 63.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 73.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 83.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 93.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 103.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {111.0, 112.0, 113.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {121.0, 122.0, 123.0};
        samples.add(Arrays.asList(newList));
        
        dataSet.addSamples(samples);
        dataSet.randomize();
       
        DoubleMatrix notExpected = new DoubleMatrix(new double [][] {
                {11.0, 12.0, 13.0},
                {21.0, 22.0, 23.0},
                {31.0, 32.0, 33.0},
                {41.0, 42.0, 43.0},
                {51.0, 52.0, 53.0},
                {61.0, 62.0, 63.0},
                {71.0, 72.0, 73.0},
                {81.0, 82.0, 83.0},
                {91.0, 92.0, 93.0},
                {101.0, 102.0, 103.0},
                {111.0, 112.0, 113.0},
                {121.0, 122.0, 123.0},
        });

        double [] result = dataSet.getSamples().toArray();
        double [] expected = notExpected.toArray();
        assertFalse(Arrays.equals(expected, result));
    }
    
    @Test
    public void testRandomizePreservesTruth() throws IOException {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 13.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 23.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 33.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 43.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 53.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 63.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 73.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 83.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 93.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 103.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {111.0, 112.0, 113.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {121.0, 122.0, 123.0};
        samples.add(Arrays.asList(newList));
        
        dataSet.addSamples(samples);
        dataSet.randomize();
       
        DoubleMatrix sampleData = dataSet.getSamples();
        DoubleMatrix truth = dataSet.getTruth();
        DoubleMatrix result = DoubleMatrix.concatHorizontally(sampleData, truth);
        result.sortRowsi();

        for (int index = 0; index < result.rows; index++) {
            DoubleMatrix row = result.getRow(index);
            assertTrue(
                    Arrays.equals(row.toArray(), new double [] {11.0, 12.0, 13.0}) ||
                    Arrays.equals(row.toArray(), new double [] {21.0, 22.0, 23.0}) ||
                    Arrays.equals(row.toArray(), new double [] {31.0, 32.0, 33.0}) ||
                    Arrays.equals(row.toArray(), new double [] {41.0, 42.0, 43.0}) ||
                    Arrays.equals(row.toArray(), new double [] {51.0, 52.0, 53.0}) ||
                    Arrays.equals(row.toArray(), new double [] {61.0, 62.0, 63.0}) ||
                    Arrays.equals(row.toArray(), new double [] {71.0, 72.0, 73.0}) ||
                    Arrays.equals(row.toArray(), new double [] {81.0, 82.0, 83.0}) ||
                    Arrays.equals(row.toArray(), new double [] {91.0, 92.0, 93.0}) ||
                    Arrays.equals(row.toArray(), new double [] {101.0, 102.0, 103.0}) ||
                    Arrays.equals(row.toArray(), new double [] {111.0, 112.0, 113.0}) ||
                    Arrays.equals(row.toArray(), new double [] {121.0, 122.0, 123.0})                    
            );
        }
    }
    
    @Test
    public void testAddSampleWorksCorrectly() {
        dataSet = new DataSet(false);
        DoubleMatrix sample = new DoubleMatrix(new double [][] {
                {0.333, 0.333, 0.333, 1.0}
        }); 
        dataSet.addSample(sample);
        
        Assert.assertArrayEquals(sample.toArray(), dataSet.getSamples().toArray(), 0.0001);
    }
    
    @Test
    public void testSplitSequentiallyWorksCorrectly() {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 13.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 23.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 33.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 43.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 53.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 63.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 73.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 83.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 93.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 103.0};
        samples.add(Arrays.asList(newList));
        dataSet.addSamples(samples);
        Pair<DataSet, DataSet> result = dataSet.splitSequentially(60);
        
        DoubleMatrix trainingSet = result.getLeft().getSamples();
        DoubleMatrix testingSet = result.getRight().getSamples();
        
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertEquals(6, trainingSet.rows);
        assertEquals(6, trainingTruth.rows);
        assertEquals(4, testingSet.rows);
        assertEquals(4, testingTruth.rows);
        
        DoubleMatrix expectedTrainingSet = new DoubleMatrix(new double [][] {
                {11.0, 12.0},
                {21.0, 22.0},
                {31.0, 32.0},
                {41.0, 42.0},
                {51.0, 52.0},
                {61.0, 62.0}
        });

        DoubleMatrix expectedTrainingTruth = new DoubleMatrix(new double [][] {
                {13.0},
                {23.0},
                {33.0},
                {43.0},
                {53.0},
                {63.0}
        });
        
        DoubleMatrix expectedTestingSet = new DoubleMatrix(new double [][] {
                {71.0, 72.0},
                {81.0, 82.0},
                {91.0, 92.0},
                {101.0, 102.0}
        });
        
        DoubleMatrix expectedTestingTruth = new DoubleMatrix(new double [][] {
                {73.0},
                {83.0},
                {93.0},
                {103.0}
        });

        Assert.assertArrayEquals(expectedTrainingSet.toArray(), trainingSet.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTrainingTruth.toArray(), trainingTruth.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingSet.toArray(), testingSet.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingTruth.toArray(), testingTruth.toArray(), 0.0001);        
    }
    
    @Test
    public void testSplitEquallyEvenNumberWorksCorrectly() {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 0.0};
        samples.add(Arrays.asList(newList));
        dataSet.addSamples(samples);
        Pair<DataSet, DataSet> result = dataSet.splitEqually(50);
        
        DoubleMatrix trainingSet = result.getLeft().getSamples();
        DoubleMatrix testingSet = result.getRight().getSamples();
        
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertEquals(4, trainingSet.rows);
        assertEquals(4, trainingTruth.rows);
        assertEquals(4, testingSet.rows);
        assertEquals(4, testingTruth.rows);
    }
    
    @Test
    public void testSplitEquallyOddNumberWorksCorrectly() {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 0.0};
        samples.add(Arrays.asList(newList));
        dataSet.addSamples(samples);
        Pair<DataSet, DataSet> result = dataSet.splitEqually(50);
        
        DoubleMatrix trainingSet = result.getLeft().getSamples();
        DoubleMatrix testingSet = result.getRight().getSamples();
        
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertEquals(6, trainingSet.rows);
        assertEquals(6, trainingTruth.rows);
        assertEquals(4, testingSet.rows);
        assertEquals(4, testingTruth.rows);
    }
    
    @Test
    public void testSplitEquallyFallsBackToSequentialWhenNotEnoughSamples() {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 1.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 0.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 0.0};
        samples.add(Arrays.asList(newList));
        dataSet.addSamples(samples);
        Pair<DataSet, DataSet> result = dataSet.splitEqually(50);
        
        DoubleMatrix trainingSet = result.getLeft().getSamples();
        DoubleMatrix testingSet = result.getRight().getSamples();
        
        DoubleMatrix trainingTruth = result.getLeft().getTruth();
        DoubleMatrix testingTruth = result.getRight().getTruth();
        
        assertEquals(5, trainingSet.rows);
        assertEquals(5, trainingTruth.rows);
        assertEquals(5, testingSet.rows);
        assertEquals(5, testingTruth.rows);
        
        DoubleMatrix expectedTrainingSet = new DoubleMatrix(new double [][] {
                {11.0, 12.0},
                {21.0, 22.0},
                {31.0, 32.0},
                {41.0, 42.0},
                {51.0, 52.0}
        });

        DoubleMatrix expectedTrainingTruth = new DoubleMatrix(new double [][] {
                {1.0},
                {0.0},
                {0.0},
                {0.0},
                {0.0}
        });
        
        DoubleMatrix expectedTestingSet = new DoubleMatrix(new double [][] {
                {61.0, 62.0},
                {71.0, 72.0},
                {81.0, 82.0},
                {91.0, 92.0},
                {101.0, 102.0}
        });
        
        DoubleMatrix expectedTestingTruth = new DoubleMatrix(new double [][] {
                {0.0},
                {0.0},
                {0.0},
                {0.0},
                {0.0}
        });

        Assert.assertArrayEquals(expectedTrainingSet.toArray(), trainingSet.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTrainingTruth.toArray(), trainingTruth.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingSet.toArray(), testingSet.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTestingTruth.toArray(), testingTruth.toArray(), 0.0001);
    }
    
    @Test
    public void testDupWorksCorrectly() {
        dataSet = new DataSet(true);
        List<List<Double>> samples = new ArrayList<>();
        Double [] newList = new Double [] {11.0, 12.0, 13.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {21.0, 22.0, 23.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {31.0, 32.0, 33.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {41.0, 42.0, 43.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {51.0, 52.0, 53.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {61.0, 62.0, 63.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {71.0, 72.0, 73.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {81.0, 82.0, 83.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {91.0, 92.0, 93.0};
        samples.add(Arrays.asList(newList));
        newList = new Double [] {101.0, 102.0, 103.0};
        samples.add(Arrays.asList(newList));
        dataSet.addSamples(samples);
        
        DataSet newDataSet = dataSet.dup();

        assertEquals(dataSet.getNumColsSamples(), newDataSet.getNumColsSamples());
        assertEquals(dataSet.getNumColsTruth(), newDataSet.getNumColsTruth());
        assertEquals(dataSet.getNumSamples(), newDataSet.getNumSamples());
        Assert.assertArrayEquals(dataSet.getSamples().toArray(), newDataSet.getSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(dataSet.getTruth().toArray(), newDataSet.getTruth().toArray(), 0.0001);
    }
}
