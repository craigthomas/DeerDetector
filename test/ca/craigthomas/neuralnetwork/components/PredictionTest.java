package ca.craigthomas.neuralnetwork.components;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class PredictionTest {

    private Prediction mPrediction;
    
    @Test
    public void testSaveSampleToClassWorksCorrectly() {
        DoubleMatrix truePositives = new DoubleMatrix(new double [][] {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        
        DoubleMatrix trueNegatives = new DoubleMatrix(new double [][] {
                {7.0, 8.0, 9.0}
        });
        
        DoubleMatrix falsePositives = new DoubleMatrix(new double[][] {
                {10.0, 11.0, 12.0},
                {13.0, 14.0, 15.0}
        });
        
        DoubleMatrix falseNegatives = new DoubleMatrix(new double [][] {
                {16.0, 17.0, 18.0}
        });
        
        NeuralNetwork mockNeuralNetwork = mock(NeuralNetwork.class);
        mPrediction = new Prediction(mockNeuralNetwork, 0.0);
        
        mPrediction.saveSampleToClass(truePositives.getRow(0), Prediction.TRUE_POS);
        mPrediction.saveSampleToClass(truePositives.getRow(1), Prediction.TRUE_POS);
        mPrediction.saveSampleToClass(trueNegatives.getRow(0), Prediction.TRUE_NEG);
        mPrediction.saveSampleToClass(falsePositives.getRow(0), Prediction.FALSE_POS);
        mPrediction.saveSampleToClass(falsePositives.getRow(1), Prediction.FALSE_POS);
        mPrediction.saveSampleToClass(falseNegatives.getRow(0), Prediction.FALSE_NEG);;

        Assert.assertArrayEquals(truePositives.toArray(), mPrediction.getTruePositiveSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(trueNegatives.toArray(), mPrediction.getTrueNegativeSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(falsePositives.toArray(), mPrediction.getFalsePositiveSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(falseNegatives.toArray(), mPrediction.getFalseNegativeSamples().toArray(), 0.0001);
    }
    
    @Test
    public void testPredictWorksCorrectly() {
        DoubleMatrix samples = new DoubleMatrix(new double [][] {
                {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
                {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
                
        });
        
        DoubleMatrix truth = new DoubleMatrix(new double [][] {
                {1.0}, {1.0}, {1.0}, {1.0}, {1.0},
                {0.0}, {0.0}, {0.0}, {0.0}, {0.0}
        });
        
        DoubleMatrix predictions = new DoubleMatrix(new double[][] {
                {1.0}, {1.0}, {1.0}, {0.0}, {1.0},
                {0.0}, {1.0}, {0.0}, {1.0}, {0.0}
        });
        
        DoubleMatrix expectedTruePositives = new DoubleMatrix(new double [][] {
                {1.0}, {2.0}, {3.0}, {5.0}
        });
        
        DoubleMatrix expectedTrueNegatives = new DoubleMatrix(new double [][] {
                {6.0}, {8.0}, {10.0}
        });

        DoubleMatrix expectedFalseNegatives = new DoubleMatrix(new double [][] {
                {4.0}
        });

        DoubleMatrix expectedFalsePositives = new DoubleMatrix(new double [][] {
                {7.0}, {9.0}
        });
        
        DataSet mockDataSet = mock(DataSet.class);
        when(mockDataSet.getSamples()).thenReturn(samples);
        when(mockDataSet.getTruth()).thenReturn(truth);
        
        NeuralNetwork mockNeuralNetwork = mock(NeuralNetwork.class);
        when(mockNeuralNetwork.predict(samples)).thenReturn(predictions);
        
        mPrediction = new Prediction(mockNeuralNetwork, 0.5);
        mPrediction.predict(mockDataSet);
        
        assertEquals(4, mPrediction.getTruePositives(), 0.0001);
        assertEquals(3, mPrediction.getTrueNegatives(), 0.0001);
        assertEquals(2, mPrediction.getFalsePositives(), 0.0001);
        assertEquals(1, mPrediction.getFalseNegatives(), 0.0001);
        
        Assert.assertArrayEquals(expectedTruePositives.toArray(), mPrediction.getTruePositiveSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(expectedFalsePositives.toArray(), mPrediction.getFalsePositiveSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTrueNegatives.toArray(), mPrediction.getTrueNegativeSamples().toArray(), 0.0001);
        Assert.assertArrayEquals(expectedFalseNegatives.toArray(), mPrediction.getFalseNegativeSamples().toArray(), 0.0001);
        
        assertEquals(0.66666, mPrediction.getPrecision(), 0.00001);
        assertEquals(0.8, mPrediction.getRecall(), 0.00001);
        assertEquals(0.727272, mPrediction.getF1(), 0.00001);
    }

}
