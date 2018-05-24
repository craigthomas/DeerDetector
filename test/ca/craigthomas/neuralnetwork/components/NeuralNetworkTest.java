/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.components;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import ca.craigthomas.neuralnetwork.components.NeuralNetwork;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {

    private NeuralNetwork mNeuralNetwork;
    private List<Integer> layerSizes;
    
    @Before
    public void setUp() {
        mNeuralNetwork = null;
    }
    
    @Test (expected=IllegalArgumentException.class)
    public void testLessThanTwoLayers() {
        layerSizes = Arrays.asList(3);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
    }
    
    @Test
    public void testPredictANDFunction() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) and x2 (0) == 0
                {0.0},       // x1 (0) and x2 (1) == 0
                {0.0},       // x1 (1) and x2 (0) == 0
                {1.0}        // x1 (1) and x2 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictORFunction() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) or x2 (0) == 0
                {1.0},       // x1 (0) or x2 (1) == 1
                {1.0},       // x1 (1) or x2 (0) == 1
                {1.0}        // x1 (1) or x2 (1) == 1
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictXORFunction() {
        layerSizes = Arrays.asList(2, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0},
                {200.0, -150.0, -150.0}
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0},       // x1 (0) xor x2 (0) == 0
                {1.0},       // x1 (0) xor x2 (1) == 1
                {1.0},       // x1 (1) xor x2 (0) == 1
                {0.0}        // x1 (1) xor x2 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictNOTFunction() {
        layerSizes = Arrays.asList(1, 1);
        DoubleMatrix theta = new DoubleMatrix(new double [][] {
                {100.0, -200.0}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0},  // x1 = 0
                {1.0}   // x1 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {1.0},       // not x1 (0) == 1
                {0.0},       // not x1 (1) == 0
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testPredictXNORFunction() {
        layerSizes = Arrays.asList(2, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {-300.0, 200.0, 200.0},
                {100.0, -200.0, -200.0},
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {-100.0, 200.0, 200.0},
        }); 
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.0, 0.0},  // x1 = 0, x2 = 0
                {0.0, 1.0},  // x1 = 0, x2 = 1
                {1.0, 0.0},  // x1 = 1, x2 = 0
                {1.0, 1.0}   // x1 = 1, x2 = 1
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {1.0},       // x1 (0) xnor x2 (0) == 1
                {0.0},       // x1 (0) xnor x2 (1) == 0
                {0.0},       // x1 (1) xnor x2 (0) == 0
                {1.0}        // x1 (1) xnor x2 (1) == 1
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.predict(testInputs);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testNeuralNetworkGetCostNoRegularization() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double expectedCost = 0.88101;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).expectedValues(expectedOutputs).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCostNoRegularization(8), 0.0005);
    }
    
    @Test
    public void testGetMatrixNoBiasRemovesBiasUnit() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.69149, 0.23113}     
        });

        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        DoubleMatrix result = mNeuralNetwork.getMatrixNoBias(theta1);
        
        Assert.assertArrayEquals(expectedOutputs.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void getThetaRegularizationReturnsZeroWithDefaultLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        double expected = 0.0;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).build();
        double result = mNeuralNetwork.getThetaRegularization(1);
        
        assertEquals(expected, result, 0.0001);
    } 
    
    @Test
    public void getThetaRegularizationReturnsCorrectWithLambdaOne() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {1.000, 0.69149, 0.23113}
        });
        
        double expected = 0.26579;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas).lambda(1.0).build();
        double result = mNeuralNetwork.getThetaRegularization(1);
        
        assertEquals(expected, result, 0.0001);
    }
    
    @Test
    public void testNeuralNetworkGetCostNoLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });

        double expectedCost = 0.88101;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCost(), 0.0005);
    }
    
    @Test
    public void testNeuralNetworkGetCostWithLambda() {
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double lambda = 1.0;

        double expectedCost = 0.91423;
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs)
                .lambda(lambda).build();
        mNeuralNetwork.predict(testInputs);

        assertEquals(expectedCost, mNeuralNetwork.getCost(), 0.0005);
    }
    
    @Test
    public void testNewNeuralNetworkHasRandomValuesForThetas() {
        layerSizes = Arrays.asList(2, 1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        DoubleMatrix result = mNeuralNetwork.getTheta(0);
        
        double maxRange = Math.sqrt(6);

        assertEquals(1, result.rows);
        assertEquals(3, result.columns);
        
        for (int row = 0; row < result.rows; row++) {
            for (int col = 0; col < result.columns; col++) {
                assertTrue(result.get(row, col) < maxRange);
                assertTrue(result.get(row, col) > -maxRange);
            }
        }
    }
    
    @Test (expected=ArrayIndexOutOfBoundsException.class)
    public void testGetThetaNegativeThetaThrowsException() {
        layerSizes = Arrays.asList(2,1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        mNeuralNetwork.getTheta(-1);
    }
    
    @Test (expected=ArrayIndexOutOfBoundsException.class)
    public void testGetThetaIndexTooLargeThrowsException() {
        layerSizes = Arrays.asList(2,1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        mNeuralNetwork.getTheta(10);
    }

    @Test (expected=ArrayIndexOutOfBoundsException.class)
    public void testGetDeltaNegativeDeltaThrowsException() {
        layerSizes = Arrays.asList(2,1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        mNeuralNetwork.getDelta(-1);
    }
    
    @Test (expected=ArrayIndexOutOfBoundsException.class)
    public void testGetDeltaIndexTooLargeThrowsException() {
        layerSizes = Arrays.asList(2,1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        mNeuralNetwork.getDelta(10);
    }
    
    @Test (expected=IllegalArgumentException.class)
    public void testGetDeltaNullDeltaThrowsException() {
        layerSizes = Arrays.asList(2,1);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).build();
        mNeuralNetwork.getDelta(0);
    }   
    
    @Test
    public void testNeuralNetworkCorrectDeltas() {
        layerSizes = Arrays.asList(2, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {0.92982, 0.33938}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double lambda = 0.0;

        double expectedCost = 0.86126;
        
        DoubleMatrix expectedDelta2 = new DoubleMatrix(new double[][] {
                {0.76255}, {0.77028}, {0.76795}, {-0.23490}, {-0.23228},
                {0.76691}, {-0.23455}, {-0.23273}
        });
        
        DoubleMatrix expectedDelta1 = new DoubleMatrix(new double[][] {
                {0.054551}, {0.037707}, {0.043747}, {-0.015350}, {-0.013405},
                {0.046172}, {-0.015107}, {-0.013754}
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs)
                .lambda(lambda).build();
        mNeuralNetwork.forwardPropagation();
        mNeuralNetwork.backPropagation();
        
        DoubleMatrix delta2 = mNeuralNetwork.getDelta(2);
        DoubleMatrix delta1 = mNeuralNetwork.getDelta(1);

        assertEquals(expectedCost, mNeuralNetwork.getCost(), 0.0005);
        Assert.assertArrayEquals(expectedDelta2.toArray(), delta2.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedDelta1.toArray(), delta1.toArray(), 0.0001);
    }
    
    @Test
    public void testThetaGetGradient() {
        layerSizes = Arrays.asList(2, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {0.92982, 0.33938}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double lambda = 0.0;

        DoubleMatrix expectedTheta1Grad = new DoubleMatrix(new double [][] {
                {0.0155702, 0.0101371, 0.0016941}
        });
        
        DoubleMatrix expectedTheta2Grad = new DoubleMatrix(new double [][] {
                {0.26665, 0.20640}
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs)
                .lambda(lambda).build();
        mNeuralNetwork.forwardPropagation();
        mNeuralNetwork.backPropagation();
        
        DoubleMatrix theta1Grad = mNeuralNetwork.getThetaGradient(0);
        DoubleMatrix theta2Grad = mNeuralNetwork.getThetaGradient(1);

        Assert.assertArrayEquals(expectedTheta1Grad.toArray(), theta1Grad.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTheta2Grad.toArray(), theta2Grad.toArray(), 0.0001);
    }
    
    @Test
    public void testThetaGetGradientWithLambda() {
        layerSizes = Arrays.asList(2, 2, 1);
        DoubleMatrix theta1 = new DoubleMatrix(new double [][] {
                {0.73258, 0.69149, 0.23113}
        });
        
        DoubleMatrix theta2 = new DoubleMatrix(new double [][] {
                {0.92982, 0.33938}
        });
        
        DoubleMatrix testInputs = new DoubleMatrix(new double [][] {
                {0.126222, 0.077800},
                {0.956743, 0.682936},
                {0.723205, 0.311276},
                {0.307307, 0.429310},
                {0.772100, 0.066606},
                {0.660782, 0.067908},
                {0.161723, 0.994278},
                {0.472773, 0.777440},
        });

        DoubleMatrix expectedOutputs = new DoubleMatrix(new double [][] {
                {0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}
        });
        
        double lambda = 1.0;

        DoubleMatrix expectedTheta1Grad = new DoubleMatrix(new double [][] {
                {0.0155700, 0.096573, 0.030585}
        });
        
        DoubleMatrix expectedTheta2Grad = new DoubleMatrix(new double [][] {
                {0.26665, 0.24882}
        });
        
        List<DoubleMatrix> thetas = Arrays.asList(theta1, theta2);
        mNeuralNetwork = new NeuralNetwork.Builder(layerSizes).theta(thetas)
                .inputs(testInputs).expectedValues(expectedOutputs)
                .lambda(lambda).build();
        mNeuralNetwork.forwardPropagation();
        mNeuralNetwork.backPropagation();
        
        DoubleMatrix theta1Grad = mNeuralNetwork.getThetaGradient(0);
        DoubleMatrix theta2Grad = mNeuralNetwork.getThetaGradient(1);

        Assert.assertArrayEquals(expectedTheta1Grad.toArray(), theta1Grad.toArray(), 0.0001);
        Assert.assertArrayEquals(expectedTheta2Grad.toArray(), theta2Grad.toArray(), 0.0001);
    }
}
