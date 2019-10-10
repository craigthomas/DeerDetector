/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.activationfunctions;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import ca.craigthomas.neuralnetwork.activationfunctions.Sigmoid;

public class SigmoidTest {
    
    private Sigmoid mSigmoid;
    
    @Before
    public void setUp() {
        mSigmoid = new Sigmoid();
    }

    @Test
    public void testSigmoidSingleValueZero() {
        assertEquals(0.5, mSigmoid.apply(0), 0.00001);
    }

    @Test
    public void testSigmoidSingleValueOne() {
        assertEquals(0.73106, mSigmoid.apply(1), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueFive() {
        assertEquals(0.99331, mSigmoid.apply(5), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueOneHundred() {
        assertEquals(1.0, mSigmoid.apply(100), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeOne() {
        assertEquals(0.26894, mSigmoid.apply(-1), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeFive() {
        assertEquals(0.0066929, mSigmoid.apply(-5), 0.00001);
    }
    
    @Test
    public void testSigmoidSingleValueNegativeOneHundred() {
        assertEquals(0.0, mSigmoid.apply(-100), 0.00001);
    }

    @Test
    public void testSigmoidMatrixSingleRowColumnZero() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 0.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.5, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 1.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.73106, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 5.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.99331, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnOneHundred() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 100.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(1.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -1.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.26894, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -5.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.0066929, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixSingleRowColumnNegativeOneHundred() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -100.0);
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        assertEquals(0.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testSigmoidMatrixMultipleRowsAndColumns() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(new double[][] {{1.0, 5.0, 100.0}, {-1.0, -5.0, -100.0}});
        DoubleMatrix expected = new DoubleMatrix(new double[][] {{0.73106, 0.99331, 1.0}, {0.26894, 0.0066929, 0.0}});
        DoubleMatrix result = mSigmoid.apply(doubleMatrix);
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testSigmoidGradient() {
        DoubleMatrix input = new DoubleMatrix(new double[][] {
                {0.83784, 1.55201, 1.30461, 1.04431, 1.28187, 1.20520, 1.07422, 1.23919}
        });
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {0.210792}, {0.144243}, {0.167854}, {0.192553}, {0.170042}, {0.177398}, {0.189780}, {0.174142}
        });
        DoubleMatrix result = mSigmoid.gradient(input);
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.0001);
    }
}
