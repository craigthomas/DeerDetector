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

import ca.craigthomas.neuralnetwork.activationfunctions.HyperbolicTangent;

public class HyperbolicTangentTest {
    
    private HyperbolicTangent mTanH;
    
    @Before
    public void setUp() {
        mTanH = new HyperbolicTangent();
    }

    @Test
    public void testTanHSingleValueZero() {
        assertEquals(0, mTanH.apply(0), 0.00001);
    }

    @Test
    public void testTanHSingleValueOne() {
        assertEquals(0.76159, mTanH.apply(1), 0.00001);
    }
    
    @Test
    public void testTanHSingleValueFive() {
        assertEquals(0.99991, mTanH.apply(5), 0.00001);
    }
    
    @Test
    public void testTanHSingleValueTen() {
        assertEquals(1.0, mTanH.apply(10), 0.00001);
    }
    
    @Test
    public void testTanHSingleValueNegativeOne() {
        assertEquals(-0.76159, mTanH.apply(-1), 0.00001);
    }
    
    @Test
    public void testTanHSingleValueNegativeFive() {
        assertEquals(-0.99991, mTanH.apply(-5), 0.00001);
    }
    
    @Test
    public void testTanHSingleValueNegativeTen() {
        assertEquals(-1.0, mTanH.apply(-10), 0.00001);
    }

    @Test
    public void testTanHMatrixSingleRowColumnZero() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 0.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 1.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(0.76159, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 5.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(0.99991, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnTen() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, 10.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(1.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnNegativeOne() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -1.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(-0.76159, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnNegativeFive() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -5.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(-0.99991, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixSingleRowColumnNegativeTen() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1, 1);
        doubleMatrix.put(0, 0, -10.0);
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        assertEquals(-1.0, result.get(0, 0), 0.00001);
    }
    
    @Test
    public void testTanHMatrixMultipleRowsAndColumns() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(new double[][] {{1.0, 5.0, 10.0}, {-1.0, -5.0, -10.0}});
        DoubleMatrix expected = new DoubleMatrix(new double[][] {{0.76159, 0.99991, 1.0}, {-0.76159, -0.99991, -1.0}});
        DoubleMatrix result = mTanH.apply(doubleMatrix);
        for (int row = 0; row < result.rows; row++) {
            for (int col = 0; col < result.columns; col++) {
                assertEquals(expected.get(row, col), result.get(row, col), 0.00001);
            }
        }
    }
    
    @Test
    public void testHyperbolicTangentGradient() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(new double [][] {
                {0.0, 1.0, -1.0, 5.0, -5.0}
        });
        
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {1.0, 0.41997434, 0.41997434, 0.00018158, 0.00018158} 
        });

        DoubleMatrix result = mTanH.gradient(doubleMatrix);
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.000001);
    }
}
