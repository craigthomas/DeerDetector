/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * A class used to calculate the sigmoid of all of the elements in a
 * DoubleMatrix. The Sigmoid function is simply:
 * 
 *   S(t) =       1
 *           ------------
 *            1 + e^(-t)
 */
public class Sigmoid implements IActivationFunction
{
    public Sigmoid() {
    }
    
    /**
     * Calculate the Sigmoid value for every element in the specified matrix.
     * 
     * @param input the DoubleMatrix to use as input
     * @return the sigmoid value of the input matrix
     */
    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix result = new DoubleMatrix().copy(input);
        DoubleMatrix ones = DoubleMatrix.ones(result.rows, result.columns);
        result.muli(-1);
        MatrixFunctions.expi(result);
        result.addi(1);
        return ones.divi(result);
    }
    
    /**
     * Computes the gradient of the sigmoid function with the specified inputs.
     * 
     * @param input the DoubleMatrix to use as input
     * @return the gradient of the sigmoid
     */
    public DoubleMatrix gradient(DoubleMatrix input) {
        DoubleMatrix sigmoid = apply(input);
        DoubleMatrix ones = DoubleMatrix.ones(input.rows, input.columns);
        return sigmoid.mul(ones.sub(sigmoid));
    }

    /**
     * Calculate the Sigmoid value for a single double.
     * 
     * @param input the double to use as input
     * @return the sigmoid value of the input
     */
    public double apply(double input) {
        return (1.0 / (1.0 + Math.exp(-input)));
    }
}
