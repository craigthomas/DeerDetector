/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * A class used to calculate the hyperbolic tangent of all of the 
 * elements in a DoubleMatrix. The HyperbolicTangent function is simply:
 * 
 *   TanH(t) =  1 - e^(-2t)
 *             -------------
 *              1 + e^(-2t)
 */
public class HyperbolicTangent implements IActivationFunction
{
    public HyperbolicTangent() {
    }
    
    /**
     * Calculate the HyperbolicTangent value for every element in the specified 
     * matrix.
     * 
     * @param input the DoubleMatrix to use as input
     * @return the tanh value of the input matrix
     */
    public DoubleMatrix apply(DoubleMatrix input) {
        return MatrixFunctions.tanh(input);
    }
    
    /**
     * Computes the gradient of the hyperbolic tangent function with the 
     * specified input. The gradient is simply:
     * 
     *           1
     *     -------------
     *     cosh^2(input)
     *     
     *  Where cosh is the hyperbolic cosine.
     *  
     *  @param input the DoubleMatrix to use as input
     *  @return the gradient of the hyperbolic tangent
     */
    public DoubleMatrix gradient(DoubleMatrix input) {
        DoubleMatrix result = MatrixFunctions.cosh(input);
        result = result.muli(result);
        DoubleMatrix ones = DoubleMatrix.ones(input.rows, input.columns);
        return ones.divi(result);
    }

    /**
     * Calculate the Sigmoid value for a single double.
     * 
     * @param input the double to use as input
     * @return the sigmoid value of the input
     */
    public double apply(double input) {
        return MatrixFunctions.tanh(input);
    }
}
