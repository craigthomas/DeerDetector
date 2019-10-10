/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;

/**
 * An interface to capture an activation function. There are several different
 * activation functions that one might wish to use within a neural network. At
 * a minimum, each activation function can be applied to a particular matrix,
 * as well as return the gradient for that function along a given input.
 */
public interface IActivationFunction
{
    DoubleMatrix apply(DoubleMatrix input);
    
    DoubleMatrix gradient(DoubleMatrix input);
    
    double apply(double input);
}
