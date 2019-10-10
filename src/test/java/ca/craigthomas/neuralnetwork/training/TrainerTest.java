/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.training;

import static org.junit.Assert.*;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import ca.craigthomas.neuralnetwork.components.DataSet;
import ca.craigthomas.neuralnetwork.activationfunctions.HyperbolicTangent;
import ca.craigthomas.neuralnetwork.activationfunctions.IActivationFunction;
import ca.craigthomas.neuralnetwork.components.NeuralNetwork;

public class TrainerTest
{
    private Trainer trainer;
    private List<Integer> layerSizes;

    @Test
    public void testTrainerLearnNOTFunction() {
        Random random = new Random();
        layerSizes = Arrays.asList(1, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 1);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                inputs.put(index, 0, 0.0);
                outputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 1.0);
                outputs.put(index, 0, 0.0);                
            }
        }

        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(10000).heartBeat(0)
                .lambda(1.0).build();
        trainer.train();

        NeuralNetwork network = trainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                testInputs.put(index, 0, 1.0);
                testOutputs.put(index, 0, 0.0);
            } else {
                testInputs.put(index, 0, 0.0);
                testOutputs.put(index, 0, 1.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }
    
    @Test
    // Test assumes that DataSet is working correctly!
    public void testTrainerLearnNOTFunctionWithDataSet() {
        Random random = new Random();
        layerSizes = Arrays.asList(1, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 1);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                inputs.put(index, 0, 0.0);
                outputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 1.0);
                outputs.put(index, 0, 0.0);                
            }
        }
        
        DataSet dataSet = new DataSet(true, inputs, outputs);

        trainer = new Trainer.Builder(layerSizes, dataSet)
                .learningRate(0.001).maxIterations(10000).heartBeat(0)
                .lambda(1.0).build();
        trainer.train();

        NeuralNetwork network = trainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value = (double)random.nextInt(100) + 1;
            if (value > 50.0) {
                testInputs.put(index, 0, 1.0);
                testOutputs.put(index, 0, 0.0);
            } else {
                testInputs.put(index, 0, 0.0);
                testOutputs.put(index, 0, 1.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }

    @Test
    public void testTrainerLearnANDFunction() {
        Random random = new Random();
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 2);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 && value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }

        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(20000).heartBeat(0)
                .lambda(1.0).build();
        trainer.train();

        NeuralNetwork network = trainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                testInputs.put(index, 0, 1.0);
            } else {
                testInputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                testInputs.put(index, 1, 1.0);
            } else {
                testInputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 && value2 > 50.0) {
                testOutputs.put(index, 0, 1.0);
            } else {
                testOutputs.put(index, 0, 0.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }
    
    @Test
    public void testTrainerLearnORFunction() {
        Random random = new Random();
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        DoubleMatrix testInputs = DoubleMatrix.ones(10, 2);
        DoubleMatrix testOutputs = DoubleMatrix.ones(10, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }

        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
                .learningRate(0.001).maxIterations(15000).heartBeat(0)
                .lambda(1.0).build();
        trainer.train();

        NeuralNetwork network = trainer.getNeuralNetwork();
        for (int index = 0; index < 10; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                testInputs.put(index, 0, 1.0);
            } else {
                testInputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                testInputs.put(index, 1, 1.0);
            } else {
                testInputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                testOutputs.put(index, 0, 1.0);
            } else {
                testOutputs.put(index, 0, 0.0);
            }
        }
        DoubleMatrix predictions = network.predict(testInputs);
        Assert.assertArrayEquals(testOutputs.toArray(), predictions.toArray(), 0.15);
    }
    
    @Test
    public void testActivationFunctionSentToNeuralNetwork() {
        layerSizes = Arrays.asList(2, 1);
        IActivationFunction activationFunction = new HyperbolicTangent();
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        
        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
        .learningRate(0.001).maxIterations(0).heartBeat(0)
        .activationFunction(activationFunction).build();
        trainer.train();

        assertEquals(activationFunction, trainer.getNeuralNetwork().getActivationFunction());
    }
    
    @Test
    public void testRecordCostsRecordsAllIterations() {
        Random random = new Random();
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }
        
        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
        .learningRate(0.001).maxIterations(200).heartBeat(0)
        .recordCosts().build();
        trainer.train();
        
        List<Double> costs = trainer.getCosts();

        assertEquals(200, costs.size());
        for (Double cost : costs) {
            assertTrue(cost > 0.0);
        }
    }
    
    @Test
    public void testHeartbeatOutputToConsole() {
        Random random = new Random();
        layerSizes = Arrays.asList(2, 1);
        DoubleMatrix inputs = DoubleMatrix.ones(500, 2);
        DoubleMatrix outputs = DoubleMatrix.ones(500, 1);
        
        for (int index = 0; index < 500; index++) {
            double value1 = (double)random.nextInt(100) + 1;
            double value2 = (double)random.nextInt(100) + 1;
            
            if (value1 > 50.0) {
                inputs.put(index, 0, 1.0);
            } else {
                inputs.put(index, 0, 0.0);
            }
            
            if (value2 > 50.0) {
                inputs.put(index, 1, 1.0);
            } else {
                inputs.put(index, 1, 0.0);
            }
            
            if (value1 > 50.0 || value2 > 50.0) {
                outputs.put(index, 0, 1.0);
            } else {
                outputs.put(index, 0, 0.0);
            }
        }
        
        ByteArrayOutputStream stdOut = new ByteArrayOutputStream();
        System.setOut(new PrintStream(stdOut));
        
        trainer = new Trainer.Builder(layerSizes, inputs, outputs)
        .learningRate(0.001).maxIterations(200).heartBeat(1)
        .recordCosts().build();
        trainer.train();
        
        String standardOut = stdOut.toString();
        String [] strings = standardOut.split("\\n");
        assertEquals(200, strings.length);
        for (int i = 0; i < strings.length; i++) {
            assertTrue(strings[i].contains("Iteration: " + (i+1)));
        }
    }
}
