/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.commandline;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ca.craigthomas.neuralnetwork.components.Image;
import ca.craigthomas.neuralnetwork.components.NeuralNetwork;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.stat.StatUtils;
import org.jblas.DoubleMatrix;

import ca.craigthomas.neuralnetwork.components.DataSet;
import ca.craigthomas.neuralnetwork.components.Prediction;
import ca.craigthomas.neuralnetwork.training.Trainer;

import static ca.craigthomas.neuralnetwork.common.ImageIO.saveImage;
import static ca.craigthomas.neuralnetwork.common.ImageIO.loadFromDirectory;

/**
 * The TrainCommand is used to train a neural network based upon a number of
 * positive and negative examples. 
 */
public class TrainCommand
{
    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());
    // The underlying data set
    private DataSet dataSet;
    // The args passed to the command
    private TrainArguments args;
    
    public TrainCommand(TrainArguments arguments) {
        this.args = arguments;
    }
    
    /**
     * Load the data from a CSV file.
     */
    public void loadFromCSV() {
        dataSet = new DataSet(true);
        dataSet.addFromCSVFile(args.csvFile);
        LOGGER.info("loaded " + dataSet.getNumSamples() + " sample(s)");
    }
    
    public void saveResults(NeuralNetwork bestModel, DataSet bestFold) {
        File directory = new File(args.saveDir);
        if (!directory.isDirectory()) {
            LOGGER.log(Level.SEVERE, "save directory [" + args.saveDir + "] is not a directory");
            return;
        }
        
        Prediction predictions = new Prediction(bestModel, args.predictionThreshold);
        predictions.predict(bestFold);
        DoubleMatrix falsePositives = predictions.getFalsePositiveSamples();
        DoubleMatrix falseNegatives = predictions.getFalseNegativeSamples();
        for (int i = 0; i < falsePositives.rows; i++) {
            Image image = new Image(falsePositives.getRow(i), args.requiredWidth, args.requiredHeight, args.color);
            saveImage(image, directory, "fp" + (i+1) + ".png");
        }
        for (int i = 0; i < falseNegatives.rows; i++) {
            Image image = new Image(falseNegatives.getRow(i), args.requiredWidth, args.requiredHeight, args.color);
            saveImage(image, directory, "fn" + (i+1) + ".png");
        }
    }
    
    public void execute() {
        NeuralNetwork bestModel = null;
        DataSet bestFold = null;
        double [] tp = new double [args.folds];
        double [] fp = new double [args.folds];
        double [] tn = new double [args.folds];
        double [] fn = new double [args.folds];
        double [] precision = new double [args.folds];
        double [] recall = new double [args.folds];
        double [] f1 = new double [args.folds];
        double bestF1 = 0;
        
        // Step 1: load the data sets from files or CSV
        dataSet = new DataSet(true);
        if (!args.csvFile.isEmpty()) {
            loadFromCSV();
        } else {
            loadFromDirectory(args.positiveDir, args.requiredWidth, args.requiredHeight, args.color, 1.0, dataSet);
            loadFromDirectory(args.negativeDir, args.requiredWidth, args.requiredHeight, args.color, 0.0, dataSet);
        }
        
        if (dataSet == null) {
            LOGGER.log(Level.SEVERE, "no data set could be built, exiting");
            return;
        }
        
        // Step 2: Generate layer information
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(dataSet.getNumColsSamples());
        if (args.layer1 != 0) {
            layerSizes.add(args.layer1);
        }
        if (args.layer2 != 0) {
            layerSizes.add(args.layer2);
        }
        layerSizes.add(args.outputLayer);
        
        // Step 3: generate the folds and train the model
        for (int fold = 0; fold < args.folds; fold++) {
            LOGGER.log(Level.INFO, "processing fold " + (fold+1));
            LOGGER.log(Level.INFO, "randomizing components");
            dataSet.randomize();
            LOGGER.log(Level.INFO, "generating training and testing sets");
            Pair<DataSet, DataSet> split = dataSet.splitEqually(args.split);
            DataSet trainingData = split.getLeft();
            DataSet testingData = split.getRight();
            LOGGER.log(Level.INFO, "training neural network...");   
            trainingData.randomize();
            Trainer trainer = new Trainer.Builder(layerSizes, trainingData)
                    .maxIterations(args.iterations)
                    .heartBeat(args.heartBeat)
                    .learningRate(args.learningRate)
                    .lambda(args.lambda).build();
            trainer.train();
            
            // Step 4: evaluate each model
            NeuralNetwork model = trainer.getNeuralNetwork();
            Prediction prediction = new Prediction(model, args.predictionThreshold);
            prediction.predict(testingData);
            System.out.println("True Positives " + prediction.getTruePositives());
            System.out.println("False Positives " + prediction.getFalsePositives());
            System.out.println("True Negatives " + prediction.getTrueNegatives());
            System.out.println("False Negatives " + prediction.getFalseNegatives());
            System.out.println("Precision " + prediction.getPrecision());
            System.out.println("Recall " + prediction.getRecall());
            System.out.println("F1 " + prediction.getF1());
            
            tp[fold] = prediction.getTruePositives();
            fp[fold] = prediction.getFalsePositives();
            tn[fold] = prediction.getTrueNegatives();
            fn[fold] = prediction.getFalseNegatives();
            precision[fold] = prediction.getPrecision();
            recall[fold] = prediction.getRecall();
            f1[fold] = prediction.getF1();
            if (f1[fold] > bestF1) {
                bestModel = model;
                bestFold = dataSet.dup();
                bestF1 = f1[fold];
            }
        }
        
        // Step 6: save the best information to the specified directory
        if (!args.saveDir.isEmpty()) {
            saveResults(bestModel, bestFold);
        }
        
        // Step 5: compute the overall statistics
        System.out.println("Overall Statistics");
        System.out.println("True Positives " + StatUtils.mean(tp) + " (" + StatUtils.variance(tp) + ")");
        System.out.println("False Positives " + StatUtils.mean(fp) + " (" + StatUtils.variance(fp) + ")");
        System.out.println("True Negatives " + StatUtils.mean(tn) + " (" + StatUtils.variance(tn) + ")");
        System.out.println("False Negatives " + StatUtils.mean(fn) + " (" + StatUtils.variance(fn) + ")");
        System.out.println("Precision " + StatUtils.mean(precision) + " (" + StatUtils.variance(precision) + ")");
        System.out.println("Recall " + StatUtils.mean(recall) + " (" + StatUtils.variance(recall) + ")");
        System.out.println("F1 " + StatUtils.mean(f1) + " (" + StatUtils.variance(f1) + ")");
    }
}
