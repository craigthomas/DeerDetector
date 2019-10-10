/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.commandline;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Parameters used for the training command.
 */
@Parameters(commandDescription="Trains a neural network")
public class TrainArguments
{
    @Parameter(names={"-b", "--heartbeat"}, description="specifies heartbeat during training")
    public Integer heartBeat = 100;

    @Parameter(names={"-l", "--learnrate"}, description="specifies learning rate")
    public Double learningRate = 0.01;

    @Parameter(names={"-c", "--csv"}, description="loads data from a CSV file")
    public String csvFile = "";

    @Parameter(names={"--color"}, description="processes images in color")
    public boolean color = false;

    @Parameter(names={"-p", "--positivedir"}, description="specifies positive image directory")
    public String positiveDir = "";

    @Parameter(names={"-n", "--negativedir"}, description="specifies negative image directory")
    public String negativeDir = "";

    @Parameter(names={"-w", "--width"}, description="ensure images have specified width in pixels")
    public Integer requiredWidth = 10;

    @Parameter(names={"-h", "--height"}, description="ensure images have specified height in pixels")
    public Integer requiredHeight = 10;

    @Parameter(names={"--savedir"}, description="save prediction results into specified directory")
    public String saveDir = "";

    @Parameter(names={"-s", "--split"}, description="splits the data between training and testing")
    public Integer split = 80;

    @Parameter(names={"-t", "--threshold"}, description="prediction threshold")
    public Double predictionThreshold = 0.5;

    @Parameter(names={"-f", "--folds"}, description="generate this many folds for cross-validation")
    public Integer folds = 1;

    @Parameter(names={"-l1", "--layer1neurons"}, description="specifies number of neurons in first hidden layer")
    public Integer layer1 = 10;

    @Parameter(names={"-l2", "--layer2neurons"}, description="specifies number of neurons in second hidden layer")
    public Integer layer2 = 0;

    @Parameter(names={"-o", "--outputneurons"}, description="specifies number of neurons in output layer")
    public Integer outputLayer = 1;

    @Parameter(names={"--lambda"}, description="specifies lambda value")
    public Double lambda = 1.0;

    @Parameter(names={"-i", "--iterations"}, description="number of iterations")
    public Integer iterations = 500;
}
