/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.commandline;

import com.beust.jcommander.JCommander;

/**
 * The Runner class parses the command line, and determines what actual command
 * to run. The current commands supported are:
 * 
 *  train - trains the neural network
 *  
 */
public class Runner
{
    public static final String TRAIN_COMMAND = "train";

    /**
     * Parse the command line options and execute the specified command.
     * 
     * @param argv the command line arguments
     */
    public static void main(String[] argv) {
        TrainArguments trainArguments = new TrainArguments();
        JCommander jCommander = JCommander.newBuilder()
                .addCommand(TRAIN_COMMAND, trainArguments)
                .build();
        jCommander.setProgramName("neuralnetwork");
        jCommander.parse(argv);
        String command = jCommander.getParsedCommand();

        if (command == null) {
            jCommander.usage();
        } else {
            switch (jCommander.getParsedCommand()) {
                case TRAIN_COMMAND:
                    TrainCommand tc = new TrainCommand(trainArguments);
                    tc.execute();
                    break;

                default:
                    jCommander.usage();
                    break;
            }
        }
    }
}
