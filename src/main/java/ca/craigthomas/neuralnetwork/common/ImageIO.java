/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.common;

import ca.craigthomas.neuralnetwork.commandline.Runner;
import ca.craigthomas.neuralnetwork.components.Image;
import ca.craigthomas.neuralnetwork.components.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Contains functions for manipulating images.
 */
public class ImageIO
{
    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());

    /**
     * Saves an image to the specified directory. Returns true if the save
     * succeeded, false otherwise
     *
     * @param image the Image to save
     * @param path the path to save to
     * @param filename the filename to save to
     */
    public static boolean saveImage(Image image, File path, String filename) {
        File saveFile = new File(path, filename);
        try {
            javax.imageio.ImageIO.write(image.getBufferedImage(), "png", saveFile);
        } catch (IOException e) {
            LOGGER.severe("could not save file [" + saveFile.getAbsolutePath() + "]");
            return false;
        }
        return true;
    }

    /**
     * Load image data from a directory into a DataSet. Assumes that all files
     * in the specified directory are images. The truth value indicates whether
     * it is a positive or negative sample.
     *
     * @param directoryString the directory to load images from
     * @param truth whether the samples are positive or negative
     */
    public static void loadFromDirectory(String directoryString, int requiredWidth, int requiredHeight, boolean color, double truth, DataSet dataSet) {
        File directory = new File(directoryString);
        File [] files = directory.listFiles();
        if (files == null) {
            LOGGER.severe("no files in directory [" + directory + "]");
            return;
        }

        for (File file : files) {
            String filename = file.getAbsolutePath();
            Image image = new Image(filename);

            if (image.getWidth() != requiredWidth || image.getHeight() != requiredHeight) {
                LOGGER.warning("image [" + filename + "] not correct size, skipping (want " + requiredWidth + "x" + requiredHeight + ", got " + image.getWidth() + "x" + image.getHeight() + ")");
            } else {
                dataSet.addSample(color ? image.convertColorToMatrix(truth) : image.convertGrayscaleToMatrix(truth));
            }
        }
    }

}
