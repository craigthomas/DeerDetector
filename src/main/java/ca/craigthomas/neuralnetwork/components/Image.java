/*
 * Copyright (C) 2014 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.components;

import java.awt.Color;
import java.awt.image.BufferedImage;

import org.jblas.DoubleMatrix;

import boofcv.core.image.ConvertImage;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.ImageUInt8;
import boofcv.struct.image.MultiSpectral;

/**
 * The Image class stores image information. Contains useful functions to 
 * transform and manipulate images.
 */
public class Image {

    private MultiSpectral<ImageUInt8> mImage;
    private final int sBufferedImageType;
    
    /**
     * Generates a new Image from a BufferedImage.
     * 
     * @param image the BufferedImage to generate from
     */
    public Image(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("image source cannot be null");
        }
        mImage = ConvertBufferedImage.convertFromMulti(image, null, true, ImageUInt8.class);
        sBufferedImageType = image.getType();
    }
    
    /**
     * Generates a new Image by loading in the image information from the 
     * specified filename.
     * 
     * @param filename the filename to load from
     */
    public Image(String filename) {
        this(UtilImageIO.loadImage(filename));
    }
    
    /**
     * Creates a grayscale image from a column vector. Assumes values are in
     * the range of 0 - 1.
     * 
     * @param imageData the pixel intensity values
     * @param width the width of the image in pixels
     * @param height the height of the image in pixels
     */
    public Image(DoubleMatrix imageData, int width, int height, boolean color) {
        if (color) {
            mImage = new MultiSpectral<>(ImageUInt8.class, width, height, 3);
            int counter = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int red = (int)(imageData.get(0, counter) * 255.0);
                    int green = (int)(imageData.get(0, counter+1) * 255.0);
                    int blue = (int)(imageData.get(0, counter+2) * 255.0);
                    mImage.getBand(0).set(x, y, red);
                    mImage.getBand(1).set(x, y, green);
                    mImage.getBand(2).set(x, y, blue);
                    counter += 3;
                }
            }
            sBufferedImageType = BufferedImage.TYPE_INT_RGB;
        } else {
            ImageUInt8 grayscale = new ImageUInt8(width, height);
            int counter = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int intensity = (int)(imageData.get(0, counter) * 255.0);
                    grayscale.set(x, y, intensity);
                    counter++;
                }
            }
            BufferedImage bufferedImage = ConvertBufferedImage.convertTo(grayscale, null);
            mImage = ConvertBufferedImage.convertFromMulti(bufferedImage, null, true, ImageUInt8.class);
            sBufferedImageType = bufferedImage.getType();            
        }
    }
    
    /**
     * Gets the width of the image in pixels.
     * 
     * @return the width of the image in pixels
     */
    public int getWidth() {
        return mImage.getWidth();
    }
    
    /**
     * Gets the height of the image in pixels.
     * 
     * @return the height of the image in pixels
     */
    public int getHeight() {
        return mImage.getHeight();
    }
    
    /**
     * Converts the image to grayscale. Returns a new copy of the image in
     * grayscale format.
     * 
     * @return a grayscale copy of the image
     */
    public Image convertToGrayscale() {
        ImageUInt8 grayscale = ConvertImage.average(mImage, null);
        return new Image(ConvertBufferedImage.convertTo(grayscale, null));
    }
    
    /**
     * Converts the image pixel intensities into a single column vector. 
     * First converts the image into a grayscale picture.
     * 
     * @param truth whether this is a positive or negative example
     * @return a column vector of the pixel intensities
     */
    public DoubleMatrix convertGrayscaleToMatrix(double truth) {
        ImageUInt8 grayscale = ConvertImage.average(mImage, null);
        byte [] data = grayscale.getData();
        DoubleMatrix result = new DoubleMatrix(1, data.length + 1);
        for (int index = 0; index < data.length; index++) {
            result.put(0, index, (double)data[index]);
        }
        result.put(0, data.length, truth * 255.0);
        return result.divi(255.0);
    }
    
    /**
     * Converts an RGB image into a single column vector. There will be
     * 3 bands of color interleaved in the columns. The data for the first
     * pixel will be in columns 0, 1, 2 - corresponding to the red, green
     * and blue components respectively.
     * 
     * @param truth whether this is a positive or negative example
     * @return a column vector of the pixel intensities
     */
    public DoubleMatrix convertColorToMatrix(double truth) {
        byte [] red = mImage.getBand(0).getData();
        byte [] green = mImage.getBand(1).getData();
        byte [] blue = mImage.getBand(2).getData();
        DoubleMatrix result = new DoubleMatrix(1, (red.length * 3) + 1);
        int counter = 0;
        for (int index = 0; index < red.length; index++) {
            result.put(0, counter, (double)red[index]);
            result.put(0, counter+1, (double)green[index]);
            result.put(0, counter+2, (double)blue[index]);
            counter += 3;
        }
        result.put(0, result.columns - 1, truth * 255.0);
        return result.divi(255.0);
    }
    
    /**
     * Generates a new image, which will be based upon the bounding box
     * of the top-left and bottom-right coordinates.
     * 
     * @param top the top y coordinate
     * @param left the left x coordinate
     * @param bottom the bottom y coordinate
     * @param right the right x coordinate
     * @return a new sub-image of the original
     */
    public Image getSubImage(int left, int top, int right, int bottom) {
        int height = bottom - top;
        int width = right - left;
        BufferedImage newImage = new BufferedImage(width, height, sBufferedImageType);
        BufferedImage oldImage = this.getBufferedImage();
        for (int x = left; x < right; x++) {
            for (int y = top; y < bottom; y++) {
                newImage.setRGB(x-left, y-top, oldImage.getRGB(x, y));
            }
        }
        return new Image(newImage);
    }
    
    /**
     * Converts an image into a BufferedImage.
     * 
     * @param image the image data to convert
     * @return a new BufferedImage
     */
    private BufferedImage convertToBufferedImage(MultiSpectral<ImageUInt8> image) {
        BufferedImage result = new BufferedImage(mImage.getWidth(), mImage.getHeight(), sBufferedImageType);
        ConvertBufferedImage.convertTo(mImage, result, true);
        return result;
    }
    
    /**
     * Returns the BufferedImage that backs the actual Image.
     * 
     * @return the BufferedImage backing the Image
     */
    public BufferedImage getBufferedImage() {
        return convertToBufferedImage(mImage);
    }
    
    /**
     * Draws a bounding box around the specified coordinates in the specified
     * color. Returns a new copy of the image with the bounding box placed
     * on the image.
     * 
     * @param top the top y position of the box
     * @param left the left x position of the box
     * @param bottom the bottom y position of the box
     * @param right the right x position of the box
     * @param color the color to draw the bounding box in
     * @return a new copy of the image with the bounding box
     */
    public Image drawBoundingBox(int top, int left, int bottom, int right, Color color) {
        return null;
    }
}
