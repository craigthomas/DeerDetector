package ca.craigthomas.neuralnetwork.components;

import static org.junit.Assert.*;

import java.awt.Color;
import java.awt.image.BufferedImage;

import ca.craigthomas.neuralnetwork.components.Image;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class ImageTest {

    private BufferedImage mBufferedImage;
    private Image mImage;
    private Color red;
    private Color blue;
    private Color green;
    
    @Before
    public void setUp() {
        mBufferedImage = new BufferedImage(3, 1, BufferedImage.TYPE_INT_RGB);
        red = new Color(255, 0, 0);
        green = new Color(0, 255, 0);
        blue = new Color(0, 0, 255);
        mBufferedImage.setRGB(0, 0, red.getRGB());
        mBufferedImage.setRGB(1, 0, green.getRGB());
        mBufferedImage.setRGB(2, 0, blue.getRGB());
    }
    
    @Test
    public void testGetWidthAndHeightWorkCorrectly() {
        mImage = new Image(mBufferedImage);
        assertEquals(3, mImage.getWidth());
        assertEquals(1, mImage.getHeight());
    }
    
    @Test
    public void testImageDataLoadedCorrectly() {
        mImage = new Image(mBufferedImage);
        BufferedImage result = mImage.getBufferedImage();
        assertEquals(red.getRGB(), result.getRGB(0, 0));
        assertEquals(green.getRGB(), result.getRGB(1, 0));
        assertEquals(blue.getRGB(), result.getRGB(2, 0));
    }
    
    @Test
    public void testConvertToGrayscaleWorksCorrectly() {
        mImage = new Image(mBufferedImage);
        Image grayscale = mImage.convertToGrayscale();
        BufferedImage result = grayscale.getBufferedImage();
        Color gray = new Color(85, 85, 85);
        assertEquals(gray.getRGB(), result.getRGB(0, 0));
        assertEquals(gray.getRGB(), result.getRGB(1, 0));
        assertEquals(gray.getRGB(), result.getRGB(2, 0));
    }
    
    @Test
    public void testConvertGrayscaleToMatrixWorksCorrectly() {
        DoubleMatrix expected = new DoubleMatrix(new double [][] {
                {0.3333, 0.3333, 0.3333, 1.0}
        });
        mImage = new Image(mBufferedImage);
        DoubleMatrix result = mImage.convertGrayscaleToMatrix(1.0);
        Assert.assertArrayEquals(expected.toArray(), result.toArray(), 0.0001);
    }
    
    @Test
    public void testGetSubImageGetsCorrectly() {
        mImage = new Image(mBufferedImage);
        Image resultImage = mImage.getSubImage(1, 0, 2, 1);
        BufferedImage result = resultImage.getBufferedImage();
        assertEquals(green.getRGB(), result.getRGB(0, 0));
        
        resultImage = mImage.getSubImage(0, 0, 1, 1);
        result = resultImage.getBufferedImage();
        assertEquals(red.getRGB(), result.getRGB(0, 0));

        resultImage = mImage.getSubImage(2, 0, 3, 1);
        result = resultImage.getBufferedImage();
        assertEquals(blue.getRGB(), result.getRGB(0, 0));
    }
    
    @Test (expected=IllegalArgumentException.class)
    public void testCreateWithNullThrowsException() {
        mImage = new Image((BufferedImage)null);
    }
}
