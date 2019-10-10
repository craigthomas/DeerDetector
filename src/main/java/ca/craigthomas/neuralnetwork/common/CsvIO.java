/*
 * Copyright (C) 2014-2018 Craig Thomas
 * This project uses an MIT style license - see LICENSE for details.
 */
package ca.craigthomas.neuralnetwork.common;

import ca.craigthomas.neuralnetwork.commandline.Runner;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Common utilities for reading and writing CSV files.
 */
public class CsvIO
{
    // The logger for the class
    private final static Logger LOGGER = Logger.getLogger(Runner.class.getName());

    /**
     * Read from a CSV file, and return the samples as a list of doubles.
     *
     * @param filename the name of the file to read from
     * @return the list of samples from the file
     */
    public static List<List<Double>> readCSVFile(String filename) {
        try {
            File file = new File(filename);
            String fileContents = FileUtils.readFileToString(file);

            Reader reader = new StringReader(fileContents);
            CSVFormat format = CSVFormat.EXCEL;
            CSVParser parser = new CSVParser(reader, format);

            List<CSVRecord> records = parser.getRecords();
            List<List<Double>> inputs = new ArrayList<>();

            for (CSVRecord record : records) {
                List<Double> inputLine = new ArrayList<>();
                for (int index = 0; index < record.size(); index++) {
                    String value = record.get(index);
                    inputLine.add(Double.parseDouble(value));
                }
                inputs.add(inputLine);
            }
            parser.close();
            return inputs;
        } catch (IOException e) {
            LOGGER.severe("error parsing CSV file [" + filename + "] - " + e.getLocalizedMessage());
            return null;
        }
    }
}
