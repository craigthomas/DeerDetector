# Neural Network 

[![Build Status](https://travis-ci.org/craigthomas/NeuralNetwork.svg?branch=master)](https://travis-ci.org/craigthomas/NeuralNetwork) 
[![Coverage Status](https://codecov.io/gh/craigthomas/NeuralNetwork/branch/master/graph/badge.svg)](https://codecov.io/gh/craigthomas/NeuralNetwork)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8a03ba66560d42a6b64118240b1615f9)](https://www.codacy.com/app/craig-thomas/NeuralNetwork?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=craigthomas/NeuralNetwork&amp;utm_campaign=Badge_Grade)


## What is it?

This project implements a general purpose Neural Network. This particular
implementation is used specifically for image recognition, however, it 
can be applied to many other machine learning problems. 

In the context of image recognition, the Neural Network was originally
built to determine if a deer (or other wildlife) is in a photo. To make 
it work, you must first `train` the network to recognize deer or other objects. Once
trained, you can then feed the neural network other photos, and it will 
attempt to find instances of the objects it knows about within those 
photos.

For more information on the Deer Detector goal, [read my blog posts on the
project](http://craigthomas.ca/blog/2014/08/04/deer-detection-with-machine-learning-part-1/)

Note: some code has recently moved around. The code to take pictures with a Kinect
still exists, but has moved to a [new GitHub repo](https://github.com/craigthomas/KinectTimeLapse).
To take pictures with a Raspberry Pi NOIR camera, see my [other GitHub repo](https://github.com/craigthomas/RPiDayNightCamera).


## License

This project makes use of an MIT style license. Please see the file called 
LICENSE for more information. Note that this project may make use of other
software that has separate license terms. See the section called `Third
Party Licenses and Attributions` below for more information on those
software components.


## Compiling

Simply copy the source files to a directory of your choice. In addition
to the source, you will need the following required software packages:

* [Java JDK 7](http://www.oracle.com/technetwork/java/javase/downloads/index.html) 1.7.0 u51 or later
* [OpenKinect libfreenect](http://openkinect.org/wiki/Getting_Started)

To build the project, switch to the root of the source directory, and
type:

    ./gradlew build

On Windows, switch to the root of the source directory, and type:

    gradlew.bat build

The compiled Jar file will be placed in the `build/libs` directory.


## Running

### Command Line Help

There are several options available for the program. Running the jar
with the `-h` option will display a helpful description of the options:

    java -jar build/libs/visualclassifier-0.1.jar -h


### Training the NeuralNetwork

The `train` sub-command allows you to train a neural network with a group
of images contained within a set of directories. The positive examples must
be contained in one directory, and the negative examples must be contained
within another. The images can be in any valid image format. They must all 
be the same width and height. For training, you must also split the dataset
into a training / testing fold. Assuming that images are 60 pixels by 60 pixels,
and you wish to use 80% of the data set for training, and 20% for testing:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80

#### Learning Rate

You can also set the learning rate with `-l`:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -l 0.001

#### Iterations

By default, the network trains for 500 iterations. You can change that with the
`-i` option:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -i 1000

#### Heartbeat

During training, you can have the network output periodic messages containing
the cost for that iteration. This is known as a heartbeat. The heartbeat will
display information after the specified number of iterations with the `-h`
option. To display after 100 iterations:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -h 100

#### Cross Validation

You can also use k-fold cross validation. You can specify the number of folds to
randomly be built from the dataset with the `-f` option. For example, to perform
10-fold cross validation:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -f 10

#### Layer Configuration

You can specify the number of nodes (neurons) to use in each layer of the network,
up to a maximum of 2 hidden layers with `-l1` for layer 1, and `-l2` for layer 2.
For example, to create a network with a hidden layer containing 20 nodes:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -l1 20

#### Prediction Threshold

When making predictions, the network uses a threshold of 0.5. This means that 
values over 0.5 will be predicted as positive, and under 0.5 will be predicted
as negative. You can change the threshold with the `-t` option:

    java -jar build/libs/visualclassifier-0.1.jar train -p /path/to/positives \
         -n /path/to/negatives -w 60 -h 60 -s 80 -t 0.7

#### False Positives and Negatives

You can also save the false positive and false negative images to a sub-directory
with the `--save` option. The directory must exist, and must be writable. Images
will take on the name `fp` for False Positive, and `fn` for False Negative.


## Current Status - September 15, 2014

### Operational

- The neural network classifier
- The ability to train the neural network classifier
- The neural network predictor

### Yet to be Implemented

- Saving the generated model to disk for future use
- Scanning an image for instances of the object


## Third Party Licenses and Attributions

### args4j

The project makes use of the args4j command-line argument parser,
which is available at https://github.com/kohsuke/args4j and is
licensed under the following terms:

    Copyright (c) 2013 Kohsuke Kawaguchi and other contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
 
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### JBlas

This project makes use of the jblas linear algebra library, which is
available at https://github.com/mikiobraun/jblas and is licensed under 
the following terms:

    Copyright (c) 2009, Mikio L. Braun and contributors
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
    
        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.
    
        * Neither the name of the Technische Universität Berlin nor the
          names of its contributors may be used to endorse or promote
          products derived from this software without specific prior
          written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### BoofCV 

This project makes use of BoofCV, which is licensed under the Apache License,
Version 2.0. The license can be downloaded from 
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from https://github.com/lessthanoptimal/BoofCV

### Apache Commons CSV

This project makes use of the Apache Commons CSV, which is licensed under the
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from http://commons.apache.org/proper/commons-csv/

### Apache Commons IO

This project makes use of the Apache Commons IO, which is licensed under the
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from http://commons.apache.org/proper/commons-io/

### Apache Commons Lang

This project makes use of the Apache Commons Lang, which is licensed under the
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from http://commons.apache.org/proper/commons-lang/

### Apache Commons Math

This project makes use of the Apache Commons Math, which is licensed under the
Apache License, Version 2.0. The license can be downloaded from
http://www.apache.org/licenses/LICENSE-2.0.txt. The source code for this
software is available from http://commons.apache.org/proper/commons-math/
