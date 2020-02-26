# convoca

Demonstrate and learn cellular automata using convolutional neural networks in TensorFlow

![Game of Life training stages](demos/glider_stages.gif)

The video above shows different stages of training a network to learn Conway's Game of Life. 
This code can both implement and analyze known CA rulesets using TensorFlow, as well as learn the rules of unknown CA given image sequences as training data.

If you find this code useful, please consider citing the accompanying publication:

> Gilpin, William. "Cellular automata as convolutional neural networks." Physical Review E 100.3 (2019): 032402. [arXiv](https://arxiv.org/abs/1809.02942)


## Installation and Requirements

Typical installation with Miniconda. This code has been tested on macOS and Ubuntu.

+ Python >3.4
+ TensorFlow >2.0
+ numpy
+ matplotlib
+ Jupyter notebooks (for demos)

## Structure

The package contains the following libraries

`train_ca` : requires TensorFlow

`ca_funcs` : requires TensorFlow

`utils` : minor functions that support the main methods. Requires numpy only.

`demos.ipynb` : demonstration of the code for learning the game of live


## Demos

The `demos.ipynb` illustrates a minimal example of training a CNN on the Game of Life, including example outputs.

## Updates

As of 2.26.2020, the code has been significantly re-factored to use Tensorflow 2.0 and Keras.  The previous implementation has been placed in the "resources" directory, for reference.

## Planned future work

+ Add methods for simulating totalistic CA
+ Add methods for Moore neighborhood CA
+ Add demos recreating classic experiments, such as the results in Langton. Physica D, 1990.
+ Add statistical physics calculations such as an efficient calculation of "activity" for a CA
+ CA on graphs using an adjacency matrix --> grid convolutional operator