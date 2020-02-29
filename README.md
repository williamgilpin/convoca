# convoca

Demonstrate and learn cellular automata using convolutional neural networks in TensorFlow

![Game of Life training stages](demos/glider_stages.gif)

The video above shows different stages of training a network to learn Conway's Game of Life. 
This code can both implement and analyze known CA rulesets using TensorFlow, as well as learn the rules of unknown CA given image sequences as training data.
If you find this code useful, please consider citing the accompanying publication:

> Gilpin, William. "Cellular automata as convolutional neural networks." Physical Review E 100.3 (2019): 032402. [arXiv](https://arxiv.org/abs/1809.02942)

## Demos and features

The `demos.ipynb` illustrates a minimal example of training a CNN on the Game of Life, including example outputs.
Models are instantiated using the `initialize_model(...)` function, which builds a network with a trainable convolutional filter as the first layer, which serves to extract information about the neighborhood of each cell. Repeated 1x1 convolutions in subsequent layers implement the CA rules, and a final softmax layer assigns an output state to each cell.
For cases in which the CA ruleset is radially symmetric, the optional `SymmetricConvolution` layer imposes radially-symmetric structure on the learned convolutional kernels, which is often the case for natural systems with "totalistic" rules. An optional `Wraparound2D` layer also allows periodic boundary conditions to be implemented in the convolutions.

## Installation and Requirements

Install directly from GitHub using

	pip install git+git://github.com/williamgilpin/convoca

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


## Updates

As of 2.26.2020, the code has been significantly re-factored to use Tensorflow 2.0 and Keras.  The previous implementation has been placed in the "resources" directory, for reference.

## Planned future work

+ Add methods for simulating totalistic CA
+ Add methods for Moore neighborhood CA
+ Add demos recreating classic experiments, such as the results in Langton. Physica D, 1990.
+ Add statistical physics calculations such as an efficient calculation of "activity" for a CA
+ CA on graphs using an adjacency matrix --> grid convolutional operator