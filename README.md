# convca

Demonstrate and learn cellular automata using convolutional neural networks in TensorFlow

This code is associated with the ArXiv preprint:
Gilpin, William. "Cellular automata as convolutional neural networks" 2018.

For now, code is only in archival form for testing and analysis; future versions of this repository will significantly re-factor code into a general-purpose tool for cellular automaton analysis. All versions until a 1.0/PyPI release are thus tentative.

## Structure

The package contains the following libraries

`train_ca` : requires TensorFlow
`ca_funcs` : requires numpy only
`utils` : minor functions that support the main methods. Requires numpy only.

## Requirements

+ Python >3.4
+ TensorFlow
+ numpy
+ matplotlib
+ Jupyter notebooks (for demos)




# To Do

+ Add methods for simulating totalistic CA
+ Add methods for Moore neighborhood CA
+ Add demos recreating classic experiments, such as the results in Langton. Physica D, 1990.
+ Add statistical physics calculations such as an efficient calculation of "activity" for a CA
+ CA on graphs using an adjacency matrix --> grid convolutional operator