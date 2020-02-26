import tensorflow as tf
import numpy as np

import collections

def periodic_padding(imbatch, padding=1):
    '''
    Create a periodic padding (wrap) around an image batch, to emulate 
    periodic boundary conditions. Padding occurs along the middle two axes
    '''
    pad_u = imbatch[:, -padding:, :]
    pad_b = imbatch[:, :padding, :]

    partial_image = tf.concat([pad_u, imbatch, pad_b], axis=1)

    pad_l = partial_image[..., -padding:, :]
    pad_r = partial_image[..., :padding, :]

    padded_imbatch = tf.concat([pad_l, partial_image, pad_r], axis=2)
          
    
    return padded_imbatch

class Wraparound2D(tf.keras.layers.Layer):
    """
    Apply periodic boundary conditions on an image by padding 
    along the axes
    padding : int or tuple, the amount to wrap around    
    """

    def __init__(self, padding=2, **kwargs):
        super(Wraparound2D, self).__init__()
        self.padding = padding
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size': 0,
            'num_layers': 1,
            'units': 0,
            'dropout': 0,
        })
        return config
    
    def call(self, inputs):
        return periodic_padding(inputs, self.padding)
    
def initialize_model(shape, layer_dims, nhood=1, num_classes=2):
    """
    Given a domain size and layer specification, initialize a model that assigns
    each pixel a class
    shape : the horizontal and vertical dimensions of the CA image
    layer_dims : list of number of hidden units per layer
    num_classes : int, the number of output classes for the automaton
    """
    wspan, hspan = shape
    diameter = 2*nhood+1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((wspan, hspan, 1)))
    model.add(Wraparound2D(padding=nhood))
    model.add(tf.keras.layers.Conv2D(layer_dims[0], kernel_size=[diameter, diameter], padding='valid', activation='relu',
                                    kernel_initializer=tf.keras.initializers.he_normal(), 
                                     bias_initializer=tf.keras.initializers.he_normal()))
    model.add(tf.keras.layers.Reshape(target_shape=(-1, layer_dims[0])))

    for i in range(1, len(layer_dims)):
        model.add(tf.keras.layers.Dense(layer_dims[i],  activation='relu',
                                        kernel_initializer=tf.keras.initializers.he_normal(), 
                                        bias_initializer=tf.keras.initializers.he_normal()))
    model.add(tf.keras.layers.Dense(num_classes,  activation='relu',
                                    kernel_initializer=tf.keras.initializers.he_normal(), 
                                    bias_initializer=tf.keras.initializers.he_normal()))
    #model.add(tf.keras.layers.Reshape(target_shape=(-1, wspan, hspan)))
    return model


def logit_to_pred(logits, shape=None):
    """
    Given logits in the form of a network output, convert them to 
    images
    """
    
    labels = tf.argmax(tf.nn.softmax(logits), 
                                axis=-1), 
    if shape:                 
        out = tf.reshape(labels, shape)
    return out


