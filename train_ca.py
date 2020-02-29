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
    
def initialize_model(shape, layer_dims, nhood=1, num_classes=2, totalistic=False, 
                      nhood_type="moore", bc="periodic"):
    """
    Given a domain size and layer specification, initialize a model that assigns
    each pixel a class
    shape : the horizontal and vertical dimensions of the CA image
    layer_dims : list of number of hidden units per layer
    num_classes : int, the number of output classes for the automaton
    totalistic : bool, whether to assume that the CA is radially symmetric, making
        it outer totalistic
    nhood_type : string, default "moore". The type of neighborhood to use for the 
        CA. Currently, the only other option, "Neumann," only works when "totalistic"
        is set to True
    bc : string, whether to use "periodic" or "constant" (zero padded) boundary conditions
    """
    wspan, hspan = shape
    diameter = 2*nhood+1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((wspan, hspan, 1)))
    
    if bc == "periodic":
        model.add(Wraparound2D(padding=nhood))
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    
    if totalistic:
        model.add(SymmetricConvolution(nhood, n_type=nhood_type, bc=bc))
        model.add(tf.keras.layers.Reshape(target_shape=(-1, nhood+1)))
    else:
        model.add(tf.keras.layers.Conv2D(layer_dims[0], kernel_size=[diameter, diameter], padding=conv_pad, 
                                         activation='relu', kernel_initializer=tf.keras.initializers.he_normal(), 
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

def augment_data(x, y, n=None):
    """
    Generate an augmented training dataset with random reflections
    and 90 degree rotations
    x, y : Image sets of shape (Samples, Width, Height, Channels) 
        training images and next images
    n : number of training examples
    """
    n_data = x.shape[0]
    
    if not n:
        n = n_data
    x_out, y_out = list(), list()
    
    for i in range(n):
        r = tf.random.uniform((1,), minval=0, maxval=n_data, dtype=tf.int32)[0]
        x_r, y_r = x[r], y[r]
        
        if tf.random.uniform((1,))[0]<0.5:
            x_r = tf.image.flip_left_right(x_r)
            y_r = tf.image.flip_left_right(y_r)
        if tf.random.uniform((1,))[0]<0.5:
            x_r = tf.image.flip_up_down(x_r)
            y_r = tf.image.flip_up_down(y_r)
            
        num_rots = tf.random.uniform((1,), minval=0, maxval=4, dtype=tf.int32)[0]
        x_r = tf.image.rot90(x_r, k=num_rots)
        y_r = tf.image.rot90(y_r, k=num_rots)
        
        x_out.append(x_r), y_out.append(y_r)
    return tf.stack(x_out), tf.stack(y_out)
        
    
    

def make_square_filters(rad):
    """
    rad : the pixel radius for the filters
    """
    m = 2*rad + 1
    square_filters = tf.stack([tf.pad(tf.ones([i, i]), [[int((m-i)/2), int((m-i)/2)], 
                                                        [int((m-i)/2), int((m-i)/2)]]) 
                               for i in range(1, m+1, 2)])
    square_filters = [square_filters[0]] + [item for item in square_filters[1:] - square_filters[:-1]]
    square_filters = tf.stack(square_filters)[..., tf.newaxis]
    
    return square_filters

def make_circular_filters(rad):
    """
    rad : the pixel radius for the filters
    """
    
    m = 2*rad + 1

    qq = tf.range(m) - int((m-1)/2)
    pp = tf.sqrt(tf.cast(qq[..., None]**2 + qq[None, ...]**2, tf.float32))

    val_range = tf.cast(tf.range((m+1)/2), tf.float32)
    circ_filters = make_square_filters(rad)*val_range[..., None, None, None]
    rr = circ_filters*(1/pp)[None, ..., None]
    rr = tf.where(tf.math.is_nan(rr), tf.zeros_like(rr), rr)
    return tf.stack([make_square_filters(rad)[0]] + [item for item in rr][1:])

class SymmetricConvolution(tf.keras.layers.Layer):
    """
    A non-trainable convolutional layer that extracts the 
    summed values in the neighborhood of each pixel. No activation
    is applied because this feature extractor does not change during training
    parametrized by the radius
    r : int, the max neighborhood size
    nhood_type : "moore" (default) uses the Moore neighborhood, while "neumann"
        uses the generalized von Neumann neighborhood, which is similar 
        to a circle at large neighborhood radii
    bc : "periodic" or "constant"
    TODO : implement the "hard" von Neumann neighborhood
    """

    def __init__(self, r, nhood_type="moore", bc="periodic", **kwargs):
        super(SymmetricConvolution, self).__init__()
        
        self.r = r
        
        if nhood_type == "moore":
            filters = make_square_filters(r)
        elif nhood_type == "neumann":
            filters = make_circular_filters(r)
        else:
            filters = make_square_filters(r)
            warnings.warn("Neighborhood specification not recognized.")
        self.filters = tf.squeeze(tf.transpose(filters))[..., None, :]
        
        if bc == "periodic":
            self.pad_type="VALID"
        else:
            self.pad_type="SAME"
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': 1,
            'units': 0,
            'dropout': 0,
        })
        return config
    
    def call(self, inputs):
        return tf.nn.convolution(inputs, self.filters, padding=self.pad_type)