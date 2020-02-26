# import numpy as np
import tensorflow as tf
import numpy as np

import collections

from ca_funcs import periodic_padding, conv_cast, kaiming_normal


class ConvNet(object):
    """
    A convolutional neural network with one convolutional layer, 
    arbitrary hidden layers, and a summation for the final layer:

    conv - relu - Nx[affine - relu] - sum

    There are two ways to initialize this object: either by passing
    architecture parameters to an explicit constructor function, or by
    passing a string to a TensorFlow .ckpt file

    The lack of pooling or a final score prediction operation makes
    this approach similar to a "soft committee machine"

    The structure of this class is meant to mirror the standard implementation
    of the ConvNet used by CS231n, among others: http://cs231n.stanford.edu/index.html


    Dev notes: Would be wonderful to make this class Eager compatible
    once the training process for Eager is a little more smooth. For now, just use
    an InterativeSession() to reduce some boilerplate

    Dev notes: add doctests

    """

    def __init__(self, sess, ckpt_path='', input_dim=(10, 10), layer_dims=[10,10,10,10,10],
        weight_scale=1e-3, filter_size=3, pad_size=1, num_classes=2, strides=[1,1,1,1],
        reg=0.0, fix_layers=False):
        """
        Initialize a new network and define internal parameters.

        Inputs:
        
        sess : tf.Session(), tf.InteractiveSession()
        ckpt_path : str pointing to a TensorFlow checkpoint, 
                    if you want to initialize from a trained model
        input_dim: Tuple (H, W) giving size of the input array
        layer_dims: List of dimensions for each layer, with the first dimension denoting
                    the number of convolutional filters
        filter_size: Size of convolutional filters to use in the convolutional layer
        pad_size: the amount of padding to use for the boundary conditions
        num_classes: Number of scores to produce from the final affine layer.
        weight_scale: Weight amplitude for Kaiming normalization
        reg: Scalar for weight of L2 regularization during training
        fix_layers: bool for dealing with a bug where tensorflow sometimes saves the network
                    twice in one file

        Dev: 
        """

        self.sess = sess
        self.test = tf.constant('init successful', name='test') # for debugging scope

        ## General properties
        self.wspan, self.hspan = input_dim
        self.weight_scale = weight_scale
        self.pad_size = pad_size
        self.all_strides = strides
        self.filter_size = filter_size

        self.X = tf.placeholder(tf.float32, shape=input_dim) 
        self.y = tf.placeholder(tf.float32, shape=input_dim) 

        if ckpt_path:
            self.ckpt_loaded = True
            self.ckpt_path = ckpt_path
            meta_path = ckpt_path+'.meta'
            saver = tf.train.import_meta_graph(meta_path)

            all_vars = tf.trainable_variables()
            if fix_layers:
                all_vars = all_vars[:int(len(all_vars)/2)] # I do not understand why import_meta_graph does this
            all_var_shapes = [np.array(var.get_shape().as_list()) for var in all_vars]

            print(str(len(all_vars)/2) + ' layers detected')

            (self.filter_size, _, _, self.num_filters) = all_var_shapes[0] 

            self.layer_dims = np.squeeze(np.array(all_var_shapes[1::2])) 
            self.num_filters = self.layer_dims[0] 
            self.num_layers = len(self.layer_dims)
            self.num_hidden = self.num_layers - 1

            assert self.load_from_ckpt()

        else:
            self.ckpt_loaded = False
            self.layer_dims = layer_dims
            self.num_filters = self.layer_dims[0] 
            self.num_layers = len(self.layer_dims)
            self.num_hidden = self.num_layers - 1

            assert self.init_new_model()


        
        


    def init_new_model(self):
        """
        Initialize a new model when a .ckpt model is not been specified in the
        constructor
        """
        weight_scale = self.weight_scale
        filter_size = self.filter_size
        layer_dims = self.layer_dims

        self.conv_filters_params = {
            'Wfilt': tf.Variable(weight_scale*kaiming_normal([filter_size, filter_size, 1, self.num_filters]), 
                name='Wfilt'),
            'bfilt': tf.Variable(weight_scale*kaiming_normal([self.num_filters,]), name='bfilt'),
        }
        
        self.hidden_params = {}
        for ii in range(1, self.num_layers):
            wh_name = 'Wh'+str(ii)
            bh_name = 'bh'+str(ii)
            self.hidden_params[wh_name] = tf.Variable(weight_scale*kaiming_normal([layer_dims[ii-1],
                                                                                      layer_dims[ii]]), name=wh_name)
            self.hidden_params[bh_name] = tf.Variable(weight_scale*kaiming_normal([layer_dims[ii],]), name=bh_name)

        self.sess.run(tf.global_variables_initializer())

        return True

        
    def load_from_ckpt(self, no_names=False):
        """Load a TensorFlow checkpoint file
        no_names : bool
            If ckpt file doesn't have correct Tensor names,
             automatically determine the appropriate names based on the currently
             defined variables. This will fail cryptically if any variables in the
              initialized graph have a different shape than the initialized model.
              This usually should not be necessary unless the .meta file is missing
        """


        all_vars = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        self.conv_filters_params = {
            'Wfilt': all_vars[0],
            'bfilt': all_vars[1],
        }

        self.hidden_params = {}
        for ii in range(1, self.num_layers):
            wh_name = 'Wh'+str(ii)
            bh_name = 'bh'+str(ii)
            self.hidden_params[wh_name] = all_vars[2*ii]
                                          
            self.hidden_params[bh_name] = all_vars[2*ii+1]

        if no_names:
            all_vars = tf.trainable_variables()
            #var_names = [v.name for v in tf.trainable_variables()]

            blank_names = ['Variable_'+str(ind) for ind in range(len(all_vars))]
            blank_names[0] = 'Variable'

            name_dict = dict(zip(blank_names, all_vars))

            saver = tf.train.Saver(max_to_keep=None, var_list=name_dict)

        else: 
            saver = tf.train.Saver(max_to_keep=None)
        
        saver.restore(self.sess, self.ckpt_path)

        return True


    def tester(self):
        """
        For testing whether various methods were loaded correctly
        """
        print(self.sess.run(tf.report_uninitialized_variables()))
        print(self.sess.run(self.test))


    def ca_cnn(self, ic_tf):
        """
        Create the CNN using Tensorflow
        """
        num_hidden = self.num_hidden
        wspan, hspan = self.wspan, self.hspan
        pad_size = self.pad_size

        # # Expand and reshape initial conditions
        state_pad = periodic_padding(ic_tf, pad_size)
        current_state = tf.cast(tf.reshape(state_pad,[1, hspan+2*pad_size, wspan+2*pad_size, 1]), tf.float32)
        
        # First convolutional layer
        conv1 = tf.nn.conv2d(current_state, self.conv_filters_params['Wfilt'], 
            strides=self.all_strides, padding='VALID')
        conv1_b = tf.nn.bias_add(conv1,self.conv_filters_params['bfilt'])
        conv1_activated = tf.nn.relu(conv1_b)
        conv1_flat = tf.reshape(conv1_activated, [wspan*hspan, self.num_filters])

        # Cycle through the hidden layers
        curr = conv1_flat
        for ii in range(1, self.num_layers):
            neural_state = tf.nn.bias_add( tf.matmul(curr, self.hidden_params['Wh'+str(ii)]), 
                self.hidden_params['bh'+str(ii)])
            neural_state_activated = tf.nn.relu(neural_state)
            curr = neural_state_activated

        final_state = tf.reduce_sum(curr, axis=1)
        # # output layer is just a sum (soft committee) over final states
        out_layer = tf.reshape(final_state, (wspan, hspan))

        return out_layer

    def loss():
        """
        Compute the L2 loss
        """
        y = tf.placeholder(tf.float32, shape=(self.wspan, self.hspan))
        loss = tf.reduce_sum(tf.nn.l2_loss(self.ca_model - y))

    def ca_map(self, ic_tf0):
        """
        Show where the model above was activated
        Dev: this function needs to be parallelized
        """

        num_hidden = self.num_hidden
        wspan, hspan = self.wspan, self.hspan
        pad_size = self.pad_size
        
        
        ic_tf = conv_cast(ic_tf0)
        

        all_where_bools = list()
        
        # Expand and reshape initial conditions
        state_pad = periodic_padding(ic_tf, pad_size)
        current_state = tf.cast(tf.reshape(state_pad,[1, hspan+2*pad_size, wspan+2*pad_size, 1]), tf.float32)
        
        # First convolutional layer
        conv1 = tf.nn.conv2d(current_state, self.conv_filters_params['Wfilt'], 
            strides=self.all_strides, padding='VALID')
        conv1_b = tf.nn.bias_add(conv1, self.conv_filters_params['bfilt'])
        conv1_activated = tf.nn.relu(conv1_b)
        conv1_flat = tf.reshape(conv1_activated, [wspan*hspan, self.num_filters])
        
        where_bool1 = tf.greater(conv1_flat, 0)
        where_sum1 = tf.reduce_sum(tf.cast(where_bool1, tf.float32), axis=[0])
        all_where_bools.append(where_bool1)
        
        # Cycle through the hidden layers
        curr = conv1_flat
        for ii in range(1, 1+num_hidden):
            neural_state = tf.nn.bias_add( tf.matmul(curr, self.hidden_params['Wh'+str(ii)]), 
                self.hidden_params['bh'+str(ii)])
            neural_state_activated = tf.nn.relu(neural_state)
            curr = neural_state_activated
            
            where_bool_curr = tf.greater(neural_state_activated, 0)
            where_sum_curr = tf.reduce_sum(tf.cast(where_bool_curr, tf.float32), axis=[0])  
            all_where_bools.append(where_bool_curr)
            
        return all_where_bools

    def get_features(self, data):
        """
        Feed a batch of training data to the modeland record the resulting 
        activation patterns. Assumes that the session has a trained model 
        available
        
        Arguments
        ---------
        data : input data of dimension (batch_size, dim1, dim2)
        
        Returns
        -------
        all_out : list of PxMxN arrays, where the list
            indexes the layer/depth, P indexes the batch
            M indexes the (flattened) dimensionality of the input data, 
            and N indexes individual neurons in the layer

        """
            
        X_train = np.copy(data)
        
        wspan, hspan = X_train.shape[-2:]
        
        all_out = list()
        for ind in range(self.num_layers):
            all_out.append(list())
            
        X = tf.placeholder(tf.float32, shape=(self.wspan, self.hspan)) 
        ca_map_inner = self.ca_map(X)

        for ind in range(self.num_layers):

            all_outs = list()  ## can probably delete this line
            for X_train_item in X_train:
                out = self.sess.run(ca_map_inner, feed_dict={X: X_train_item})
                out = [np.squeeze(item) for item in out]

                all_out[ind].append(out[ind])

        all_out = [np.array(item).astype(np.float) for item in all_out]
        
        return all_out

    def load_ca(self, model_path):
        """
        Given a path to a checkpoint file, load the trained model and fill out the appropriate parameters
        Must be done within an open Tensorflow session
        """
        X = tf.placeholder(tf.float32, shape=(self.wspan, self.hspan))
        init = tf.global_variables_initializer()

        # sess = currentSession

        self.sess.run(init)
        saver = tf.train.Saver(max_to_keep=None)

        # initialize a blank model
        ca_model = self.ca_cnn()

        saver.restore(self.sess, model_path)

        return ca_model



