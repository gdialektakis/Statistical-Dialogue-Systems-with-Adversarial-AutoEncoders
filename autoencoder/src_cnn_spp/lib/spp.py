from __future__ import division

import numpy as np
import tensorflow as tf


def pool_n_bins_segments(inputs, n_bins, pooling_type):
    """
    Performs a pooling operation that results in a fixed size: output_size.
    
    Used by spatial_pyramid_pool. Refer to appendix A in [1].
    
    Args:
        inputs: A 4D Tensor (B, H=1, W, C=1)
        output_size: The output size of the pooling operation.
        mode: The pooling mode {max, avg}
        
    Returns:
        A list of tensors, for each output bin.
        The list contains output_size elements, where
        each elment is a Tensor (N, C).
        
    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.
            
    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    inputs_shape = inputs.get_shape().as_list()
    b = inputs_shape[0]
    w = inputs_shape[2]
    ch = inputs_shape[3]
    
    pooling_type = pooling_type.upper()

    if pooling_type == 'MAX':
        pooling_op = tf.nn.max_pool
    elif pooling_type == 'AVG':
        pooling_op = tf.nn.avg_pool
    else:
        msg = "pooling_type must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode)) 

    k = int(np.ceil(w / n_bins))
    w1 = k*n_bins
    dw = w1 - w
    if dw > 0:
        inputs = tf.concat([inputs, tf.zeros((tf.shape(inputs)[0], 1, dw, ch), dtype=inputs.dtype)], axis=2)

    result = pooling_op(inputs, ksize=[1,1,k,1], strides=[1,1,k,1], padding='VALID')

    result = tf.layers.flatten(result)

    return result


def spatial_pyramid_pool(inputs, level_n_bins_list, pooling_type='max'):
    """
    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 1D input of arbitrary size into an output of fixed
    dimenson.
    Hence, the convolutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimension of the input
    vector is unknown.
    
    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i`, with :math:`n_i` as the number
    of pooling operations per dimension level :math:`i`.
    
    The length of the parameter dimensions is the level of the spatial pyramid.
    
    Args:
        inputs: A 4D Tensor (B, H=1, W, C=1).
        dimensions: The list of :math:`n_i`'s that define the output dimension
        of each pooling level :math:`i`. The length of dimensions is the level of
        the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
        implementation: The implementation to use, either 'kaiming' or 'fast'.
        kamming is the original implementation from the paper, and supports variable
        sizes of input vectors, which fast does not support.
    
    Returns:
        A fixed length vector representing the inputs.
    
    Notes:
        SPP should be inserted between the convolutional part of a DNN and it's
        dense part. Convolutions can be used for arbitrary input dimensions, but
        the size of their output will depend on their input dimensions.
        Connecting the output of the convolutional to the dense part then
        usually demands us to fix the dimensons of the network's input.
        The spatial pyramid pooling layer, however, allows us to leave 
        the network input dimensions arbitrary. 
        The advantage over a global pooling layer is the added robustness 
        against object deformations due to the pooling on different scales.
        
    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.
            
    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    pool_list = []
    for n_bins in level_n_bins_list:
        pool_list.append( pool_n_bins_segments(inputs, n_bins, pooling_type) )
    
    return tf.concat(pool_list, 1)


