import tensorflow as tf

def get_objective(name):
    if (name == 'mean_squared_error'):
        return mean_squared_error
    elif (name == 'mean_cross_entropy'):
        return mean_cross_entropy
    elif (name == 'mean_softmax_cross_entropy_with_logits'):
        return mean_softmax_cross_entropy_with_logits
    else:
        raise Exception('The objective function ' + name + ' has not been implemented.')

# Y_net = Y_predicted, Y_data = Y_true
def mean_squared_error(Y_net, Y_data):
    return tf.reduce_mean(tf.square(Y_net - Y_data)) 

def mean_cross_entropy(Y_net, Y_data):
    return -tf.reduce_mean(Y_data * tf.log(Y_net)) 

def mean_softmax_cross_entropy_with_logits(Y_net, Y_data):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_data, logits=Y_net))




