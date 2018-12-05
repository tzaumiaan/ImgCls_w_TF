import tensorflow as tf
from tensorflow.contrib import slim

def lenet(images,
          num_classes=10,
          is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  # dict for components to be monitored
  end_points = {}
  
  # model body
  with tf.variable_scope(scope, 'LeNet', [images]):
    net = end_points['conv1'] = slim.conv2d(images, 32, [5,5], scope='conv1')
    net = end_points['pool1'] = slim.max_pool2d(net, [2,2], 2, scope='pool1')
    net = end_points['conv2'] = slim.conv2d(net, 64, [5,5], scope='conv2')
    net = end_points['pool2'] = slim.max_pool2d(net, [2,2], 2, scope='pool2')
    net = end_points['Flatten'] = slim.flatten(net)
    net = end_points['fc3'] = slim.fully_connected(net, 2014, scope='fc3')
    if not num_classes:
      return net, end_points
    net = end_points['dropout3'] = slim.dropout(net, dropout_keep_prob, 
                                                is_training=is_training,
                                                scope='dropout3')
    logits = end_points['Logits'] = slim.fully_connected(net, num_classes,
                                                         activation_fn=None,
                                                         scope='fc4')
  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

def lenet_arg_scope(weight_decay=0.0):
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularaizer(weight_decay),
      weights_initializaer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc

