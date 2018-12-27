import tensorflow as tf
from tensorflow.contrib import slim

def lenet(
    images,
    num_classes=10,
    is_training=False,
    dropout_keep_prob=0.5,
    l2_scale=0.0,
    scope='LeNet'):
  # dict for components to be monitored
  end_points = {}
  
  # model body
  with tf.variable_scope(scope, 'LeNet', [images]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(scale=l2_scale),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        activation_fn=tf.nn.relu):
      net = end_points['conv1'] = slim.conv2d(images, 64, [5,5], scope='conv1')
      net = end_points['pool1'] = slim.max_pool2d(net, [2,2], 2, scope='pool1')
      net = end_points['conv2'] = slim.conv2d(net, 32, [5,5], scope='conv2')
      net = end_points['pool2'] = slim.max_pool2d(net, [2,2], 2, scope='pool2')
      net = end_points['flatten'] = slim.flatten(net, scope='flatten')
      net = end_points['fc3'] = slim.fully_connected(net, 512, scope='fc3')
      net = end_points['fc4'] = slim.fully_connected(net, 128, scope='fc4')
      
      if not num_classes:
        # without final layer, can be used for transfer learning
        return net, end_points
      
      net = end_points['dropout4'] = slim.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout4')
      logits = end_points['logits'] = slim.fully_connected(
          net, num_classes, activation_fn=None, scope='fc5')
  return logits, end_points

