import tensorflow as tf
from tensorflow.contrib import slim

def cifarnet(
    images,
    num_classes=10,
    is_training=False,
    dropout_keep_prob=0.5,
    l2_scale=0.0,
    bn_decay=0.90,
    scope='CIFARNet'):
  # dict for components to be monitored
  end_points = {}
  
  # model body
  with tf.variable_scope(scope, 'CIFARNet', [images]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(scale=l2_scale),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        activation_fn=tf.nn.relu):
      with slim.arg_scope([slim.batch_norm], decay=bn_decay):
        net = end_points['conv1'] = slim.conv2d(images, 32, [5,5], scope='conv1')
        net = end_points['conv2'] = slim.conv2d(net, 32, [5,5], scope='conv2')
        net = end_points['pool2'] = slim.max_pool2d(net, [2,2], 2, scope='pool2')
        net = end_points['bn2'] = slim.batch_norm(net, is_training=is_training, scope='bn2')
        net = end_points['conv3'] = slim.conv2d(net, 64, [3,3], scope='conv3')
        net = end_points['conv4'] = slim.conv2d(net, 64, [3,3], scope='conv4')
        net = end_points['pool4'] = slim.max_pool2d(net, [2,2], 2, scope='pool4')
        net = end_points['bn4'] = slim.batch_norm(net, is_training=is_training, scope='bn4')
        net = end_points['flatten'] = slim.flatten(net, scope='flatten')
        net = end_points['fc5'] = slim.fully_connected(net, 100, scope='fc5')
        
        if not num_classes:
          # without final layer, can be used for transfer learning
          return net, end_points
        
        net = end_points['dropout5'] = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout5')
        logits = end_points['logits'] = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='fc6')
  return logits, end_points

