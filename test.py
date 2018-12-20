import tensorflow as tf

from model.lenet import lenet

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_integer(name='batch_size',
                    default=100,
                    help='batch size')

MNIST_IMAGE_SIZE = 28
MNIST_NUM_CHANNELS = 1

model_path = 'train/mnist_bs_100_lr_0.1/model_epoch6.ckpt'

def create_placeholder_for_input():
  if flags.FLAGS.dataset is 'mnist':
    labels = tf.placeholder(tf.int64, [flags.FLAGS.batch_size])
    images = tf.placeholder(
        tf.float32,
        [flags.FLAGS.batch_size, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  return images, labels

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  
  with tf.Graph().as_default():
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph(model_path + '.meta')
      saver.restore(sess, model_path)
      print("Model restored")
     
      ops = sess.graph.get_operations()
      for op_ in ops:
        print(op_.values())
    
      sess.run(init_op)
      print("Model initialized")


if __name__ == '__main__':
  tf.app.run()

