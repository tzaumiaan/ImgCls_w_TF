import tensorflow as tf
from tensorflow.contrib import slim
import os

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')

from model import lenet

DATA_BASE = 'data'
TRAIN_DATA = 'train.tfrecord'
NUM_EPOCHS = 5

MNIST_IMAGE_SIZE = 28
MNIST_NUM_CHANNELS = 1

def parse_tfrecord_for_mnist(serialized_example):
  feat = tf.parse_single_example(
      serialized_example,
      features={'train/label': tf.FixedLenFeature([], tf.int64),
                'train/image': tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(feat['train/image'], tf.float32)
  image = tf.reshape(image, [MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  label = tf.cast(feat['train/label'], tf.int64)
  return image, label

def input_pipe(is_training=True, batch_size=10):
  filename = os.path.join(DATA_BASE,TRAIN_DATA)
  assert os.path.exists(filename), 'dataset not found'
  
  dataset = tf.data.TFRecordDataset(filename)
  
  if(flags.FLAGS.dataset is 'mnist'):
    dataset = dataset.map(parse_tfrecord_for_mnist)
    dataset = dataset.repeat(NUM_EPOCHS) 
    dataset = dataset.batch(batch_size)

  return dataset.make_one_shot_iterator().get_next()


def main(args):
  print('dataset = ', flags.FLAGS.dataset)
  # enable printing training log
  tf.logging.set_verbosity(tf.logging.INFO)
  
  train_log_dir = 'train'
  if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)
  
  
  with tf.Graph().as_default(): 
    # dataset input
    images, labels = input_pipe(is_training=True, batch_size=100)
    
    # neural network model
    logits, end_points = lenet.lenet(images, is_training=True)
    # print name and shape of each tensor
    print("layers:")
    for k_, v_ in end_points.items():
      print('name =', v_.name, ', shape =', v_.get_shape())
    
    # loss function
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)
    
    # specify the optimizer and create the train op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    final_loss = slim.learning.train(
        train_op,
        logdir=train_log_dir,
        #number_of_steps=10,
        #save_summaries_secs=5,
        log_every_n_steps=10)
    print("Last loss", final_loss)

# # test session
# import cv2
# image_orig = tf.cast(image * 255 + 128, tf.uint8)
# image_for_view = tf.image.convert_image_dtype(image_orig, dtype=tf.uint8)
# with tf.Session() as sess:
#   while True:
#     xxx = sess.run({'image':image_for_view, 'label':label})
#     print(xxx['label'])
#     cv2.imshow("", xxx['image'])
#     cv2.waitKey(0)
   
  print('training done')  

if __name__ == '__main__':
  tf.app.run()

