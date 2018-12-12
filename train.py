import tensorflow as tf
import os
from datetime import datetime

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_integer(name='log_frequency',
                    default=10,
                    help='log frequency')

from model import lenet

DATA_BASE = 'data'
TRAIN_DATA = 'train.tfrecord'
NUM_EPOCHS = 5
TRAIN_SIZE = 6000
TEST_SIZE = 1000
BATCH_SIZE = 100
MAX_STEPS = NUM_EPOCHS * (TRAIN_SIZE / BATCH_SIZE)
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01      # Initial learning rate.


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
    global_step = tf.train.get_or_create_global_step()
    
    # dataset input
    images, labels = input_pipe(is_training=True, batch_size=BATCH_SIZE)
    
    # neural network model
    logits, end_points = lenet.lenet(images, is_training=True)
    # print name and shape of each tensor
    print("layers:")
    for k_, v_ in end_points.items():
      print('name =', v_.name, ', shape =', v_.get_shape())
    
    # loss function
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
        name = 'cross_entropy')
    total_loss = loss    
    tf.summary.scalar('total_loss', total_loss)

    # specify learning rate
    num_batches_per_epoch = TRAIN_SIZE / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    # add histograms for trainable variables
    for var_ in tf.trainable_variables():
      tf.summary.histogram(var_.op.name, var_)

    # specify optimizer
    opt = tf.train.GradientDescentOptimizer(lr)

    # just to visualize the gradients
    grads = opt.compute_gradients(total_loss)
    # add histograms for gradients
    for grad_, var_ in grads:
      if grad_ is not None:
        tf.summary.histogram(var_.op.name + '/gradients', grad_)
    
    # train op
    train_op = opt.minimize(total_loss, global_step=global_step)
    
    # run
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_log_dir,
        config=tf.ConfigProto(log_device_placement=False),
        save_summaries_steps=10
    ) as mon_sess:
      while not mon_sess.should_stop():
        _, loss_, step_ = mon_sess.run([train_op, total_loss, global_step])
        if step_ % flags.FLAGS.log_frequency == 0:
          print(datetime.now(), 'step=', step_, '/', MAX_STEPS, 'loss=', loss_)

  print('training done')  

if __name__ == '__main__':
  tf.app.run()

