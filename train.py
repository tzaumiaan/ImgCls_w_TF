import tensorflow as tf
import os
from datetime import datetime

from model.lenet import lenet

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_integer(name='log_freq',
                    default=10,
                    help='log frequency')
flags.DEFINE_integer(name='num_epochs',
                    default=6,
                    help='number of epochs')
flags.DEFINE_integer(name='batch_size',
                    default=100,
                    help='batch size')
flags.DEFINE_float(name='init_lr',
                    default=0.1,
                    help='initial learning rate')

DATA_BASE = 'data'
TRAIN_DATA = 'train.tfrecord'

LOG_BASE = 'train'

TRAIN_SIZE = 16000
VALID_SIZE = 4000
TRAIN_STEPS_PER_EPOCH = int(TRAIN_SIZE // flags.FLAGS.batch_size)
VALID_STEPS_PER_EPOCH = int(VALID_SIZE // flags.FLAGS.batch_size)
# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 2.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

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

def get_dataset(mode='train', batch_size=10):
  if mode is 'train' or 'valid':
    filename = os.path.join(DATA_BASE,TRAIN_DATA)
  
  assert os.path.exists(filename), 'dataset not found'
  dataset = tf.data.TFRecordDataset(filename)
  
  if flags.FLAGS.dataset is 'mnist':
    dataset = dataset.map(parse_tfrecord_for_mnist)
    if mode is 'train':
      dataset = dataset.take(TRAIN_SIZE)
    elif mode is 'valid':
      dataset = dataset.skip(TRAIN_SIZE)
      dataset = dataset.take(VALID_SIZE)

    dataset = dataset.batch(batch_size)
    return dataset

def create_placeholder_for_input():
  if flags.FLAGS.dataset is 'mnist':
    labels = tf.placeholder(tf.int64, [flags.FLAGS.batch_size])
    images = tf.placeholder(
        tf.float32,
        [flags.FLAGS.batch_size, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  return images, labels

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  # enable printing training log
  tf.logging.set_verbosity(tf.logging.INFO)
  
  train_log_base = LOG_BASE
  train_case = flags.FLAGS.dataset 
  train_case += '_bs_' + str(flags.FLAGS.batch_size)
  train_case += '_lr_' + str(flags.FLAGS.init_lr)
  train_log_dir = os.path.join(LOG_BASE,train_case) 
  if not tf.gfile.Exists(train_log_base):
    tf.gfile.MakeDirs(train_log_base)
  if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)
  
  with tf.Graph().as_default(): 
    global_step = tf.train.get_or_create_global_step()
    
    # dataset input, always using CPU for this section
    with tf.device('/cpu:0'):
      # dataset source
      trn_dataset = get_dataset(mode='train', batch_size=flags.FLAGS.batch_size)
      vld_dataset = get_dataset(mode='valid', batch_size=flags.FLAGS.batch_size)
      # iterator 
      iterator = tf.data.Iterator.from_structure(
          trn_dataset.output_types,
          trn_dataset.output_shapes)
      # get a new batch from iterator
      get_batch = iterator.get_next()
      # ops for initializing the iterators
      # for choosing dataset for one epoch
      trn_init_op = iterator.make_initializer(trn_dataset)
      vld_init_op = iterator.make_initializer(vld_dataset)
    
    # placeholder for images and labels
    images, labels = create_placeholder_for_input()
    is_training = tf.placeholder(tf.bool)
    tf.summary.image('images', images)

    # neural network model
    logits, end_points = lenet(images, is_training=is_training)
    
    # print name and shape of each tensor
    print("layers:")
    for k_, v_ in end_points.items():
      print('name =', v_.name, ', shape =', v_.get_shape())
    
    # prediction of this batch
    pred = tf.argmax(tf.nn.softmax(logits), axis=1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(pred,labels), tf.float32)) / flags.FLAGS.batch_size
    tf.summary.scalar('accuracy', accuracy)
     
    # loss function
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
        name = 'cross_entropy')
    total_loss = loss    
    tf.summary.scalar('total_loss', total_loss)

    # specify learning rate
    decay_steps = int(TRAIN_STEPS_PER_EPOCH * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        flags.FLAGS.init_lr,
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
    grads = opt.compute_gradients(total_loss)
    
    # add histograms for gradients
    for grad_, var_ in grads:
      if grad_ is not None:
        tf.summary.histogram(var_.op.name + '/gradients', grad_)
    
    # train op
    train_op = opt.apply_gradients(grads, global_step=global_step)
    
    # summerize all
    summary = tf.summary.merge_all()
    # summary writer
    summary_writer = tf.summary.FileWriter(train_log_dir)
    # checkpoint saver
    saver = tf.train.Saver()
    
    # session part
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      # initialization
      sess.run(init_op)
      summary_writer.add_graph(sess.graph)
      
      # epoch loop
      for epoch in range(flags.FLAGS.num_epochs):
        print(datetime.now(), 'epoch:', epoch+1, '/', flags.FLAGS.num_epochs)
        
        # training phase
        print('==== training phase ====')
        # specify dataset for training
        sess.run(trn_init_op)
        # training loop
        for step in range(TRAIN_STEPS_PER_EPOCH):
          # get batch for training
          trn_images, trn_labels = sess.run(get_batch)
          # run taining op
          _, l_, acc_, sum_ = sess.run(
              [train_op, total_loss, accuracy, summary],
              feed_dict={
                  images: trn_images,
                  labels: trn_labels,
                  is_training: True})
          if (step+1) % flags.FLAGS.log_freq == 0:
            print(
                datetime.now(),
                'training step:', step+1, '/', TRAIN_STEPS_PER_EPOCH,
                'loss={:.5f}'.format(l_),
                'acc={:.4f}'.format(acc_))
            summary_writer.add_summary(sum_, epoch*TRAIN_STEPS_PER_EPOCH + step)
        
        # validation phase
        print('==== validation phase ====')
        # specify dataset for validation
        sess.run(vld_init_op)
        # going through validation batches
        vld_acc = 0.0
        vld_batch_count = 0
        for b_ in range(VALID_STEPS_PER_EPOCH):
          # get batch for training
          vld_images, vld_labels = sess.run(get_batch)
          # run taining op
          acc_ = sess.run(accuracy, feed_dict={
              images: vld_images,
              labels: vld_labels,
              is_training: False})
          vld_acc += acc_
          vld_batch_count += 1
        vld_acc /= vld_batch_count
        print(datetime.now(), 'validation result: acc={:.4f}'.format(vld_acc))
        
        # checkpoint saving
        print(datetime.now(), 'saving checkpoint of model ...')
        ckpt_name = os.path.join(train_log_dir,'model_epoch'+str(epoch+1)+'.ckpt')
        saver.save(sess, ckpt_name)
        print(datetime.now(), ckpt_name, 'saved')
        
        # epoch end
        print(datetime.now(), 'epoch:', epoch+1, 'done')
  
  print('training done')
      

if __name__ == '__main__':
  tf.app.run()

