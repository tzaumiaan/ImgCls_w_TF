import tensorflow as tf
from datetime import datetime
from data_utils import get_dataset
from data_utils import dataset_size

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_integer(name='batch_size',
                    default=100,
                    help='batch size')

model_path = 'train/cifar10_bs_100_lr_0.1/model_epoch6.ckpt'

TEST_SIZE = dataset_size[flags.FLAGS.dataset]['test']
TEST_STEPS_PER_EPOCH = int(TEST_SIZE // flags.FLAGS.batch_size)

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  
  with tf.Graph().as_default():
    # dataset input, always using CPU for this section
    with tf.device('/cpu:0'):
      # dataset source
      test_dataset = get_dataset(
          dset=flags.FLAGS.dataset, mode='test',
          batch_size=flags.FLAGS.batch_size)
      # iterator 
      iterator = tf.data.Iterator.from_structure(
          test_dataset.output_types,
          test_dataset.output_shapes)
      # get a new batch from iterator
      get_batch = iterator.get_next()
      # ops for initializing the iterators
      # for choosing dataset for one epoch
    test_init_op = iterator.make_initializer(test_dataset)
    

    # restore saved model and run testing
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      model_meta = model_path + '.meta'
      saver = tf.train.import_meta_graph(model_meta)
      print(datetime.now(), 'meta graph imported from', model_meta)
      saver.restore(sess, model_path)
      print(datetime.now(), 'model restored')
     
      # import operators for reference
      accuracy = tf.get_collection('accuracy')[0]
      images = tf.get_collection('images')[0]
      labels = tf.get_collection('labels')[0]
      is_training = tf.get_collection('is_training')[0]
       
      sess.run(init_op)
      print(datetime.now(), 'model initialized')
      
      # testing phase
      print('==== testing phase ====')
      # specify dataset for validation
      sess.run(test_init_op)
      # going through validation batches
      test_acc = 0.0
      test_batch_count = 0
      for b_ in range(TEST_STEPS_PER_EPOCH):
        # get batch for testing
        test_images, test_labels = sess.run(get_batch)
        # run taining op
        acc_ = sess.run(accuracy, feed_dict={
            images: test_images,
            labels: test_labels,
            is_training: False})
        print(datetime.now(), 'batch {}/{} acc={:.4f}'.format(b_+1, TEST_STEPS_PER_EPOCH, acc_))
        test_acc += acc_
        test_batch_count += 1
      test_acc /= test_batch_count
      print(datetime.now(), 'testing result: acc={:.4f}'.format(test_acc))


if __name__ == '__main__':
  tf.app.run()

