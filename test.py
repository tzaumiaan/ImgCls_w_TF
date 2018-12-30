import tensorflow as tf
from datetime import datetime
from data_utils import get_dataset

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_string(name='ckpt_path',
                    default='train/mnist_bs_200_lr_0.1_l2s_1e-06/model_epoch20.ckpt',
                    help='checkpoint path')

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  
  with tf.Graph().as_default():
    # dataset input, always using CPU for this section
    with tf.device('/cpu:0'):
      # dataset source
      test_dataset = get_dataset(
          dset=flags.FLAGS.dataset, mode='test')
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
      model_path = flags.FLAGS.ckpt_path
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
      # specify dataset for test
      sess.run(test_init_op)
      # get batch for testing
      test_images, test_labels = sess.run(get_batch)
      # run taining op
      test_acc = sess.run(accuracy, feed_dict={
          images: test_images,
          labels: test_labels,
          is_training: False})
      print(datetime.now(), 'testing result: acc={:.4f}'.format(test_acc))

if __name__ == '__main__':
  tf.app.run()

