import tensorflow as tf
from absl import app, flags, logging
import os
from datetime import datetime

from data_utils import get_dataset, create_placeholder_for_input
from data_utils import dataset_size

from model.lenet import lenet
from model.cifarnet import cifarnet

flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')
flags.DEFINE_integer(name='log_freq',
                    default=10,
                    help='log frequency')
flags.DEFINE_string(name='log_directory',
                    default='train',
                    help='log directory')
flags.DEFINE_integer(name='num_epochs',
                    default=20,
                    help='number of epochs')
flags.DEFINE_integer(name='batch_size',
                    default=50,
                    help='batch size')
flags.DEFINE_float(name='init_lr',
                    default=0.1,
                    help='initial learning rate')
flags.DEFINE_float(name='l2_scale',
                    default=1e-8,
                    help='l2 regularizer scale')

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  TRAIN_SIZE = dataset_size[flags.FLAGS.dataset]['train']
  TRAIN_STEPS_PER_EPOCH = int(TRAIN_SIZE // flags.FLAGS.batch_size)
  # Constants describing the training process.
  NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
  LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
  
  train_log_base = flags.FLAGS.log_directory
  train_case = flags.FLAGS.dataset 
  train_case += '_bs_' + str(flags.FLAGS.batch_size)
  train_case += '_lr_' + str(flags.FLAGS.init_lr)
  train_case += '_l2s_' + str(flags.FLAGS.l2_scale)
  train_log_dir = os.path.join(train_log_base, train_case) 
  if not tf.gfile.Exists(train_log_base):
    tf.gfile.MakeDirs(train_log_base)
  if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)
  
  with tf.Graph().as_default(): 
    # create global step
    global_step = tf.train.get_or_create_global_step()
    
    with tf.name_scope('input_pipe'):
      # use epoch count to pick fold index for cross validation
      epoch_count = tf.floordiv(global_step, TRAIN_STEPS_PER_EPOCH)
      fold_index = tf.floormod(epoch_count, 10) # 10-fold dataset
      
      # dataset input, always using CPU for this section
      with tf.device('/cpu:0'):
        # dataset source
        trn_dataset = get_dataset(
            dset=flags.FLAGS.dataset, mode='train',
            batch_size=flags.FLAGS.batch_size,
            fold_index=fold_index)
        vld_dataset = get_dataset(
            dset=flags.FLAGS.dataset, mode='valid',
            fold_index=fold_index)
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
    images, labels = create_placeholder_for_input(
        dset=flags.FLAGS.dataset)
    is_training = tf.placeholder(tf.bool, name='is_training')
    tf.add_to_collection('is_training', is_training)
    tf.summary.image('images', images)

    # neural network model
    if flags.FLAGS.dataset == 'mnist':
      model_network = lenet
    elif flags.FLAGS.dataset == 'cifar10':
      model_network = cifarnet
    else:
      raise(ValueError, 'Invalid dataset') 
    logits, end_points = model_network(images, is_training=is_training,
          l2_scale=flags.FLAGS.l2_scale)
    
    # print name and shape of each tensor
    print("layers:")
    for k_, v_ in end_points.items():
      print('name =', v_.name, ', shape =', v_.get_shape())
    # print the total size of trainable variables
    n_params = 0
    for var_ in tf.trainable_variables():
      var_shape = var_.get_shape()
      n_params_var = 1
      for dim_ in var_shape:
        n_params_var *= dim_.value
      n_params += n_params_var
    print("model parameter size:", n_params)
    
    # prediction of this batch
    with tf.name_scope('prediction'):
      pred = tf.argmax(tf.nn.softmax(logits), axis=1)
      match_count =  tf.reduce_sum(tf.cast(tf.equal(pred,labels), tf.float32))
      # note: here the running batch size can be changed in testing mode,
      #       so we cannot reuse the batch size from flags
      running_batch_size = tf.cast(tf.size(pred),tf.float32)
      accuracy = match_count / running_batch_size
      tf.add_to_collection('accuracy', accuracy)
      tf.summary.scalar('accuracy', accuracy)
 
    # loss function
    with tf.name_scope('losses'):
      raw_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
          name = 'cross_entropy')
      tf.summary.scalar('raw_loss', raw_loss)
      regu_loss = tf.add_n(tf.losses.get_regularization_losses())
      tf.summary.scalar('regu_loss', regu_loss)
      total_loss = raw_loss + regu_loss    
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
    
    # compute gradients and apply
    # note: with batch norm layers we have to use update_ops
    #       to get hidden variables into the list needed
    #       to be trained
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): 
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
        # get batch for validation
        vld_images, vld_labels = sess.run(get_batch)
        # run taining op
        vld_acc, vld_loss = sess.run([accuracy, total_loss],feed_dict={
            images: vld_images,
            labels: vld_labels,
            is_training: False})
        print(
            datetime.now(),
            'validation result: loss={:.5f}'.format(vld_loss),
            'acc={:.4f}'.format(vld_acc))
        
        # checkpoint saving
        print(datetime.now(), 'saving checkpoint of model ...')
        ckpt_name = os.path.join(train_log_dir,'model_epoch'+str(epoch+1)+'.ckpt')
        saver.save(sess, ckpt_name)
        print(datetime.now(), ckpt_name, 'saved')
        
        # epoch end
        print(datetime.now(), 'epoch:', epoch+1, 'done')
  
  print('training done')
      

if __name__ == '__main__':
  app.run(main)

