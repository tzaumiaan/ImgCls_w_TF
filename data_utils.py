import tensorflow as tf
import os
import gzip
import numpy as np

DATA_BASE = 'data'

MNIST_IMAGE_SIZE = 28
MNIST_NUM_CHANNELS = 1
MNIST_PIXEL_DEPTH = 255

CIFAR_IMAGE_SIZE = 32
CIFAR_NUM_CHANNELS = 3
CIFAR_PIXEL_DEPTH = 255

CIFAR10_TRAIN_FILES = ['cifar-10-batches-py/data_batch_1',
                       'cifar-10-batches-py/data_batch_2',
                       'cifar-10-batches-py/data_batch_3',
                       'cifar-10-batches-py/data_batch_4',
                       'cifar-10-batches-py/data_batch_5']
CIFAR10_TEST_FILES = ['cifar-10-batches-py/test_batch']

dataset_size = {
    'mnist': {
        'train': 16000,
        'valid':  4000,
        'test' :  2000},
    'cifar10': {
        'train': 40000,
        'valid': 10000,
        'test' : 10000}
}

def per_image_uniform_scaling(images, max_range, min_range):
  """
  perform the uniform normalization with given range
  input:
    images: numpy array with dimension (dnum, height, width, channel)
  """
  images = images.astype(np.float32)
  mean = 0.5*(max_range - min_range)
  scale = np.float32(max_range - min_range)
  return (images - mean)/scale

def per_image_normalization(images):
  """
  perform the gaussian normalization on each image
  input:
    images: numpy array with dimension (dnum, height, width, channel)
  """
  images = images.astype(np.float32)
  mean = np.mean(images, axis=(1,2)).reshape(images.shape[0],1,1,images.shape[3])
  stddev = np.std(images, axis=(1,2)).reshape(images.shape[0],1,1,images.shape[3])
  scale = np.maximum(stddev, 1/np.sqrt(images.shape[1]*images.shape[2]))
  return (images - mean)/scale

def global_mean_normalization(images):
  """
  substract with the mean image
  input:
    images: numpy array with dimension (dnum, height, width, channel)
  """
  images = images.astype(np.float32)
  mean = np.mean(images, axis=0)
  return (images - mean)

def unpickle(pickle_file):
  import pickle
  with open(pickle_file, 'rb') as f:
    unpickled_dict = pickle.load(f, encoding='bytes')
  return unpickled_dict

def generate_tfrecord(dset, mode='train', file_dict=None):
  assert mode in ['train', 'test'], "Invalid data set type"   
  if dset == 'mnist': 
    assert file_dict is not None, "Invalid file_dict for mnist dataset"   
    
    dnum = dataset_size['mnist'][mode]
    if mode is 'train':
      # in MNIST we further split training set into training and validation
      dnum += dataset_size['mnist']['valid']

    image_source = os.path.join(DATA_BASE, file_dict[mode+'_image'])
    print("extracting", image_source, "...")
    with gzip.open(image_source) as bytestream:
      bytestream.read(16)
      buf = bytestream.read(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE * dnum * MNIST_NUM_CHANNELS)
      images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
      images = images.reshape(dnum, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS)
      images = per_image_uniform_scaling(images, min_range=0, max_range=MNIST_PIXEL_DEPTH)
    #print(images.shape)
    
    label_source = os.path.join(DATA_BASE, file_dict[mode+'_label'])
    print("extracting", label_source, "...")
    with gzip.open(label_source) as bytestream:
      bytestream.read(8)
      buf = bytestream.read(1 * dnum)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    #print(labels.shape)
  
  elif dset=='cifar10':
    filelist = CIFAR10_TRAIN_FILES if mode == 'train' else CIFAR10_TEST_FILES
    labels, images = None, None
    for f_ in filelist:
      dict_ = unpickle(os.path.join(DATA_BASE, f_))
      batch_labels = np.array(dict_[b'labels'])
      batch_images = np.array(dict_[b'data'])
      # reshape to channel, height, width dimemsions
      batch_images = batch_images.reshape(batch_images.shape[0],
          CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)
      # put the channel to last dimension
      batch_images = batch_images.transpose(0,2,3,1)
      # for image visualization
      batch_images = batch_images.astype(np.float32)
      if labels is None and images is None:
        labels, images = batch_labels, batch_images
      else:
        labels = np.concatenate((labels, batch_labels))
        images = np.concatenate((images, batch_images))
      #print(labels.shape, images.shape)
    # for debug: visualize the image and print out the label
    #dict_ = unpickle(os.path.join(DATA_BASE, 'cifar-10-batches-py/batches.meta'))
    #print(dict_[b'label_names'][labels[0]])
    #import matplotlib.pyplot as plt
    #plt.imshow((images[0]).astype(np.uint8))
    #plt.show()
    
    # normalization
    images = per_image_normalization(images)
    #images = per_image_uniform_scaling(images, min_range=0, max_range=CIFAR_PIXEL_DEPTH)
    #images = global_mean_normalization(images)
    
    # for debug: visualize after normalization
    #plt.imshow((images[0]*0.22 + 0.5).astype(np.float32))
    #plt.show()
  
  # numpy array of labels and images are all prepared 
  assert labels.shape[0]==images.shape[0], "Sizes not the same between images and labels"
   
  # start to write out tfrecord
  tfrec_out = os.path.join(DATA_BASE, mode+'.tfrecord') 
  tfrec_writer = tf.python_io.TFRecordWriter(tfrec_out)
  print("writing to", tfrec_out, "...")
  for i in range(images.shape[0]):
    # convert data to features
    list_label = tf.train.Int64List(value=[labels[i]])
    list_image = tf.train.BytesList(value=[tf.compat.as_bytes(images[i].tostring())])
    feat_label = tf.train.Feature(int64_list=list_label)
    feat_image = tf.train.Feature(bytes_list=list_image)
    feat = {'label': feat_label,
            'image': feat_image}
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    tfrec_writer.write(example.SerializeToString())
  tfrec_writer.close()

def parse_tfrecord(serialized_example, dset='mnist'):
  feat = tf.parse_single_example(
      serialized_example,
      features={'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)})
  label = tf.cast(feat['label'], tf.int64)
  image = tf.decode_raw(feat['image'], tf.float32)
  if dset == 'mnist':
    image = tf.reshape(image, [MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  elif dset == 'cifar10':
    image = tf.reshape(image, [CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS])
  else:
    raise ValueError('Invalid dataset: {}'.format(dset))
  
  return image, label


def get_dataset(dset='mnist', mode='train', batch_size=10, fold_index=None):
  if mode is 'valid':
    tfrec_in = 'train.tfrecord'
  else:
    tfrec_in = mode + '.tfrecord'
  filename = os.path.join(DATA_BASE, tfrec_in)
  assert os.path.exists(filename), 'dataset not found'
  dataset = tf.data.TFRecordDataset(filename)
  
  # parse dataset from tfrecord  
  dataset = dataset.map(lambda x: parse_tfrecord(x, dset=dset))
  
  # this part serves the purpose of 5-fold cross validation
  # we roll the dataset based on fold index by 
  # repeating twice then performing skip-and-take
  if mode != 'test' and fold_index is not None:
    dataset = dataset.repeat(2)
    dataset = dataset.skip(fold_index*dataset_size[dset]['valid'])
    dataset = dataset.take(dataset_size[dset]['train'] + dataset_size[dset]['valid'])
  
  # this part generates the desired dataset with desired size
  if mode == 'valid':
    dataset = dataset.skip(dataset_size[dset]['train'])
    dataset = dataset.take(dataset_size[dset]['valid'])
  else:
    dataset = dataset.take(dataset_size[dset][mode])
  
  # shuffle the dataset and split them into batches for training
  if mode == 'train':
    dataset = dataset.shuffle(dataset_size[dset][mode])
    dataset = dataset.batch(batch_size)
  else: # otherwise just make the entire dataset as one batch
    dataset = dataset.batch(dataset_size[dset][mode])
   
  return dataset

def create_placeholder_for_input(dset='mnist', batch_size=None):
  # note: None for batch size makes it dynamically adjustable
  #       the size will be decided at the moment data are fed in
  labels = tf.placeholder(tf.int64, [batch_size])
  tf.add_to_collection('labels', labels)
  
  if dset == 'mnist':
    images = tf.placeholder(
        tf.float32,
        [batch_size, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  elif dset == 'cifar10':
    images = tf.placeholder(
        tf.float32,
        [batch_size, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS])
  else:
    raise ValueError('Invalid dataset: {}'.format(dset))
  tf.add_to_collection('images', images)
  return images, labels
