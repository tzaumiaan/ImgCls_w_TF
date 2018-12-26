import tensorflow as tf
import os
import gzip
import numpy as np

DATA_BASE = 'data'

MNIST_IMAGE_SIZE = 28
MNIST_NUM_CHANNELS = 1
MNIST_PIXEL_DEPTH = 255

dataset_size = {
    'mnist': {
        'train': 16000,
        'valid':  4000,
        'test' :  2000}
}

def generate_tfrecord_for_mnist(file_dict, mode='train'):
  assert mode in ['train', 'test'], "Invalid data set type"   
  
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
    images = (images - (MNIST_PIXEL_DEPTH / 2.0)) / MNIST_PIXEL_DEPTH
    images = images.reshape(dnum, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS)
  #print(images.shape)
  
  label_source = os.path.join(DATA_BASE, file_dict[mode+'_label'])
  print("extracting", label_source, "...")
  with gzip.open(label_source) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * dnum)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  #print(labels.shape)
  
  assert labels.shape[0]==images.shape[0], "Sizes not the same between images and labels"
   
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

def parse_tfrecord_for_mnist(serialized_example):
  feat = tf.parse_single_example(
      serialized_example,
      features={'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(feat['image'], tf.float32)
  image = tf.reshape(image, [MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
  label = tf.cast(feat['label'], tf.int64)
  return image, label

def get_dataset(dset='mnist', mode='train', batch_size=10, fold_index=None):
  if mode is 'valid':
    tfrec_in = 'train.tfrecord'
  else:
    tfrec_in = mode + '.tfrecord'
  filename = os.path.join(DATA_BASE, tfrec_in)
  assert os.path.exists(filename), 'dataset not found'
  dataset = tf.data.TFRecordDataset(filename)
  
  if dset is 'mnist':
    dataset = dataset.map(parse_tfrecord_for_mnist)
    
    # this part serves the purpose of 5-fold cross validation
    # we roll the dataset based on fold index by 
    # repeating twice then performing skip-and-take
    if mode is not 'test' and fold_index is not None:
      dataset = dataset.repeat(2)
      dataset = dataset.skip(fold_index*dataset_size[dset]['valid'])
      dataset = dataset.take(dataset_size[dset]['train'] + dataset_size[dset]['valid'])
    
    # this part generates the desired dataset with desired size
    if mode is 'valid':
      dataset = dataset.skip(dataset_size[dset]['train'])
      dataset = dataset.take(dataset_size[dset]['valid'])
    else:
      dataset = dataset.take(dataset_size[dset][mode])

    dataset = dataset.batch(batch_size)
    return dataset

def create_placeholder_for_input(dset='mnist', batch_size=None):
  # note: None for batch size makes it dynamically adjustable
  #       the size will be decided at the moment data are fed in
  if dset is 'mnist':
    labels = tf.placeholder(tf.int64, [batch_size])
    images = tf.placeholder(
        tf.float32,
        [batch_size, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS])
    tf.add_to_collection('labels', labels)
    tf.add_to_collection('images', images)
  return images, labels
