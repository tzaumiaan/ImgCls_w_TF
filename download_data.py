import tensorflow as tf
import numpy as np
import os
from six.moves import urllib
import gzip

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')

DATA_BASE = 'data'

MNIST_URL_BASE = "http://yann.lecun.com/exdb/mnist"
MNIST_FILES = { 'train_image': 'train-images-idx3-ubyte.gz',
                'train_label': 'train-labels-idx1-ubyte.gz',
                'test_image': 't10k-images-idx3-ubyte.gz',
                'test_label': 't10k-labels-idx1-ubyte.gz' }
MNIST_IMAGE_SIZE = 28
MNIST_NUM_CHANNELS = 1
MNIST_PIXEL_DEPTH = 255

def maybe_download(url, download_dir):
  """
  download a file with link if it doesn't exist yet
  
  input:
    url: URL link of tarball file, including the 'http://' 
         as beginning 
    download_dir: the folder for the downloaded file  
  
  output: none
  """
  # filename is the last part from url
  filename = url.split('/')[-1]
  filepath = os.path.join(download_dir, filename)

  # check if the file already exists
  # only download and extract when file does not exist
  if not os.path.exists(filepath):
    # if download_dir not yet exists, make a new one
    if not os.path.exists(download_dir):
      os.makedirs(download_dir)
    #download the file
    filepath, _ = urllib.request.urlretrieve(
        url = url,
        filename = filepath )
    print(filepath, ": download finished. ")
  else:
    print(filepath, ": already exists. nothing done.")

def _tffeat_int64(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _tffeat_bytes(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecord_for_mnist(dset='train', dnum=100):
  assert dset in ['train', 'test'], "Invalid data set type" 
  
  image_source = os.path.join(DATA_BASE, MNIST_FILES[dset+'_image'])
  print("extracting", image_source, "...")
  with gzip.open(image_source) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE * dnum * MNIST_NUM_CHANNELS)
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = (images - (MNIST_PIXEL_DEPTH / 2.0)) / MNIST_PIXEL_DEPTH
    images = images.reshape(dnum, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_NUM_CHANNELS)
  #print(images.shape)
  
  label_source = os.path.join(DATA_BASE, MNIST_FILES[dset+'_label'])
  print("extracting", label_source, "...")
  with gzip.open(label_source) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * dnum)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  #print(labels.shape)
  
  assert labels.shape[0]==images.shape[0], "Sizes not the same between images and labels"
   
  tfrec_out = os.path.join(DATA_BASE, dset+'.tfrecord') 
  tfrec_writer = tf.python_io.TFRecordWriter(tfrec_out)
  print("writing to", tfrec_out, "...")
  for i in range(images.shape[0]):
    # convert data to features
    list_label = tf.train.Int64List(value=[labels[i]])
    list_image = tf.train.BytesList(value=[tf.compat.as_bytes(images[i].tostring())])
    feat_label = tf.train.Feature(int64_list=list_label)
    feat_image = tf.train.Feature(bytes_list=list_image)
    feat = {dset+'/label': feat_label,
            dset+'/image': feat_image}
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    tfrec_writer.write(example.SerializeToString())
  tfrec_writer.close()

def main(args):
  print('dataset = ', flags.FLAGS.dataset)
  if(flags.FLAGS.dataset is 'mnist'):
    for f_ in MNIST_FILES:
      url_ = os.path.join(MNIST_URL_BASE, MNIST_FILES[f_])
      maybe_download(url_, DATA_BASE)
    generate_tfrecord_for_mnist('train', dnum=20000)
    generate_tfrecord_for_mnist('test', dnum=2000)
  print('dataset ready')

if __name__ == '__main__':
  tf.app.run()


