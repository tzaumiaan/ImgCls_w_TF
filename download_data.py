import tensorflow as tf
import os
from six.moves import urllib

from data_utils import DATA_BASE
from data_utils import generate_tfrecord_for_mnist

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')

MNIST_URL_BASE = "http://yann.lecun.com/exdb/mnist"
MNIST_FILES = { 'train_image': 'train-images-idx3-ubyte.gz',
                'train_label': 'train-labels-idx1-ubyte.gz',
                'test_image': 't10k-images-idx3-ubyte.gz',
                'test_label': 't10k-labels-idx1-ubyte.gz' }

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

def main(args):
  print('dataset = ', flags.FLAGS.dataset)
  if(flags.FLAGS.dataset is 'mnist'):
    for f_ in MNIST_FILES:
      url_ = os.path.join(MNIST_URL_BASE, MNIST_FILES[f_])
      maybe_download(url_, DATA_BASE)
    generate_tfrecord_for_mnist(MNIST_FILES, 'train')
    generate_tfrecord_for_mnist(MNIST_FILES, 'test')
  print('dataset ready')

if __name__ == '__main__':
  tf.app.run()


