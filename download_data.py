import tensorflow as tf
import os
import tarfile
import zipfile
from six.moves import urllib

from data_utils import DATA_BASE
from data_utils import generate_tfrecord_for_mnist
from data_utils import generate_tfrecord_for_cifar10

flags = tf.app.flags
flags.DEFINE_string(name='dataset',
                    default='mnist',
                    help='dataset type')

MNIST_URL_BASE = 'http://yann.lecun.com/exdb/mnist'
MNIST_FILES = { 'train_image': 'train-images-idx3-ubyte.gz',
                'train_label': 'train-labels-idx1-ubyte.gz',
                'test_image': 't10k-images-idx3-ubyte.gz',
                'test_label': 't10k-labels-idx1-ubyte.gz' }

CIFAR_URL_BASE = 'https://www.cs.toronto.edu/~kriz'
CIFAR_FILES = {'cifar10': 'cifar-10-python.tar.gz'}

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
        filename = filepath)
    print(filepath, ": download finished.")
  else:
    print(filepath, ": already exists. nothing done.")

def extract(zfile, extract_path='.'):
  """
  extract a zipped file with supported types
  """
  # specify the opener and mode
  if zfile.endswith('.zip'):
    opener, mode = zipfile.ZipFile, 'r'
  elif zfile.endswith('.tar.gz') or path.endswith('.tgz'):
    opener, mode = tarfile.open, 'r:gz'
  elif zfile.endswith('.tar.bz2') or path.endswith('.tbz'):
    opener, mode = tarfile.open, 'r:bz2'
  else: 
    raise ValueError('Could not extract {} as no appropriate extractor is found'.format(zfile))
  
  if not os.path.exists(extract_path):
    # if not yet exists, make a new one
    os.makedirs(extract_path)
  
  # extract the zfile
  with opener(zfile, mode) as f:
    f.extractall(extract_path)
    f.close()
    print(zfile, 'extracted to', extract_path)

def main(args):
  print('dataset =', flags.FLAGS.dataset)
  if flags.FLAGS.dataset == 'mnist':
    for f_ in MNIST_FILES:
      url_ = os.path.join(MNIST_URL_BASE, MNIST_FILES[f_])
      maybe_download(url_, DATA_BASE)
    generate_tfrecord_for_mnist(MNIST_FILES, 'train')
    generate_tfrecord_for_mnist(MNIST_FILES, 'test')
  elif flags.FLAGS.dataset == 'cifar10':
    url_ = os.path.join(CIFAR_URL_BASE, CIFAR_FILES['cifar10'])
    maybe_download(url_, DATA_BASE)
    zfile_ = os.path.join(DATA_BASE, CIFAR_FILES['cifar10'])
    extract(zfile_, extract_path=DATA_BASE)
    generate_tfrecord_for_cifar10('train')
    generate_tfrecord_for_cifar10('test')
  print('dataset ready')

if __name__ == '__main__':
  tf.app.run()

