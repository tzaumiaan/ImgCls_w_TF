# Image classification with Tensorflow
This practical exercise aims to make me as a learner understand several topics 
when using Tensorflow to do image classification.
Since there are so many available codes doing the same thing, including the
[Tensorflow tutorials for MNIST](https://www.tensorflow.org/tutorials/)
[Tensorflow tutorials for CIFAR10](https://www.tensorflow.org/tutorials/images/deep_cnn)
and others,
it seems to me they are either pure Tensorflow implementation with very high-level maturity and
all those non-trivial advanced coding frameworks,
or just using Tensorflow to glue high-level APIs like Keras.
Therefore I started this exercise to get familiar with all those basic tools Tensorflow provide
to make me have a solid idea how things work,
which might be necessary while reading more and more Tensorflow implementations.
Topics include:
- To learn how to package data into `tfrecord` files
- Using `tf.data.Dataset` to implement to data pipeline for training, validation, and testing sets 
  with k-fold cross validation
- To model neural network with `tf.contrib.slim`, which is an easy-to-use and clean package of common neural layers
- To build a graph for model inference, loss & predection computation, and the optimization process (or trainer).
- To save a trained model as `checkpoint` files, 
  with inputs as `tf.placeholder` and necessary operators stored by `tf.add_to_collection`,
  and later loaded back to make inference.
  
## Versions Tested
- Python 3.5 or 3.6
- CUDA 9.0
- Tensorflow-GPU 1.12 (with cuDNN 7.1.4) or 1.9 (with cuDNN 7.0.5)
- NumPy 1.15

## Usage
Currently two image datasets
([MNIST](http://yann.lecun.com/exdb/mnist/) and 
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html))
and their corresponding models are implemented.

It is suggested to use the virtual environment wrapper for python packages.
So the following commands are under such virtual environment,
which runs the Python 3 with simply `python` commands

First download the dataset,
note the argument `dataset` can be either `cifar10` or `mnist`.
```
python download_data.py --dataset=cifar10
```
After this step, `train.tfrecord` and `test.tfrecord` will appear in `data` folder.

Then train the dataset,
note the argument `dataset` should be chosen the same as previous step.
Arguments also include batch size (`batch_size`), initial learning rate (`init_lr`), number of epochs (`num_epochs`),
which are free to choose.
By default the checkpoints will be stored in `train` folder,
which is also adaptable with argument `log_dir`.
```
python train.py --dataset=cifar10 
```
During the training, you can use Tensorboard on the other terminal to inspect the training details.
```
tensorboard --logdir=train
```
By default it will show up at `http://localhost:6006`.

After training, the model graph and checkpoints are stored in `log_dir`.
Finally test the model with test dataset.
Use argument `cpkt_path` to specify the model we would like to test.
```
python test.py --dataset=cifar10 --ckpt_path=train/cifar10_bs_50_lr_0.1_l2s_1e-08/model_epoch20.ckpt
```
For sanity check, CIFAR10 can get around 70% accuracy with default model and hyperparameters.

Note: Running with CUDA-based GPU requires no extra changes in codes or running procedures.
Refer to [this](
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)
for checking if CUDA driver and cuDNN is properly installed and configured.

## Future work
- To learn how to freeze the model into `.pb` file
- To visualize training versus validation for each epoch

## Useful references
- [Finetune AlexNet with Tensorflow](https://github.com/kratzert/finetune_alexnet_with_tensorflow),
  an implementation with nicely structured training graph as an implementation example
- [TensorFlow: saving/restoring and mixing multiple models](
  https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125),
  a clear explanation on how model checkpoints are saved and stored
- [Stanford CS 20: Tensorflow for Deep Learning Research](
  https://web.stanford.edu/class/cs20si/),
  the course which aims to explore more details about the practical aspects of Tensorflow
  
