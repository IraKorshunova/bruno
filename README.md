# BRUNO: A Deep Recurrent Model for Exchangeable Data

This is an official implementation for reproducing the results of [BRUNO: A Deep Recurrent Model for Exchangeable Data](https://arxiv.org/abs/1802.07535)

### Requirements

The code was used with the following settings:

- tensorflow-gpu==1.7.0
- scikit-image==0.13.1

### Datasets

Below we list files for every dataset that should be stored in a `data/` directory inside a project folder.


**MNIST** 

Download from [yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

```
 data/train-images-idx3-ubyte.gz
 data/train-labels-idx1-ubyte.gz
 data/t10k-images-idx3-ubyte.gz
 data/t10k-labels-idx1-ubyte.gz
```


**Fashion MNIST**

Download from [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

```
data/fashion_mnist/train-images-idx3-ubyte.gz
data/fashion_mnist/train-labels-idx1-ubyte.gz
data/fashion_mnist/t10k-images-idx3-ubyte.gz
data/fashion_mnist/t10k-labels-idx1-ubyte.gz
```

**Omniglot**

Download and unzip files from  [github.com/brendenlake/omniglot/tree/master/python](https://github.com/brendenlake/omniglot/tree/master/python ) 

```
data/images_background
data/images_evaluation
```

Download .pkl files from [github.com/renmengye/few-shot-ssl-public#omniglot](https://github.com/renmengye/few-shot-ssl-public#omniglot). These are used to make train-test-validation split.

```
data/train_vinyals_aug90.pkl
data/test_vinyals_aug90.pkl
data/val_vinyals_aug90.pkl
```

Run `utils.py` to preprocess Omniglot images

```
data/omniglot_x_train.npy
data/omniglot_y_train.npy
data/omniglot_x_test.npy
data/omniglot_y_test.npy
data/omniglot_valid_classes.npy
``` 

**CIFAR-10**

This dataset will be downloaded directly with the first call to CIFAR-10 models.

```
data/cifar/cifar-10-batches-py
```


### Training and testing

There are configuration files in `config_rnn` for every model we used in the paper
and a bunch of testing scripts. Below are examples on how to train and test Omniglot models.   

**Training (supports multi-gpu)**
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_rnn.train  --config_name bn2_omniglot_tp --nr_gpu 2
```

**Fine-tuning (to be used on one gpu only)**
```
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.train_finetune  --config_name bn2_omniglot_tp_ft_1s_20w
```

**Generating samples**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_samples  --config_name bn2_omniglot_tp_ft_1s_20w
```

**Few-shot classification**
   
```
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot  --config_name bn2_omniglot_tp --seq_len 2 --batch_size 20
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot  --config_name bn2_omniglot_tp_ft_1s_20w --seq_len 2 --batch_size 20
```
Here, `batch_size = k` and `seq_len = n + 1` to test the model in a *k*-way, *n*-shot setting.


### Acknowledgments

Most of the code for Real NVP was adapted from [github.com/taesung89/real-nvp](https://github.com/taesung89/real-nvp). 

Weight normalization code was taken from [github.com/openai/pixel-cnn](https://github.com/openai/pixel-cnn).





