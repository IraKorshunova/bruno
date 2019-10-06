# BRUNO: A Deep Recurrent Model for Exchangeable Data

This is an official code for reproducing the main results from our NIPS'18 paper:

I. Korshunova, J. Degrave, F. Huszár, Y. Gal, A. Gretton, J. Dambre<br>
**BRUNO: A Deep Recurrent Model for Exchangeable Data** <br>
[arxiv.org/abs/1802.07535](https://arxiv.org/abs/1802.07535)

and from our NIPS'18 Bayesian Deep Learning workshop paper:

I. Korshunova, Y. Gal, J. Dambre, A. Gretton<br>
**Conditional BRUNO: A Deep Recurrent Process for Exchangeable Labelled Data**
[bayesiandeeplearning.org/2018/papers/40.pdf](http://bayesiandeeplearning.org/2018/papers/40.pdf) 


### Requirements

The code was used with the following settings:

- python3
- tensorflow-gpu==1.7.0
- scikit-image==0.13.1
- numpy==1.14.2
- scipy==1.0.0

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

**Training (supports multiple gpus)**
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


### Citation

Please cite our paper when using this code for your research. If you have any questions, please send me an email at `irene.korshunova@gmail.com`

```
@incollection{bruno2018,
    title = {BRUNO: A Deep Recurrent Model for Exchangeable Data},
    author = {Korshunova, Iryna and Degrave, Jonas and Huszar, Ferenc and Gal, Yarin and Gretton, Arthur and Dambre, Joni},
    booktitle = {Advances in Neural Information Processing Systems 31},
    year = {2018}
}
```



