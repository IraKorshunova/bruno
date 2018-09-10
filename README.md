# BRUNO: A Deep Recurrent Model for Exchangeable Data

This is an official implementation for reproducing the results of   
[BRUNO: A Deep Recurrent Model for Exchangeable Data](https://arxiv.org/abs/1802.07535)

### Dependencies

The code was used with the following settings:

python3
CUDA 9.0, V9.0.176
scikit-image==0.13.1
six==1.10.0
tensorflow-gpu==1.7.0
matplotlib==2.2.0

### Datasets

MNIST: http://yann.lecun.com/exdb/mnist/
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
Omngilot: 
CIFAR-10: there is a code that will download it



### Training

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_rnn.train  --config_name a_cifar_wn_elu --nr_gpu 2
s
```

For any question, please feel free to contact Ira Korshunova (irene.korshunova@gmail.com)

### Acknowledgements

Most of the code for Real NVP was adapted from [github.com/taesung89/real-nvp](https://github.com/taesung89/real-nvp)

### References





