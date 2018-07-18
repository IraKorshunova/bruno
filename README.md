# BRUNO: A Deep Recurrent Model for Exchangeable Data

https://arxiv.org/abs/1802.07535


Real NVP implementation was adapted from https://github.com/taesung89/real-nvp



Requirements:
python3
tensorflow-gpu 1.7.0

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_rnn.train  --config_name a_cifar_wn_elu --nr_gpu 2

```