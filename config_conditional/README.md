# Conditional BRUNO

This is an official code for reproducing the ShapeNet experiments from our NIPS'18 Bayesian Deep Learning workshop paper:

I. Korshunova, Y. Gal, J. Dambre, A. Gretton<br>
**Conditional BRUNO: A Deep Recurrent Process for Exchangeable Labelled Data** 


### Requirements

The code was used with the following settings:

- tensorflow-gpu==1.7.0


### Shapenet dataset

   1. Download the two shapenet dataset files 02691156.zip and 03001627.zip
   from https://drive.google.com/drive/folders/1x4EZFEE_bT9lvBu25ZnsMtV4LNKhYaG5?usp=sharing.
   2. In the data directory create a new directory called shapenet and unzip the 2 files:

 ```
 data/shapenet/02691156
 data/shapenet/03001627
```


    3. Run `utils_conditional.py` to process the images to get 2 .npy files:
    
```
 data/shapenet/ 
 data/shapenet/
```  


### Training and testing

There are configuration files in `config_rnn` for every model we used in the paper
and a bunch of testing scripts. Below are examples on how to train and test Omniglot models.   

**Training (supports multi-gpu)**
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_rnn.train  --config_name bn2_omniglot_tp --nr_gpu 2
```

**Generating samples**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_samples  --config_name bn2_omniglot_tp_ft_1s_20w
```