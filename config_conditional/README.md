# Conditional BRUNO

### ShapeNet dataset

   *1.* Download the two shapenet dataset files 02691156.zip and 03001627.zip
   from https://drive.google.com/drive/folders/1x4EZFEE_bT9lvBu25ZnsMtV4LNKhYaG5?usp=sharing.
   
   *2.* In `/data` create a new directory called `shapenet` and unzip the 2 files:
 ```
 data/shapenet/02691156
 data/shapenet/03001627
```

   *3.* Run `python3 utils_conditional.py` to get the two .npy files:
    
```
 data/shapenet/shapenet_chairs.npy
 data/shapenet/shapenet_planes.npy
```  

Note: most of the code for ShapeNet is taken from Versa: [github.com/Gordonjo/versa](https://github.com/Gordonjo/versa), which is also a nice meta-learning model.

### Training and testing

There is a configuration file `m1_shapenet.py` in `config_conditional` for the model used in the paper.
Below are the examples on how to train and test this model.   

**Training (supports multiple gpus)**
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_conditional.train  --config_name m1_shapenet --nr_gpu 2
```

**Generating samples**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_conditional.test_samples  --config_name m1_shapenet --seq_len 13 --n_context 1
```

**Generating samples from the prior**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_conditional.test_samples_prior  --config_name m1_shapenet
```