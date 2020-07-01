# Conditional BRUNO

### ShapeNet dataset

   *1.* Download the dataset from
   from https://drive.google.com/file/d/1y_-FcpKwPCOihizbQG0XqRxbg8lDUekz/view?usp=sharing.
   
   *2.* In `/data` create a new directory called `shapenet_12classes` and unzip the files:
 ```
 data/shapenet_12classes/02691156
 data/shapenet_12classes/03001627
 etc.
```

   *3.* Run `python3 utils_conditional.py` to get 12 numpy files:
    
```
 data/shapenet_12classes/02691156.npy
 data/shapenet_12classes/03001627.npy
 etc.
```  

Note: most of the code for ShapeNet is taken from Versa: [github.com/Gordonjo/versa](https://github.com/Gordonjo/versa), which is also a nice meta-learning model.

### Training and testing

There is a configuration file `m1_shapenet_b.py` in `config_conditional` for the model used in the paper.
Below are the examples on how to train and test this model.   

**Training (supports multiple gpus)**
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m config_conditional.train  --config_name m1_shapenet_b --nr_gpu 2
```

**Generating samples**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_conditional.test_samples  --config_name m1_shapenet_b --seq_len 13 --n_context 1
```

**Generating samples from the prior**

```
CUDA_VISIBLE_DEVICES=0 python3 -m config_conditional.test_samples_prior  --config_name m1_shapenet_b
``` 