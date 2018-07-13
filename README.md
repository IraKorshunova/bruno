# bruno_6

Epoch 39, time = 136s, train bits_per_dim = 0.2061, test bits_per_dim = 3.0644

samples have some weird blobs sometimes

# bruno_6_nobn

Epoch 39, time = 72s, train bits_per_dim = 0.5112, test bits_per_dim = 3.3989

samples are quite noisy


# todo

- try elu nonlinearity
- why degrees of freedom are so high?



# Good models

bruno_fashion_mnist2

# bruno_mnist_dense_wn 

didn't work too well: samples are okay, not too good for classification

Test Loss 912.918701171875
Bits per dim 1.6799276567799204
Train Loss 888.7155151367188
Bits per dim 1.6353896255724805

# bruno_mnist_wn

samples are better, sometimes a wrong digit

Test Loss 574.2996826171875
Bits per dim 1.0568103368570434
Train Loss 528.000732421875
Bits per dim 0.9716122936872188



CUDA_VISIBLE_DEVICES=2 bruno_relu_dwn1_omniglot2_aug_ft_pp5_20 eira // 16 K
CUDA_VISIBLE_DEVICES=1 bruno_relu_dwn1_omniglot2_aug_ft_pp1_5  wira  /// 14K
CUDA_VISIBLE_DEVICES=0 config_name bruno_relu_dwn1_omniglot2_aug_ft_pp2 rira // 22K
CUDA_VISIBLE_DEVICES=1 bruno_relu_dwn1_omniglot2_aug_ft_pp1_5  wira  /// 14K



CUDA_VISIBLE_DEVICES=3 bruno_relu_dwn1_omniglot2_aug_ft_pp5_5_all  qira /// 50K
CUDA_VISIBLE_DEVICES=3 bruno_relu_dwn1_omniglot2_aug_ft_pp1_5_all  wira /// 40K
CUDA_VISIBLE_DEVICES=2 bruno_relu_dwn1_omniglot2_aug_ft_pp5_20_all eira // 50 K
CUDA_VISIBLE_DEVICES=2 bruno_relu_dwn1_omniglot2_aug_ft_pp1_20_all rira // 40 K


