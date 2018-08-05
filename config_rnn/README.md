# BRUNO: A Deep Recurrent Model for Exchangeable Data


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
a_cifar_gp_wn_elu |7329.147| 7783.228| 7798.605 | 7329.147
a_cifar_wn_exp_nu | 7299.273 | 7757.062 | 7764.780 | 7293.519


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
c_mnist_even_gp |618.014| 581.818| 582.389 | 623.311
c_mnist_even_exp_nu |628.273|563.630|567.262|632.574


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
d_mnist_even_gp |915.827  | 993.393 |  999.477 | 915.827
d_mnist_even_exp_nu | 916.381 | 909.871 | 909.250 | 914.604


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:      
b_omniglot_gp_wn     |450.701  | 518.627 | 526.617 | 460.579
b_omniglot_wn_exp_nu | 454.110 | 541.375 | 551.029 | 463.803



Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:      
a_fashion_mnist_gp |1428.178| 1684.472| 1685.691 | 1431.016
a_fashion_mnist_exp_nu | 1411.402 | 1679.955 | 1685.874 | 1413.315



Model          | Test            | Test prior      | Train           | Train prior 
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
bn_omniglot_gp  softplus_sqr | 536.452 | 542.575 | 437.537 | 445.650
bn_omniglot_gp2 sqr          | 542.093 |  548.312 | 444.239 | 452.292
bn_omniglot_tp softplus sqr  | 532.406 | 541.793 | 439.014 | 449.153
bn_omniglot_tp2 sqr  | 530.726 |  540.828 | 441.630 | 451.857




restoring parameters from metadata/en_nice_mnist_even_gp2-2018_08_03/params.ckpt
Sequence length: 20
Test Loss 982.953
Bits per dim 1.809
Test loss under prior 987.637
Train Loss 916.754
Bits per dim 1.687
Train loss under prior 920.563

restoring parameters from metadata/en_nice_mnist_even_gp-2018_08_03/params.ckpt
Sequence length: 20
Test Loss 983.236
Bits per dim 1.809
Test loss under prior 989.850
Train Loss 916.023
Bits per dim 1.686
Train loss under prior 919.909


restoring parameters from metadata/en_nice_mnist_even2-2018_08_03/params.ckpt
Sequence length: 20
Test Loss 914.678
Bits per dim 1.683
Test loss under prior 914.938
Train Loss 916.900
Bits per dim 1.687
Train loss under prior 915.105

restoring parameters from metadata/en_nice_mnist_even-2018_08_03/params.ckpt
Sequence length: 20
Test Loss 912.581
Bits per dim 1.679
Test loss under prior 913.962
^[^[Train Loss 914.758
Bits per dim 1.683
Train loss under prior 913.115
