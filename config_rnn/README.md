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
bn_omniglot_gp3 softplus     | 534.376 | 543.175 | 439.676 | 450.237
bn_omniglot_gp2 sqr          | 542.093 |  548.312 | 444.239 | 452.292
bn_omniglot_gp  softplus_sqr | 536.452 | 542.575 | 437.537 | 445.650
bn_omniglot_tp2 sqr          | 530.726 |  540.828 | 441.630 | 451.857
bn_omniglot_tp softplus sqr  | 532.406 | 541.793 | 439.014  | 449.153




Model          | Test            | Test prior      | Train           | Train prior 
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
an_cifar_tp3 softplus    | 7789.359 | 7811.993  | 7314.296 | 7312.620
an_cifar_tp2 sqr         | 7812.924 |  7834.504 | 7342.390 | 7341.136
an_cifar_tp softplus_sqr | 7778.768 |  7800.585 | 7303.793 | 7302.401



Model          | Test            | Test prior      | Train           | Train prior 
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
cn_fashion_tp_sp | 1672.030 | 1684.074 | 1421.806 | 1423.788
cn_fashion_tp_sqr | 1659.947 |  1670.360 | 1426.245 |  1427.708
cn_fashion_tp_sqr_sp | 1647.122 | 1656.965 | 1422.699 | 1424.085

