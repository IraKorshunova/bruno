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




Model          | Test            | Test prior      | Train           | Train prior 
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
cn2_fashion_tp | 1661.209(3.057) | 1671.190        | 1417.329(2.608) | 1418.707
cn2_fashion_gp | 1635.426(3.009) | 1642.390        | 1436.831(2.644) | 1439.928



Model          | Test            | Test prior      | Train           | Train prior 
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
bn2_omniglot_tp | 537.097 (0.988) | 546.987 | 438.032 (0.806) | 447.527
bn2_omniglot_gp | 537.043 (0.988) | 544.449 | 434.759 (0.800) | 443.724



restoring parameters from metadata/dn2_mnist_even_tp-2018_08_15/params.ckpt
Sequence length: 20
Test Loss 567.653
Bits per dim 1.045
Test loss under prior 570.687
Train Loss 621.790
Bits per dim 1.144
Train loss under prior 626.456

restoring parameters from metadata/dn2_mnist_even_gp-2018_08_15/params.ckpt
Sequence length: 20
Test Loss 567.321
Bits per dim 1.044
Test loss under prior 569.963
Train Loss 627.947
Bits per dim 1.156
Train loss under prior 632.165


bn2_omniglot_tp-2018_08_08_test_class_20_2_20  0.7263593380614657  +  0.6568262411347517 = 0.6915927895981087
bn2_omniglot_tp-2018_08_08_test_class_20_2_5   0.8811170212765959 + 0.8438829787234041   = 0.8625
bn2_omniglot_tp-2018_08_08_test_class_20_6_20  0.89822695035461 +  0.8552009456264775    = 0.8767139479905437
bn2_omniglot_tp-2018_08_08_test_class_20_6_5   0.9656028368794326  + 0.947163120567376   = 0.9563829787234043


bn2_omniglot_tp_ft_1s_20w-2018_08_13_test_class_20_2_20 0.9085697399527188 + 0.9178486997635933 =  0.9132092198581561
bn2_omniglot_tp_ft_1s_20w-2018_08_13_test_class_20_2_5  0.9692080378250593 + 0.9719858156028369 = 0.970596926713948  
bn2_omniglot_tp_ft_1s_20w-2018_08_13_test_class_20_6_20 0.9766548463356974 + 0.9801418439716313 = 0.9783983451536644
bn2_omniglot_tp_ft_1s_20w-2018_08_13_test_class_20_6_5  0.9935874704491727 + 0.9937647754137116 = 0.9936761229314421


---- GP ----

bn2_omniglot_gp-2018_08_11_test_class_20_2_20 0.6912234042553191
bn2_omniglot_gp-2018_08_11_test_class_20_2_5  0.8653368794326243
bn2_omniglot_gp-2018_08_11_test_class_20_6_20 0.8848995271867613
bn2_omniglot_gp-2018_08_11_test_class_20_6_5  0.9607565011820333


bn2_omniglot_gp_ft_1s_20w-2018_08_13_test_class_20_2_20 0.9009456264775413
bn2_omniglot_gp_ft_1s_20w-2018_08_13_test_class_20_2_5  0.9647163120567376
bn2_omniglot_gp_ft_1s_20w-2018_08_13_test_class_20_6_20 0.9747340425531915
bn2_omniglot_gp_ft_1s_20w-2018_08_13_test_class_20_6_5  0.9923167848699764