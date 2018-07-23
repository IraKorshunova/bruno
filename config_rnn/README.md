# BRUNO: A Deep Recurrent Model for Exchangeable Data


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
c_mnist_dense | 684.936 | 801.075 |  |
c_mnist_dense_gp | 1016.743 | 1016.743 | |
c_mnist_dense_wn | 721.111 | 817.574 | 817.171 | 723.703
c_mnist_dense_gp_wn |728.531 |797.308 | 810.478 | 733.307

c_fashion_mnist_dense_gp_wn |1623.750| 2515.585| 2540.207 | 1624.647
c_fashion_mnist_dense_wn | 1611.253 |  2217.091 |  2374.920 | 1612.622


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
a_cifar_wn_elu | 7359.661 (3.456 bpd)| 7789.363 (3.658 bpd) |  7803.375 |
a_cifar_gp_wn_elu |7329.147| 7783.228| 7798.605 | 7329.147


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
c_mnist_dense_even | 791.944 |927.0195 | 942.5439 
c_mnist_dense_even_wn | 824.220 |934.635 | 955.8727 | 824.278
c_mnist_dense_gp_even_wn | 841.385  | 1260.406   | 1254.100 | 844.031


Model | Train NLL | Test NLL | Test NLL under prior | Train NLL under prior 
------------ | :-------------: | :-------------: | :-------------: | :-------------:
b_omniglot_wn | 429.048 | 534.055 | 543.826 | 435.637
b_omniglot_wn_elu | 450.580 | 539.713 | 549.890 |  459.689        
b_omniglot_gp_wn |450.701| 518.627| 526.617 | 460.579


b_omniglot_wn_elu
20 way 1-shot
0.7510933806146571

b_omniglot_gp_wn_elu
0.779403073286052
