# BRUNO: A Deep Recurrent Model for Exchangeable Data


Below are the negative log-likelihoods and bits per dimensions for the trained models (produced by the `test_nll.py` script). Due to tensorflow's randomness, these numbers could be a slightly different when one retrains a model. 
 

Model          | Test NLL(bpd)  | Test prior NLL(bpd) | Train NLL(bpd) | Train prior NLL(bpd)
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
an2_cifar_tp   | 7773.217(3.651) | 7793.673(3.660) | 7298.111(3.427) | 7295.626(3.426)
an2_cifar_gp   | 7781.070(3.654) | 7800.581(3.663) | 7340.239(3.447) | 7338.131(3.446)
cn2_fashion_tp | 1661.208(3.057) | 1671.190(3.075) | 1417.329(2.608) | 1418.707(2.611)
cn2_fashion_gp | 1635.426(3.009) | 1642.390(3.022) | 1436.831(2.644) | 1439.928(2.650)
bn2_omniglot_tp |537.097(0.988) | 546.987(1.007) | 438.032(0.806) | 447.527(0.824)
bn2_omniglot_gp | 537.043(0.988) | 544.449(1.002) | 434.759(0.800) | 443.724(0.817)
dn2_mnist_even_gp | 567.321(1.044) | 569.963(1.049) | 627.947(1.156) | 632.165(1.163)
dn2_mnist_even_tp | 567.653(1.045) | 570.687(1.050) | 621.790(1.144) | 626.456(1.153)
en2_noconv_mnist_even_tp | 932.178(1.715)  | 928.966(1.709)  | 916.042(1.686) | 914.340(1.683)
en2_noconv_mnist_even_gp | 1014.797(1.867) | 1019.180(1.875) | 914.720(1.683) | 918.399(1.690)
en2_noconv_mnist_tp      | 836.276(1.539) | 838.364(1.543) | 804.832(1.481) | 807.448(1.486)
en2_noconv_mnist_gp      | 848.537(1.561) | 852.761(1.569) | 807.956(1.487) | 811.818(1.494)


In the paper, we reported average *n*-shot, *k*-way accuracy from models with two different random seeds. 

Model          | n=1,k=5         | n=5,k=5         | n=1,k=20        | n=5,k=20
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
bn2_omniglot_tp| 0.881| 0.966 |0.726 | 0.898
bn2_omniglot_tp_s2 | 0.844 | 0.947 | 0.659 | 0.855
bn2_omniglot_tp_ft_1s_20w | 0.969 | 0.994 | 0.909 | 0.977
bn2_omniglot_tp_s2_ft_1s_20w |0.972 | 0.994 | 0.918 | 0.980

Few-shot accuracies for a *GP*-based model:

Model          | n=1,k=5         | n=5,k=5         | n=1,k=20        | n=5,k=20
-------------- | :-------------: | :-------------: | :-------------: | :-------------:
bn2_omniglot_gp| 0.865 |0.961|0.691|0.885
bn2_omniglot_gp_ft_1s_20w | 0.965 | 0.992 | 0.901 | 0.975