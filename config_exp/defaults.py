# do not change these parameters here
seq_len = 20
seq_len_mnist = 32
eval_only_last = False
eps_corr = None
mask_dims = False


def set_parameters(args):
    params_dict = vars(args)
    global seq_len
    global seq_len_mnist
    if 'seq_len' in params_dict:
        seq_len = params_dict['seq_len']
        seq_len_mnist = params_dict['seq_len']

    global eval_only_last
    if 'eval_only_last' in params_dict:
        eval_only_last = bool(params_dict['eval_only_last'])

    global eps_corr
    if 'eps_corr' in params_dict:
        eps_corr = params_dict['eps_corr']

    global mask_dims
    if 'mask_dims' in params_dict:
        mask_dims = bool(params_dict['mask_dims'])
