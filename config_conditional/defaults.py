# do not change these parameters here
seq_len = 16
n_context = 1


def set_parameters(args):
    params_dict = vars(args)
    global seq_len
    global n_context
    if 'seq_len' in params_dict:
        seq_len = params_dict['seq_len']

    if 'n_context' in params_dict:
        n_context = params_dict['n_context']
