import matplotlib
import numpy as np

from data_iter import BaseExchSeqDataIterator

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_omniglot_batch():
    rng = np.random.RandomState(42)

    data_iter = BaseExchSeqDataIterator(seq_len=10,
                                        batch_size=5,
                                        set='test',
                                        rng=rng)

    for iteration, x_batch in zip(range(10), data_iter.generate_diagonal_roll(same_class=False)):
        n_samples, seq_len = x_batch.shape[0], x_batch.shape[1]
        sample_plt = x_batch.swapaxes(1, 2)
        sample_plt = sample_plt.reshape((n_samples * 28, seq_len * 28, 1))

        print(np.max(x_batch), np.min(x_batch))

        fig = plt.figure()
        a = fig.add_subplot(2, 1, 2)
        img = sample_plt[:, :, 0]
        # print img
        plt.imshow(img, cmap='gray', interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('/mnt/storage/users/ikorshun/bruno/metadata/' + '/%0d_tile_%slen.png' % (
            iteration, seq_len),
                    bbox_inches='tight', pad_inches=0, format='png')


if __name__ == '__main__':
    test_omniglot_batch()
