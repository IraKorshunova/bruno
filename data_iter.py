import numpy as np
import utils


class OmniglotExchSeqDataIterator(object):
    def __init__(self, seq_len, batch_size, set='train', valid_split=False, rng=None, augment=True, infinite=True):

        (x_train, y_train), (x_test, y_test), valid_classes = utils.load_omniglot()

        if set == 'train':
            self.x = x_train
            self.y = y_train
            if valid_split:
                train_idxs = np.where(np.isin(y_train, valid_classes, invert=True))[0]
                self.x = x_train[train_idxs]
                self.y = y_train[train_idxs]
        elif set == 'test':
            self.x = x_test
            self.y = y_test
        elif set == 'valid':
            valid_idxs = np.where(np.isin(y_train, valid_classes))[0]
            self.x = x_train[valid_idxs]
            self.y = y_train[valid_idxs]
        else:
            ValueError('which set?')

        self.input_dim = self.x.shape[-1]
        self.img_shape = (int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)), 1)
        self.x = np.reshape(self.x, (self.x.shape[0],) + self.img_shape)
        self.x = np.float32(self.x)

        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.y2idxs = {}
        self.nsamples = 0
        for i in list(self.classes):
            self.y2idxs[i] = np.where(self.y == i)[0]
            self.nsamples += len(self.y2idxs[i])

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.RandomState(42) if not rng else rng
        self.augment = augment
        self.infinite = infinite

        print(set, 'dataset size:', self.x.shape)
        print(set, 'N classes', self.n_classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.nsamples)
        print('--------------')

    def get_observation_size(self):
        return (self.seq_len,) + self.img_shape

    def generate(self, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng

        while True:
            x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')

            for i in range(self.batch_size):
                j = rng.choice(self.classes)
                idxs = self.y2idxs[j]
                rng.shuffle(idxs)
                rotation = rng.randint(0, 4)
                for k in range(self.seq_len):
                    x_batch[i, k, :] = self.x[idxs[k], :]
                    if self.augment:
                        x_batch[i, k] = np.rot90(x_batch[i, k], k=rotation, axes=(0, 1))

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch

            if not self.infinite:
                break

    def generate_each_digit(self, same_image=False, noise_rng=np.random.RandomState(42)):
        for i in list(self.classes):
            x_batch = np.zeros((1,) + self.get_observation_size(), dtype='float32')
            y_batch = np.zeros((1, self.seq_len), dtype='float32')

            idxs = self.y2idxs[i]
            assert len(idxs) >= self.seq_len
            self.rng.shuffle(idxs)

            rotation = self.rng.randint(0, 4)
            for k in range(self.seq_len):
                x_batch[0, k, :] = self.x[idxs[0], :] if same_image else self.x[idxs[k], :]
                if self.augment:
                    x_batch[0, k] = np.rot90(x_batch[0, k], k=rotation, axes=(0, 1))
                y_batch[0, k] = i

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch

    def generate_diagonal_roll(self, rng=None, same_class=True, same_image=False, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng
        batch_size = self.seq_len

        while True:
            x_batch = np.zeros((batch_size,) + self.get_observation_size(), dtype='float32')
            j = rng.choice(self.classes)
            idxs = self.y2idxs[j]
            assert len(idxs) >= self.seq_len
            rng.shuffle(idxs)

            sequence = np.zeros((1,) + self.get_observation_size(), dtype='float32')
            for k in range(self.seq_len):
                sequence[0, k] = self.x[idxs[k]]

            if not same_class:
                other_digits = list(self.classes)
                other_digits.remove(j)
                j2 = rng.choice(other_digits)
                idxs = self.y2idxs[j2]
                sequence[0, 0] = self.x[rng.choice(idxs)]

            sequence += noise_rng.uniform(size=sequence.shape)

            for i in range(batch_size):
                x_batch[i, :, :] = np.roll(sequence, i, axis=1)

            yield x_batch

            if not self.infinite:
                break


class OmniglotTestBatchSeqDataIterator(object):
    def __init__(self, seq_len, batch_size, set='test', rng=None, augment=False):

        (x_train, y_train), (x_test, y_test), _ = utils.load_omniglot()

        if set == 'train':
            self.x = x_train
            self.y = y_train
        elif set == 'test':
            self.x = x_test
            self.y = y_test
        else:
            self.x = np.concatenate((x_train, x_test))
            self.y = np.concatenate((y_train, y_test))

        self.input_dim = self.x.shape[-1]
        self.img_shape = (int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)), 1)
        self.x = np.reshape(self.x, (self.x.shape[0],) + self.img_shape)
        self.x = np.float32(self.x)

        self.classes = list(np.unique(self.y))
        self.n_classes = len(self.classes)
        self.y2idxs = {}
        self.nsamples = 0
        for i in self.classes:
            self.y2idxs[i] = np.where(self.y == i)[0]
            self.nsamples += len(self.y2idxs[i])

        self.seq_len = seq_len
        self.rng = np.random.RandomState(42) if not rng else rng
        self.nsamples = self.x.shape[0]
        self.batch_size = batch_size
        self.set = set
        self.augment = augment

        print(set, 'dataset size:', self.x.shape)
        print(set, 'N classes', self.n_classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.nsamples)
        print('--------------')

    def get_observation_size(self):
        return (self.seq_len,) + self.img_shape

    def generate(self, trial=0, **kwargs):

        rng = np.random.RandomState(trial)
        n_rotations = 4 if self.augment else 0
        n_batches = -1
        for i in self.classes:  # iterate over classes
            for rotation in range(n_rotations):  # rotate class
                x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')
                y_batch = np.zeros((self.batch_size, self.seq_len), dtype='float32')

                # choose test/ train images from this class
                img_idxs = rng.choice(self.y2idxs[i], size=self.seq_len, replace=False)
                for k in range(self.seq_len):
                    x_batch[0, k] = np.rot90(self.x[img_idxs[k]], k=rotation, axes=(0, 1))
                    y_batch[0, k] = i

                # choose other classes
                other_classes = list(self.classes)
                other_classes.remove(i)
                class_idxs = rng.choice(other_classes, size=self.batch_size - 1, replace=False)
                for bi, j in zip(range(1, self.batch_size), class_idxs):
                    img_idxs = rng.choice(self.y2idxs[j], size=self.seq_len, replace=False)
                    other_rotation = self.rng.randint(0, 4)
                    for k in range(self.seq_len):
                        x_batch[bi, k] = self.x[img_idxs[k]]
                        y_batch[bi, k] = j
                        if self.augment:
                            x_batch[bi, k] = np.rot90(x_batch[bi, k], k=other_rotation, axes=(0, 1))

                    x_batch[bi, -1] = np.copy(x_batch[0, -1])
                    y_batch[bi, -1] = np.copy(y_batch[0, -1])

                noise_sequence = rng.uniform(size=self.get_observation_size())
                for bi in range(x_batch.shape[0]):
                    x_batch[bi] += noise_sequence

                n_batches += 1
                yield x_batch, y_batch, n_batches


class OmniglotEpisodesDataIterator(OmniglotTestBatchSeqDataIterator):
    def __init__(self, seq_len, batch_size, meta_batch_size, set='train', rng=None, augment=True):
        super(OmniglotEpisodesDataIterator, self).__init__(seq_len, batch_size, set, rng, augment)
        self.meta_batch_size = meta_batch_size

    def generate(self, trial=0, rng=None):
        rng = self.rng if rng is None else rng

        while True:
            x_meta_batch = np.zeros((self.meta_batch_size, self.batch_size,) + self.get_observation_size(),
                                    dtype='float32')
            y_meta_batch = np.zeros((self.meta_batch_size, self.batch_size, self.seq_len), dtype='float32')

            for m in range(self.meta_batch_size):
                i = rng.choice(self.classes)
                rotation = self.rng.randint(0, 4)

                x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')
                y_batch = np.zeros((self.batch_size, self.seq_len), dtype='float32')

                # choose test/ train images from this class
                img_idxs = rng.choice(self.y2idxs[i], size=self.seq_len, replace=False)
                for k in range(self.seq_len):
                    x_batch[0, k] = np.rot90(self.x[img_idxs[k]], k=rotation, axes=(0, 1))
                    y_batch[0, k] = i

                # choose other classes
                other_classes = list(self.classes)
                other_classes.remove(i)
                class_idxs = rng.choice(other_classes, size=self.batch_size - 1, replace=False)
                for bi, j in zip(range(1, self.batch_size), class_idxs):
                    img_idxs = rng.choice(self.y2idxs[j], size=self.seq_len, replace=False)
                    other_rotation = self.rng.randint(0, 4)
                    for k in range(self.seq_len):
                        x_batch[bi, k] = self.x[img_idxs[k]]
                        y_batch[bi, k] = j
                        if self.augment:
                            x_batch[bi, k] = np.rot90(x_batch[bi, k], k=other_rotation, axes=(0, 1))

                    x_batch[bi, -1] = np.copy(x_batch[0, -1])

                noise_sequence = rng.uniform(size=self.get_observation_size())
                for bi in range(x_batch.shape[0]):
                    x_batch[bi] += noise_sequence

                x_meta_batch[m] = x_batch
                y_meta_batch[m] = y_batch

            x_meta_batch = np.reshape(x_meta_batch,
                                      (self.meta_batch_size * self.batch_size,) + self.get_observation_size())
            y_meta_batch = np.reshape(y_meta_batch, (self.meta_batch_size * self.batch_size, self.seq_len))

            yield x_meta_batch, y_meta_batch


class BaseExchSeqDataIterator(object):
    def __init__(self, seq_len, batch_size, dataset='mnist', set='train',
                 rng=None, infinite=True, digits=None):

        if dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = utils.load_fashion_mnist()
            if set == 'train':
                self.x = x_train
                self.y = y_train
            else:
                self.x = x_test
                self.y = y_test
        elif dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = utils.load_mnist()
            if set == 'train':
                self.x = x_train
                self.y = y_train
            elif set == 'test':
                self.x = x_test
                self.y = y_test
        elif dataset == 'cifar10':
            self.x, self.y = utils.load_cifar('data/cifar', subset=set)
            self.x = np.transpose(self.x, (0, 2, 3, 1))  # (N,3,32,32) -> (N,32,32,3)
            self.x = np.float32(self.x)
            self.img_shape = self.x.shape[1:]
            self.input_dim = np.prod(self.img_shape)
        else:
            raise ValueError('wrong dataset name')

        if dataset == 'mnist' or dataset == 'fashion_mnist':
            self.input_dim = self.x.shape[-1]
            self.img_shape = (int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)), 1)
            self.x = np.reshape(self.x, (self.x.shape[0],) + self.img_shape)
            self.x = np.float32(self.x)

        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.y2idxs = {}
        self.nsamples = 0
        for i in list(self.classes):
            self.y2idxs[i] = np.where(self.y == i)[0]
            self.nsamples += len(self.y2idxs[i])

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.RandomState(42) if not rng else rng
        self.infinite = infinite
        self.digits = digits if digits is not None else np.arange(self.n_classes)

        print(set, 'dataset size:', self.x.shape)
        print(set, 'N classes', self.n_classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.nsamples)
        print(set, 'digits', self.digits)
        print('--------------')

    def get_observation_size(self):
        return (self.seq_len,) + self.img_shape

    def generate(self, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng

        while True:
            x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')

            for i in range(self.batch_size):
                j = rng.randint(0, 10) if self.digits is None else rng.choice(self.digits)
                idxs = self.y2idxs[j]
                assert len(idxs) >= self.seq_len
                rng.shuffle(idxs)

                for k in range(self.seq_len):
                    x_batch[i, k, :] = self.x[idxs[k], :]

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch

            if not self.infinite:
                break

    def generate_each_digit(self, same_image=False, noise_rng=np.random.RandomState(42)):
        for i in range(10):
            x_batch = np.zeros((1,) + self.get_observation_size(), dtype='float32')
            y_batch = np.zeros((1, self.seq_len), dtype='float32')

            idxs = self.y2idxs[i]
            assert len(idxs) >= self.seq_len
            self.rng.shuffle(idxs)

            for k in range(self.seq_len):
                x_batch[0, k, :] = self.x[idxs[0], :] if same_image else self.x[idxs[k], :]
                y_batch[0, k] = i

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch

    def generate_anomaly(self, noise_rng=np.random.RandomState(42)):
        while True:
            x_batch = np.zeros((1,) + self.get_observation_size(), dtype='float32')
            y_batch = np.zeros((1, self.seq_len), dtype='float32')

            j = self.rng.randint(0, 10) if self.digits is None else self.rng.choice(self.digits)
            idxs = self.y2idxs[j]
            assert len(idxs) >= self.seq_len
            self.rng.shuffle(idxs)

            for k in range(self.seq_len):
                x_batch[0, k, :] = self.x[idxs[k], :]
                y_batch[0, k] = j

            # true anomaly
            j = self.rng.randint(0, 10) if self.digits is None else self.rng.choice(self.digits)
            idx = self.rng.choice(self.y2idxs[j])
            x_batch[0, self.seq_len - 5, :] = self.x[idx, :]
            y_batch[0, self.seq_len - 5] = j

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch

            if not self.infinite:
                break

    def generate_diagonal_roll(self, same_class=True, same_image=False, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng
        batch_size = self.seq_len

        while True:
            x_batch = np.zeros((batch_size,) + self.get_observation_size(), dtype='float32')

            j = rng.randint(0, 10) if self.digits is None else rng.choice(self.digits)
            idxs = self.y2idxs[j]
            assert len(idxs) >= self.seq_len
            rng.shuffle(idxs)

            sequence = np.zeros((1,) + self.get_observation_size(), dtype='float32')
            for k in range(self.seq_len):
                if same_image:
                    sequence[0, k] = self.x[idxs[0]]
                else:
                    sequence[0, k] = self.x[idxs[k]]

            if not same_class:
                other_digits = list(self.digits)
                other_digits.remove(j)
                j2 = rng.choice(other_digits)
                idxs = self.y2idxs[j2]
                sequence[0, 0] = self.x[rng.choice(idxs)]

            sequence += noise_rng.uniform(size=sequence.shape)

            for i in range(batch_size):
                x_batch[i] = np.roll(sequence, i, axis=1)

            yield x_batch

            if not self.infinite:
                break

    def generate_1class(self, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng

        while True:
            x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')

            for i in range(self.batch_size):
                j = 0
                idxs = self.y2idxs[j]
                assert len(idxs) >= self.seq_len
                rng.shuffle(idxs)

                for k in range(self.seq_len):
                    x_batch[i, k, :] = self.x[idxs[k], :]

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch

            if not self.infinite:
                break


class BaseTestBatchSeqDataIterator(object):
    def __init__(self, seq_len, set='train', dataset='mnist', rng=None, infinite=True, digits=None):

        if dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = utils.load_fashion_mnist()
        elif dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = utils.load_mnist()
        else:
            raise ValueError('wrong dataset name')

        self.x_train = x_train
        self.y_train = y_train

        self.y_train2idxs = {}
        for i in range(10):
            self.y_train2idxs[i] = np.where(self.y_train == i)[0]

        if set == 'train':
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_test
            self.y = y_test

        self.input_dim = self.x.shape[-1]
        self.img_shape = (int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)), 1)
        self.x = np.reshape(self.x, (self.x.shape[0],) + self.img_shape)
        self.x = np.float32(self.x)

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0],) + self.img_shape)
        self.x_train = np.float32(self.x_train)

        self.y2idxs = {}
        for i in range(10):
            self.y2idxs[i] = np.where(self.y == i)[0]

        self.seq_len = seq_len
        self.rng = np.random.RandomState(42) if not rng else rng
        self.nsamples = self.x.shape[0]
        self.infinite = infinite
        self.digits = digits if digits is not None else range(10)
        self.n_classes = len(self.digits)
        self.batch_size = self.n_classes
        self.set = set

        print(set, 'dataset size:', self.x.shape)
        print(set, 'N classes', self.n_classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.nsamples)
        print('--------------')

    def get_observation_size(self):
        return (self.seq_len,) + self.img_shape

    def generate(self, trial=0, n_random_samples=1, condition_on_train=False):

        rng = np.random.RandomState(trial)

        x_batch = np.zeros((self.batch_size,) + self.get_observation_size(), dtype='float32')
        y_batch = np.zeros((self.batch_size, self.seq_len), dtype='float32')

        idxs_used = []
        for i in range(self.batch_size):
            idxs = self.y_train2idxs[self.digits[i]] if condition_on_train else self.y2idxs[self.digits[i]]
            idxs_cond = idxs[trial * self.seq_len: (trial + 1) * self.seq_len]
            x_batch[i, :, :] = self.x_train[idxs_cond] if condition_on_train else self.x[idxs_cond]
            idxs_used.extend(idxs_cond[:-1])
            y_batch[i, :] = self.digits[i]

        idxs_used = [] if condition_on_train else idxs_used

        noise_sequence = rng.uniform(size=self.get_observation_size())
        x_batch += noise_sequence[None, :, :]

        for i in range(self.nsamples):
            if i not in idxs_used and self.y[i] in self.digits:
                for _ in range(n_random_samples):
                    test_noise = rng.uniform(size=self.img_shape)
                    for k in range(self.batch_size):
                        x_batch[k, -1, :] = self.x[i] + test_noise
                        y_batch[k, -1] = self.y[i]

                    yield x_batch, y_batch, i
