from collections import defaultdict

import numpy as np

import utils_conditional


class ShapenetConditionalNPDataIterator():
    def __init__(self, seq_len, batch_size, set='train', rng=None):
        self.x, self.y, self.info = utils_conditional.load_shapenet(set)

        self.n_samples = len(self.x)
        self.img_shape = self.x.shape[1:]

        self.classes = np.unique(self.y)
        print(set, self.classes)
        self.y2idxs = defaultdict(list)
        for i in range(self.n_samples):
            self.y2idxs[self.y[i]].append(i)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = rng
        self.set = set

        print(set, 'dataset size:', self.x.shape)
        print(set, 'classes', self.classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.n_samples)
        print('--------------')

    def process_angle(self, x):
        angle_rad = x * np.pi / 180
        return np.sin(angle_rad), np.cos(angle_rad)

    def deprocess_angle(self, x):
        x1, x2 = x
        angle = np.arctan2(x1, x2) * 180 / np.pi
        angle += 360 if angle < 0 else 0
        return angle

    def get_label_size(self):
        return (self.seq_len, self.info.shape[-1])

    def get_observation_size(self):
        return (self.seq_len,) + self.img_shape

    def generate(self, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng

        while True:
            x_batch = np.zeros((self.batch_size, self.seq_len,) + self.img_shape, dtype='float32')
            y_batch = np.zeros((self.batch_size,) + self.get_label_size(), dtype='float32')

            for i in range(self.batch_size):
                c = rng.choice(self.classes)
                img_idxs = rng.choice(self.y2idxs[c], size=self.seq_len, replace=False)

                for j in range(self.seq_len):
                    x_batch[i, j] = self.x[img_idxs[j]]
                    y_batch[i, j] = self.info[img_idxs[j]]

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch

    def generate_each_digit(self, rng=None, noise_rng=None, random_classes=False):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng
        if random_classes:
            rng.shuffle(self.classes)
        for c in self.classes:
            x_batch = np.zeros((self.batch_size, self.seq_len,) + self.img_shape, dtype='float32')
            y_batch = np.zeros((self.batch_size,) + self.get_label_size(), dtype='float32')

            for i in range(self.batch_size):
                img_idxs = rng.choice(self.y2idxs[c], size=self.seq_len, replace=False)

                for j in range(self.seq_len):
                    x_batch[i, j] = self.x[img_idxs[j]]
                    y_batch[i, j] = self.info[img_idxs[j]]

            x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch
