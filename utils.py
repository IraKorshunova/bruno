import glob
import gzip
import os
import pickle as pkl
import sys
import tarfile

import numpy as np
import scipy.misc
from six.moves import urllib


def autodir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def find_model_metadata(metadata_dir, config_name):
    metadata_paths = glob.glob(metadata_dir + '/%s-*/' % config_name)
    print(metadata_dir, config_name, metadata_paths)
    if not metadata_paths:
        raise ValueError('No metadata files for config %s' % config_name)
    elif len(metadata_paths) > 1:
        raise ValueError('Multiple metadata files for config %s' % config_name)
    return metadata_paths[0]


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return np.float32(data)


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def load_mnist():
    X_train = load_mnist_images('data/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist():
    X_train = load_mnist_images('data/fashion_mnist/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/fashion_mnist/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('data/fashion_mnist/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/fashion_mnist/t10k-labels-idx1-ubyte.gz')
    return (X_train, y_train), (X_test, y_test)


def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000, 3, 32, 32)), 'y': np.array(d['labels']).astype(np.uint8)}


def load_cifar(data_dir, subset='train'):
    download_and_extract_cifar(data_dir)
    if subset == 'train':
        train_data = [unpickle(os.path.join(data_dir, 'cifar-10-batches-py', 'data_batch_' + str(i))) for i in
                      range(1, 6)]
        trainx = np.concatenate([d['x'] for d in train_data], axis=0)
        trainy = np.concatenate([d['y'] for d in train_data], axis=0)
        return trainx, trainy
    elif subset == 'test':
        test_data = unpickle(os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')


def download_and_extract_cifar(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)


def load_omniglot():
    x_train = np.load('data/omniglot_x_train.npy')
    y_train = np.load('data/omniglot_y_train.npy')

    x_test = np.load('data/omniglot_x_test.npy')
    y_test = np.load('data/omniglot_y_test.npy')

    valid_classes = np.load('data/omniglot_valid_classes.npy')

    return (x_train, y_train), (x_test, y_test), valid_classes


def load_omniglot_vinyals():
    """
    pkl files from https://github.com/renmengye/few-shot-ssl-public#omniglot
    """
    with open('data/train_vinyals_aug90.pkl', 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        x_train = data[b'images']
        y_train = data[b'labels']
        y_str_train = data[b'label_str']

    with open('data/val_vinyals_aug90.pkl', 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        x_val = data[b'images']
        y_val = data[b'labels']
        y_val += np.max(y_train) + 1
        y_str_val = data[b'label_str']

    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))
    y_str_train.extend(y_str_val)

    with open('data/test_vinyals_aug90.pkl', 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        x_test = data[b'images']
        y_test = data[b'labels']
        y_test += np.max(y_train) + 1
        y_str_test = data[b'label_str']

    return (x_train, y_train, y_str_train), (x_test, y_test, y_str_test)


def process_omniglot_vinyals_split():
    """
    Dataset without rotations. Rotation will happen on-fly.
    """
    # this import is here so that it doesn't interact with matplotlib imports in other modules
    import skimage.io

    class_n = -1

    (_, _, y_str_train), (_, _, y_str_test) = load_omniglot_vinyals()
    y_str_train_new, y_str_test_new = [], []
    for l in y_str_train:
        y_str_train_new.append(l[:-2].decode("utf-8"))
        print(y_str_train_new[-1])
    y_str_train_new = list(set(y_str_train_new))
    print(y_str_train_new)
    print(len(y_str_train_new))

    for l in y_str_test:
        y_str_test_new.append(l[:-2].decode("utf-8"))
    y_str_test_new = list(set(y_str_test_new))
    print(y_str_test_new)
    print(len(y_str_test_new))

    images_dirpath_train = 'data/images_background/'
    images_dirpath_test = 'data/images_evaluation/'

    alphabets_dirs = glob.glob(images_dirpath_train + '/*/')
    alphabets_dirs_test = glob.glob(images_dirpath_test + '/*/')
    alphabets_dirs.extend(alphabets_dirs_test)

    x_train, y_train = [], []
    x_test, y_test = [], []

    # make a pkl with validation indices
    valid_alphabets = ['/Armenian/', '/Bengali/', '/Early_Aramaic/', '/Hebrew/', '/Mkhedruli_(Georgian)/']
    valid_classes = []

    for a_dir in alphabets_dirs:
        print('-----', a_dir)
        valid_set = True if any([a in a_dir for a in valid_alphabets]) else False
        chars_dirs = glob.glob(a_dir + '/*/')
        for c_dir in chars_dirs:
            class_n += 1
            img_paths = glob.glob(c_dir + '/*.png')
            print(c_dir, class_n)
            if valid_set:
                valid_classes.append(class_n)
            for i_path in img_paths:
                img = 255 - skimage.io.imread(i_path)
                img = scipy.misc.imresize(img, (28, 28))
                img = np.reshape(img, (784,))
                if c_dir.replace('data/images_background/', '')[:-1] in y_str_train_new \
                        or c_dir.replace('data/images_evaluation/', '')[:-1] in y_str_train_new:
                    print('train:', c_dir)
                    y_train.append(class_n)
                    x_train.append(img)
                else:
                    print('test', c_dir)
                    y_test.append(class_n)
                    x_test.append(img)

    valid_classes = np.asarray(valid_classes)
    np.save('data/omniglot_valid_classes', valid_classes)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    print(x_train.shape, y_train.shape)
    print(np.unique(y_train), len(np.unique(y_train)))

    np.save('data/omniglot_x_train', x_train)
    np.save('data/omniglot_y_train', y_train)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print(x_test.shape, y_test.shape)
    print(np.unique(y_test), len(np.unique(y_test)))

    np.save('data/omniglot_x_test', x_test)
    np.save('data/omniglot_y_test', y_test)


if __name__ == '__main__':
    process_omniglot_vinyals_split()
