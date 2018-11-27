import os
import sys

import numpy as np
from PIL import Image

import utils

"""
Shapenet code is taken from https://github.com/Gordonjo/versa
"""


def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def load_pngs_and_save_as_npy(input_dir, save_file, model_type, size, convert_to_grayscale):
    data = []
    items = get_subdirs(input_dir)
    for item_index, item in enumerate(items):
        print(item)
        item_images = []
        # we have generated 36 orientations for each item
        # but they are numbered 1-9 (with a single digit),
        # and 10-35 (with 2 digits) and we want them in order
        instances = []
        # There are 36 generated orientations for each item
        if model_type == 'plane':
            # Planes  are numbered 1-9 (with a single digit), and 10-35 (with 2 digits) and we want them in order.
            for i in range(0, 10):
                instances.append("model_normalized.obj-{0:1d}.png".format(i))
            for i in range(10, 36):
                instances.append("model_normalized.obj-{0:2d}.png".format(i))
        elif model_type == 'chair':
            # Chairs are numbered consistently with 2 digits.
            for i in range(0, 36):
                instances.append("{0:02d}.png".format(i))
        else:
            sys.exit("Unsupported model type (%s)." % model_type)

        for instance_index, instance in enumerate(instances):
            im = Image.open(os.path.join(item, instance))
            if convert_to_grayscale:
                im = im.convert("L")
            if size:
                im = im.resize((size, size), resample=Image.LANCZOS)
            if convert_to_grayscale:
                image = np.array(im.getdata()).astype('float32').reshape(size, size) / 255.  # grayscale image
            else:
                image = np.array(im.getdata()).astype('float32').reshape(size, size, 3) / 255.  # colour image
            item_images.append((image, item_index, instance_index))

        data.append(item_images)

    np.save(save_file, np.array(data))


def process_shapenet():
    data_dir = 'data/shapenet'
    planes_dir = '02691156'
    chairs_dir = '03001627'

    print('Starting ShapeNet planes.')
    load_pngs_and_save_as_npy(os.path.join(data_dir, planes_dir), data_dir + '/shapenet_planes.npy',
                              model_type='plane', size=32, convert_to_grayscale=True)
    print('Finished ShapeNet planes')

    print('Starting ShapeNet chairs.')
    load_pngs_and_save_as_npy(os.path.join(data_dir, chairs_dir), data_dir + '/shapenet_chairs.npy',
                              model_type='chair', size=32, convert_to_grayscale=True)
    print('Finished ShapeNet chairs')


def load_shapenet(set='train', train_fraction=0.7, val_fraction=0.1, num_instances_per_item=36, image_height=32,
                  image_width=32, image_channels=1):
    data_paths = ['data/shapenet/shapenet_planes.npy', 'data/shapenet/shapenet_chairs.npy']
    x, y, angles = [], [], []
    for p in data_paths:
        data = np.load(p)

        total_items = data.shape[0]
        train_size = (int)(train_fraction * total_items)
        val_size = (int)(val_fraction * total_items)
        print("Training Set Size = {0:d}".format(train_size))
        print("Validation Set Size = {0:d}".format(val_size))
        print("Test Set Size = {0:d}".format(total_items - train_size - val_size))
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        if set == 'train':
            train_images, train_item_indices, train_item_angles = shapenet_extract_data(data[:train_size],
                                                                                        num_instances_per_item,
                                                                                        image_height,
                                                                                        image_width, image_channels)
        elif set == 'valid':
            train_images, train_item_indices, train_item_angles = shapenet_extract_data(
                data[train_size:train_size + val_size], num_instances_per_item, image_height,
                image_width, image_channels)
        elif set == 'test':
            train_images, train_item_indices, train_item_angles = shapenet_extract_data(data[train_size + val_size:],
                                                                                        num_instances_per_item,
                                                                                        image_height,
                                                                                        image_width, image_channels)
        else:
            raise ValueError('wrong set')

        x.append(train_images)
        if y:
            y.append(train_item_indices + np.max(y[-1] + 1))
        else:
            y.append(train_item_indices)

        angles.append(train_item_angles)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    angles = np.concatenate(angles, axis=0)
    x *= 255.
    return x, y, angles


def shapenet_extract_data(data, num_instances_per_item, image_height, image_width, image_channels):
    """
    Unpack ShapeNet data.
    """
    images, item_indices, item_angles = [], [], []
    for item_index, item in enumerate(data):
        for m, instance in enumerate(item):
            images.append(instance[0])
            item_indices.append(item_index)
            item_angles.append(shapenet_convert_index_to_angle(instance[2], num_instances_per_item))
    images = np.reshape(np.array(images), (len(images), image_height, image_width, image_channels))
    indices, angles = np.array(item_indices), np.array(item_angles)
    return images, indices, angles


def shapenet_convert_index_to_angle(index, num_instances_per_item):
    """
    Convert the index of an image to a representation of the angle
    :param index: index to be converted
    :param num_instances_per_item: number of images for each item
    :return: a biterion representation of the angle
    """
    degrees_per_increment = 360. / num_instances_per_item
    angle = index * degrees_per_increment
    angle_radians = np.deg2rad(angle)
    return np.sin(angle_radians), np.cos(angle_radians)


if __name__ == "__main__":
    process_shapenet()
