"""
Mostly VERSA shapenet code from https://github.com/Gordonjo/versa.
"""

import os

import numpy as np
from PIL import Image


def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def load_pngs(input_dir, data, size):
    items = get_subdirs(input_dir)
    for item_index, item in enumerate(items):
        print(item)
        item_images = []
        instances = []
        # There are 36 generated orientations for each item
        for i in range(0, 36):
            instances.append("{0:02d}.png".format(i))

        for instance_index, instance in enumerate(instances):
            im = Image.open(os.path.join(item, instance))
            if size:
                im = im.resize((size, size), resample=Image.LANCZOS)
            image = np.array(im.getdata()).astype('float32').reshape(size, size) / 255.  # grayscale image
            item_images.append((image, item_index, instance_index))

        data.append(item_images)

    return data


def process_shapenet():
    data_dir = 'data/shapenet_12classes'
    categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117', '03636649',
                  '03691459', '04256520', '04379243', '04530566']
    for cat in categories:
        data = []
        print('Starting', cat)
        data = load_pngs(os.path.join(data_dir, cat), data, size=32)
        np.save('{0:s}.npy'.format(cat), np.array(data))
    print('Finished', cat)


def load_shapenet(set='train', train_fraction=0.7, val_fraction=0.1, num_instances_per_item=36, image_height=32,
                  image_width=32, image_channels=1, path='data/shapenet_12classes'):
    """
    airplane 02691156 4045
    bench 02828884 1813
    cabinet 02933112 1571
    car 02958343 3533
    phone 02992529 831
    chair 03001627 6778
    display 03211117 1093
    lamp 03636649 2318
    speaker 03691459 1597
    sofa 04256520 3173
    table 04379243 8436
    boat 04530566 1939
    """
    categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117',
                  '03636649', '03691459', '04256520', '04379243', '04530566']
    data = None
    for category in categories:
        file = os.path.join(path, '{0:s}.npy'.format(category))
        if category == categories[0]:
            data = np.load(file)
        else:
            data = np.concatenate((data, np.load(file)), axis=0)

    total_items = data.shape[0]
    train_size = int(train_fraction * total_items)
    val_size = int(val_fraction * total_items)
    print("Categories", categories)
    print("Training Set Size = {0:d}".format(train_size))
    print("Validation Set Size = {0:d}".format(val_size))
    print("Test Set Size = {0:d}".format(total_items - train_size - val_size))
    rng = np.random.RandomState(42)
    rng.shuffle(data)
    if set == 'train':
        x, y, angles = shapenet_extract_data(data[:train_size],
                                             num_instances_per_item,
                                             image_height,
                                             image_width, image_channels)

    elif set == 'valid':
        x, y, angles = shapenet_extract_data(
            data[train_size:train_size + val_size], num_instances_per_item, image_height,
            image_width, image_channels)
    elif set == 'test':
        x, y, angles = shapenet_extract_data(data[train_size + val_size:],
                                             num_instances_per_item,
                                             image_height,
                                             image_width, image_channels)
    else:
        raise ValueError('wrong set')

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
