import click
import h5py
import numpy as np

from audiomate.feeding import PartitioningFeatureIterator


@click.command()
@click.option('--h5-file', help='h5 file with features to read', type=click.Path(exists=True), required=True)
@click.option('--partition/--no-partition', default=False)
@click.option('--shuffle/--no-shuffle', default=False)
@click.option('--partition-size', default='12g')
def run(h5_file, partition, shuffle, partition_size):
    file = h5py.File(h5_file, 'r')
    counter = 0
    features_bytes_read = 0

    num_of_features = 0
    dsets = {}
    for dset in file:
        length = len(file[dset])
        dsets[dset] = length
        num_of_features += length

    if partition:
        for feature in PartitioningFeatureIterator(file, partition_size, shuffle=shuffle):
            features_bytes_read += feature[2].dtype.itemsize * len(feature[2])
            counter += 1
    else:
        if shuffle:
            indices = np.random.permutation(num_of_features)

            for idx in indices:
                for dset, length in dsets.items():
                    if idx >= length:
                        idx -= length
                        continue

                    feature = file[dset][idx]
                    features_bytes_read += feature.dtype.itemsize * len(feature)
                    counter += 1
                    break
        else:
            for dset in file:
                for feature in file[dset]:
                    features_bytes_read += feature.dtype.itemsize * len(feature)
                    counter += 1

    print('Total number of features: {0}'.format(num_of_features))
    print('Features read: {0}'.format(counter))
    print('Bytes of features read: {0}'.format(features_bytes_read))


if __name__ == '__main__':
    run()
