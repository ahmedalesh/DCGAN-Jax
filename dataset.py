import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp

class MnistDataset(object):
    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.data_dir = '/tmp/tfds'

    def get_train_batches(self):
        # as_supervised=True gives us the (image, label) as a tuple instead of a dict
        ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=self.data_dir)
        # You can build up an arbitrary tf.data input pipeline
        ds = ds.map(self.normalize_img)
        ds = ds.batch(self.batch_size).prefetch(1)
        # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
        return tfds.as_numpy(ds)

    def normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return (tf.cast(image, tf.float32) / 127.5 - 1.0), label