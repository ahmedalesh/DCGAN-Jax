from jax.experimental import stax
from jax import random
from jax.experimental import optimizers
from jax import jit, value_and_grad
import jax.numpy as jnp
from functools import reduce

key = random.PRNGKey(1)

class Discriminator(object):
    def __init__(self, batch_size:int=1, input_shape:tuple=(28,28), im_channel=1, hidden_size=64, random_key=key):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.im_channel = im_channel
        self.hidden_size = hidden_size
        self.random_key = random_key
        self._construct_model()

    def make_disc_block(self, output_channel, kernel_size, strides, use_batchnorm=False, final_layer:bool=True):
        if final_layer:
            layers = [
                stax.Conv(output_channel, kernel_size, strides),
                stax.Sigmoid
            ]
        else:
            layers = [
                stax.Conv(output_channel, kernel_size, strides),
            ]
            if use_batchnorm:
                layers.append(stax.BatchNorm())
            layers.append(stax.LeakyRelu)
        return tuple(layers)

    def _construct_model(self):
        init_fun, self.model = stax.serial(
            *self.make_disc_block(self.hidden_size, kernel_size=(5,5), strides=(2,2)),
            # output_size = 1,12, 12, 64
            *self.make_disc_block(self.hidden_size*2, kernel_size=(5, 5), strides=(2, 2)),
            # output_size = 1,4, 4, 128
            *self.make_disc_block(self.hidden_size*4, kernel_size=(3, 3), strides=(2, 2)),
            #output_size = 1,1,1,128
            *self.make_disc_block(self.im_channel, kernel_size=(1, 1), strides=(1, 1)),
        )
        output_shape, self.params = init_fun(self.random_key, (self.batch_size, self.input_shape[0], self.input_shape[1], self.im_channel))
        print('discriminator output shape {}'.format(output_shape))

    def construct_optimizers(self, step_size = 1e-3):
        opt_init, self.opt_update, self.get_params = optimizers.adam(step_size)
        return opt_init(self.params)

    def update(self, params, x, y, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(self.loss_fn)(params, x, y)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value

    def loss_fn(self, params, x, y):
        logits = self.model(params, x)
        logits = jnp.expand_dims(jnp.squeeze(logits), axis=-1)
        return jnp.mean(-(logits * y) - ((1 - y) * (1 - logits)))

if __name__ == '__main__':
    import jax.numpy as jnp
    image = random.uniform(key, shape=(16, 28, 28, 1))
    disc = Discriminator(batch_size=16)
    pred = disc.model(disc.params, image)
    batch_size, *input_shape = pred.shape
    input_shape = reduce(lambda a,b: a*b, input_shape)
    pred = jnp.reshape(pred, (16, input_shape))
    print(pred.shape)
