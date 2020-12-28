from jax import random
from jax.experimental import stax
from jax.experimental import optimizers
from jax import jit, value_and_grad
import jax.numpy as jnp

key = random.PRNGKey(1)

class Generator(object):
    def __init__(self, z_dim:int=128, hidden_dim:int=64, im_channel:int=1, batch_size:int=1, random_key=key):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.im_channel = im_channel
        self.batch_size = batch_size
        self.random_key = random_key
        self.model = None
        self.params = None
        self._construct_model()

    def make_gen_block(self, output_channel, final_layer=False, kernel_size=(3,3), strides=(2,2)):
        if not final_layer:
            return (
                stax.ConvTranspose(output_channel, kernel_size, strides),
                stax.BatchNorm(),
                stax.Relu
            )
        else:
            return (
                stax.ConvTranspose(output_channel, kernel_size, strides),
                stax.Tanh
            )
    def _construct_model(self):
        init_fun, self.model = stax.serial(
            *self.make_gen_block(self.hidden_dim*4, kernel_size=(5,5)),
            #output_Size= (5,5,256)
            *self.make_gen_block(self.hidden_dim*2),
            #output_size=(11,11,128
            *self.make_gen_block(self.hidden_dim, kernel_size=(5,5)),
            #output_size=(25,25, 64)
            *self.make_gen_block(self.im_channel, kernel_size=(4,4), strides=(1,1), final_layer=True),
            # output_size=(28,28, 1)
        )
        output_shape, self.params = init_fun(self.random_key, (self.batch_size,1,1,self.z_dim))
        print('generator output shape {}'.format(output_shape))

    def construct_optimizers(self, step_size = 1e-3):
        opt_init, self.opt_update, self.get_params = optimizers.adam(step_size)
        return opt_init(self.params)

    def update(self, params, x, y, opt_state, disc):
        value, grads = value_and_grad(self.loss_fn)(params, x, y, disc)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value

    def loss_fn(self, params, x, y, disc):
        fake_images = self.model(params, x)
        logits = disc.model(disc.params, fake_images)
        logits = jnp.expand_dims(jnp.squeeze(logits), axis=-1)
        return jnp.mean(-(logits * y) - ((1 - y) * (1 - logits)))

if __name__ == '__main__':
    gen = Generator()
    noise = random.uniform(key, shape=(1,128))
    noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=1)
    pred = gen.model(gen.params, noise)
    print(pred.shape)