from jax.experimental import optimizers
import jax.numpy as jnp
import time
from discriminator import Discriminator
from generator import Generator
from dataset import MnistDataset
from jax import random
import matplotlib.pyplot as plt

#Create a gan experiment
key = random.PRNGKey(1)
batch_size = 16
epoch_size = 100
hidden_dim = 128
display_step = 500

dataset = MnistDataset(batch_size=batch_size)

gen = Generator(batch_size=batch_size, random_key=key)
gen_opt_init = gen.construct_optimizers(1e-2)

disc = Discriminator(batch_size=batch_size, random_key=key)
disc_opt_init = disc.construct_optimizers(1e-2)
mean_discriminator_loss = 0
mean_generator_loss = 0
cur_step = 0


start_time = time.time()
for epoch in range(epoch_size):
    for real_images, _ in dataset.get_train_batches():
        noise = random.uniform(key, shape=(batch_size, 1, 1, hidden_dim))
        fake_images = gen.model(gen.params, noise)
        disc.params, disc_opt_init, disc_fake_loss = disc.update(
            disc.params, fake_images, jnp.zeros(shape=(batch_size, 1)), disc_opt_init
        )
        disc.params, disc_opt_init, disc_real_loss = disc.update(
            disc.params, real_images, jnp.ones(shape=(batch_size, 1)), disc_opt_init
        )

        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        mean_discriminator_loss += disc_loss / display_step

        noise = random.uniform(key, shape=(batch_size, 1, 1, hidden_dim))
        gen.params, gen_opt_init, gen_loss = gen.update(
            gen.params, noise, jnp.ones(shape=(batch_size, 1)), gen_opt_init, disc
        )

        mean_generator_loss += gen_loss / display_step

        if cur_step != 0 and cur_step % display_step == 0:
            epoch_time = time.time() - start_time
            print("Step {} | T: {:0.2f} | Train Discriminator Loss: {:0.3f} | Train Generator loss: {:0.3f}".format(
                cur_step, epoch_time, mean_discriminator_loss, mean_generator_loss))
            plt.imshow(fake_images[0].squeeze())
            plt.imshow(real_images[0].squeeze())
            start_time = time.time()
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step+=1