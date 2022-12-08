import keras
from keras import layers
import tensorflow

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tensorflow.shape(z_mean)[0]
        dim = tensorflow.shape(z_mean)[1]
        epsilon = tensorflow.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tensorflow.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return[
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
              ]

    def train_step(self, data):
        with tensorflow.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(keras.losses.mse(data, reconstruction)))
            # reconstruction_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(mean_squared_error(data, reconstruction)))
            kl_loss = -0.5 * (1 + z_log_var - tensorflow.square(z_mean) - tensorflow.exp(z_log_var))
            kl_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
                }

def define_VAE_CSI_MODEL():
    latent_dim = 8*5

    encoder_inputs = keras.Input(shape=(64, 100, 1))
    x = layers.Conv2D(16, (3,3), activation="tanh", padding='same',name = 'h1')(encoder_inputs)
    x = layers.MaxPooling2D((2,2), name = 'h2')(x)#32 50
    
    x = layers.Conv2D(32, (3,3), activation="tanh", padding='same',name = 'h3')(x)
    x = layers.MaxPooling2D((2,2), name = 'h4')(x)#16 25
    
    x = layers.Conv2D(64, (3,3), activation="tanh", padding='same',name = 'h5')(x)
    x = layers.MaxPooling2D((2,5), name = 'h6')(x)#8 5
    
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    vae_encoder_imag = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_imag")

    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = keras.layers.Dense(40, name = 'h7')(latent_inputs)
    x = keras.layers.Reshape((8, 5, 1), name = 'h8')(x)
    
    x = layers.Conv2D(64, (3,3), activation="tanh", padding='same',name = 'h9')(x)
    x = layers.UpSampling2D((2,5), name = 'h10')(x)#16 25
    
    x = layers.Conv2D(64, (3,3), activation="tanh", padding='same',name = 'h11')(x)
    x = layers.UpSampling2D((2,2), name = 'h12')(x)#32 50

    x = layers.Conv2D(64, (3,3), activation="tanh", padding='same',name = 'h13')(x)
    x = layers.UpSampling2D((2,2), name = 'h14')(x)#64 100

    decoder_outputs = layers.Conv2D(1, (3,3), activation="tanh", padding='same',name = 'h15')(x)
    vae_decoder_imag = keras.Model(latent_inputs, decoder_outputs, name="decoder_imag")
    vae_imag = VAE(vae_encoder_imag, vae_decoder_imag)
    vae_imag.compile(optimizer=keras.optimizers.Adam())
    
    vae_encoder_real = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_real")
    vae_decoder_real = keras.Model(latent_inputs, decoder_outputs, name="decoder_real")
    vae_real = VAE(vae_encoder_real, vae_decoder_real)
    vae_real.compile(optimizer=keras.optimizers.Adam())
    
    return vae_encoder_imag, vae_decoder_imag, vae_encoder_real, vae_decoder_real, vae_imag, vae_real