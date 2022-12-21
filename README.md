Not finished yet!!!
DATASET will be added soon

The repo may not be well organized. This is my first big project the file management is a bit chaotic ;(

# Content:
  1) Introduction
  2) CSI dataset
  3) Code description
  4) My observations

# Introduction
---
Implementation of **Federated Learning** (a.k.a collaborative learning) for **CSI data** transmission between UE (user equipment) and BS (base station).
**Federated learning** is a subpart of AI and Machine learning where the **Central model** (also refered as server model) is trained/improved using local (decentralized models) `user` models. Several implementations of Fed. learning is Amazon Alexa, Apple Siri, Google Keyboard and etc.

# CSI dataset
---
**CSI** or **Channel State Information**, in wireless communications is the known channel properties of a communication link between transmitter and receiver. This information describes how a signal propagates from the transmitter to the receiver and represents the combined effect of, for example, scattering, fading, and power decay with distance. The CSI makes it possible to adapt transmissions to current channel conditions, which is crucial for achieving reliable communication with high data rates in multiantenna systems.

I have sampled **unbalanced_dataset** (i will add it soon) from https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

Dataset description you can find on the website. Note that all 64 antennas are not in phase.
**CSI dataset** consist of multiple csi data samples for training and validation. Decentralized models have training dataset of size 17000 and validation dataset of size 1500 csi datasamples.

Visualization of csi data sample.
Plot of 1 data sample 64x100 matrix on the left and it's 2D DFT on the right. As you can see a 2D DFT is very sparse.
![data1](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/csi_data_and_fft.png)

Plot of signal's magnitude in dB over 100 subcarriers of 1 antennas.
![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/mag_subcarriers_data_over_1_antenna.png)

Plot of signal's phase in degrees over 100 subcarriers of 1 antennas.
![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/phase_subcarriers_data_over_1_antenna.png)

# Code description
---
for now only variational autoencoder simultion is available.
Use VAE_simulation.py to train decentralized models. 

In this project autoencoder architecture is implemented in order to compress the CSI data and transmit it from UE to BS.
The goal of the project is to compress the data as much as possible and reach smallest reconstruction loss on the server.
However, there is one detail that must be mentioned, it is data heterogeneity. Remember that each local model is trained on its own dataset. This is done because in real life there won't be perfectly balanced UE dataset. (So, we also explore how biasness affects model performances)
Data heterogeneity describes the biasness of the data, and heterogeneity level or (epsilon) is variable that describes the data heterogeneity in %.

![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/heterogeeity_level.png)

In our case we split dataset into 3 classes using L2-norm.

  CLASS 1: csi data with l2-norm value in interval [40, 45]
  
  CLASS 2: csi data with l2-norm value in interval [50, 55]
  
  CLASS 3: csi data with l2-norm value in interval [60, 65]

# Variational Autoencoder:
In VAE_CSI_MODEL I have a function ` def define_VAE_CSI_MODEL()` that is used in to create vae models.
I use simple CNN structure in order to do the compression and decompression.
Note! I use 2 different vae models for complex csi data, one is for real part, another is for imaginary. But they follow the same structure.

I use `tanh` activation function since output should have positive and negative values.

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

Next part of the code.

    class Sampling(layers.Layer):
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
                
In line 71 we have a custom layer that is called Sampling. This is for generation new samples, since VAE is a generative model.
When the layer is called:
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tensorflow.shape(z_mean)[0]
        dim = tensorflow.shape(z_mean)[1]
        epsilon = tensorflow.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tensorflow.exp(0.5 * z_log_var) * epsilon 
        
as you can see inputs are mean and log variance of your model input data (in this case input is CSI data). 

This function takes two input tensors, `z_mean` and `z_log_var`, and uses them to sample from a normal distribution. The shape of the normal distribution is determined by the batch size and number of dimensions of the input tensors. The function returns a tensor of samples from the normal distribution.
As from <br return z_mean + tensorflow.exp(0.5 * z_log_var) * epsilon />, this equation is an expression for sampling from a normal distribution with a mean and a standard deviation that are determined by two other variables, `mean` and `log_var`. The `epsilon` variable is a random value that is drawn from a standard normal distribution (mean = 0, standard deviation = 1). This value is multiplied by the standard deviation of the normal distribution, which is computed as the exponential of 0.5 times the logarithm of the variance. This ensures that the standard deviation of the normal distribution is correctly reflected in the final sample.

The factor 0.5 is included in the equation to compute the standard deviation of a normal distribution from the logarithm of the variance. The standard deviation is half the size of the variance, so the factor 0.5 is used to correctly scale the standard deviation. The standard deviation is then used to scale a random normal tensor, which is used to sample from the normal distribution.

In the same file you may find `class VAE`. 
The VAE class has three `trackers` for mean total loss, mean reconstruction loss, and mean KL loss, which are used to keep track of the average loss during training. These trackers are instances of the `Mean` class from the `keras.metrics` module.

VAE class has a `train_step` method that is used to perform a single training step on a batch of data. The method computes the reconstruction loss, KL loss, and total loss for the batch, and uses these losses to compute gradients for the model's trainable weights. The gradients are then used to update the model's weights using the optimizer's `apply_gradients` method. The method also updates the mean loss trackers with the current batch's loss values.

The VAE class also has a `metrics` property that returns a list of the mean loss trackers. This can be used to track the loss values during training and evaluate the performance of the model.

Everything in this class is pretty much straigt forward. Except `train_step` function. Let me clarify it.
`kl_loss = -0.5 * (1 + z_log_var - tensorflow.square(z_mean) - tensorflow.exp(z_log_var))`
This line of code computes the KL loss, which is a measure of the difference between the distribution of the latent representation and a prior distribution that is assumed to be known. The KL loss is used to encourage the latent representation to have a certain distribution, which can improve the quality of the reconstructions produced by the decoder. (More information is available on the internet)

The KL loss is computed as follows:

The term `1 + z_log_var` represents the logarithm of the variance of the latent distribution.

The term `tensorflow.square(z_mean)` represents the square of the mean of the latent distribution.

The term `tensorflow.exp(z_log_var)` represents the variance of the latent distribution.

The terms `1 + z_log_var`, `tensorflow.square(z_mean)`, and `tensorflow.exp(z_log_var)` are subtracted from each other and multiplied by -0.5 to compute the KL loss.

The factor -0.5 is included in the equation to compute the KL loss as a common convention in machine learning, and to scale the KL loss to a more manageable value. The variance is represented by the term `tensorflow.exp(z_log_var)` because the logarithm of the variance is equal to the logarithm of the square of the standard deviation, and taking the exponential of the logarithm of a value is equivalent to taking the square root of the value. Therefore, the term `tensorflow.exp(z_log_var)` represents the square root of the variance, which is equal to the standard deviation. The standard deviation is a measure of the spread of the latent distribution.

In line `kl_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))` of code computes the mean KL loss over the batch of data by applying the `reduce_mean` function to the KL loss tensor. The `reduce_mean` function computes the mean of a tensor along a particular axis. In this case, the `reduce_sum` function is used to sum the KL loss values over the batch axis (axis=1), and the result is passed as an argument to the `reduce_mean` function. This results in a scalar value that represents the mean KL loss over the batch.

### Setup 1: UE selection
As for now only [setup 1] is available.
In this case pre-trained encoders are delivered to UE. When UE receives signal that it needs to send to BS the following algorithm is being executed under setup 1. 
UE compresses input using all 3 encoders that it has received. Now UE has 3 different encoded features of


main2 python code smaller copy of the main file which is not yet included in this repository.
You can use main2 to get familiar with our setup and etc.

Right now i'm still running simulations, I will add more infomation regarding this project later.

# My observations
---
Graveyard of failed ideas. 
