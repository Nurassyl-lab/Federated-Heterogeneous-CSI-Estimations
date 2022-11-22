import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from skimage import io
import keras
from keras import layers
from numpy.random import permutation
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import random
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow
from numpy.random import seed
import random as rd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from numpy import swapaxes
from numpy import array
import pandas as pd
from os import listdir
from os.path import isfile, join
from keras.layers.advanced_activations import PReLU
iters = 20
num_clusters = 9
num_users = 45

h_dim = 392
e_dim = 196

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
        # self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return[
                # self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
              ]

    def train_step(self, data):
        with tensorflow.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction)))
            reconstruction_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(keras.losses.mse(data, reconstruction)))
            kl_loss = -0.5 * (1 + z_log_var - tensorflow.square(z_mean) - tensorflow.exp(z_log_var))
            kl_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            # "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
                }

def pre_process(X):
    # X = X/255.0
    X = X.reshape((len(X), 784)) 
    X = X[permutation(len(X))]
    return X

def show_data(X, n=10, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)
    return

def show_one_data(X, n=1, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    plt.imshow(X.reshape((height,width)))
    plt.gray()
    plt.suptitle(title, fontsize = 20)
    return

def save_one_image(X, height = 28, width = 28, path = ""):
    plt.imsave(path, X.reshape((height,width)))
    return



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): 
        img = io.imread(os.path.join(folder,filename), as_gray=True)
        if img is not None:
            images.append(img)
    return images

class CLASS:
    def __init__(self, number):
        self.number = number
        # self.train_data = []
        # self.test_data = []
        self.model = None
        self.hist = []
        return
    
    
    def load_data(self, path):
        'load train data'
        arr = []
        datanames = [f for f in listdir(path+'train_data/') if isfile(join(path+'train_data/', f))]
        for data in datanames:
            tmp = np.load(path+'train_data/'+data)
            nump_data = np.stack((tmp.real, tmp.imag))
            arr.append(nump_data.reshape(2, 64, 100, 1))
        self.train_data = np.array(arr)
        
        arr = []
        datanames = [f for f in listdir(path+'val_data/') if isfile(join(path+'val_data/', f))]
        for data in datanames:
            tmp = np.load(path+'val_data/'+data)
            nump_data = np.stack((tmp.real, tmp.imag))
            arr.append(nump_data.reshape(2, 64, 100, 1))
        self.val_data = np.array(arr)
    
    
    def define_dense_model(self):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='softmax', name = 'out'))
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
        return 
    
    def define_cnn(self, act_function = 'tanh'):
        self.model = Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=(2, 64, 100, 1)))
        self.model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))#32 50
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4'))#16 25
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5'))
        self.model.add(layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6'))#8 5
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7'))

        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
        self.model.add(layers.UpSampling3D((1, 2, 5), name = 'h11'))#16, 25
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#32, 50
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h15'))
        self.model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse')
    
    def define_vae(self):
        latent_dim = 2#maybe

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
        self.vae_encoder_imag = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_imag")

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
        self.vae_decoder_imag = keras.Model(latent_inputs, decoder_outputs, name="decoder_imag")
        self.vae_imag = VAE(self.vae_encoder_imag, self.vae_decoder_imag)
        self.vae_imag.compile(optimizer=keras.optimizers.Adam())
        
        
        self.vae_encoder_real = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_real")
        self.vae_decoder_real = keras.Model(latent_inputs, decoder_outputs, name="decoder_real")
        self.vae_real = VAE(self.vae_encoder_real, self.vae_decoder_real)
        self.vae_real.compile(optimizer=keras.optimizers.Adam())
        

class SERVER:
    def __init__(self):
        self.classes = []
    
    
    def load_data(self, path):
        'load train data'
        self.train_data = []
        datanames = [f for f in listdir(path+'train_data/') if isfile(join(path+'train_data/', f))]
        for data in datanames:
            self.train_data.append(np.load(path+'train_data/'+data))
            
            
        'load validation data'
        self.val_data = []
        datanames = [f for f in listdir(path+'val_data/') if isfile(join(path+'val_data/', f))]
        for data in datanames:
            self.val_data.append(np.load(path+'val_data/'+data))
    
    
    def set_classes(self, list_class):
        self.classes += list_class