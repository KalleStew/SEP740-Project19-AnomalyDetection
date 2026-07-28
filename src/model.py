import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

class Autoencoder(tf.keras.Model):

    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(latent_dim, activation="relu")
        ])

        self.decoder = Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])


    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x