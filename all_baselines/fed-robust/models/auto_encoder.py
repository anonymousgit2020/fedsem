from tensorflow.python import pywrap_tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session


class AutoEncoder: 
    def __init__(self, encoder_file, dataset, model):
        self._model = model
        self.encoder_file = encoder_file % (dataset, model)

    def create_rnn_model(self, dataset):
        input_dim = 256
        output_dim = 5000
        num_classes = 53
        if dataset == "sent140":
            input_dim = 100
            output_dim = 2500
            num_classes = 128
        input_shape = (input_dim, num_classes, 1)
        batch_size = 16
        kernel_size = 3

        # encoder/decoder number of CNN layers and filters per layer
        layer_filters = [64, 128]
        inputs = Input(shape=input_shape, name='encoder_input')

        x = inputs
        x = Reshape((input_dim, num_classes))(x)
        x = Dense(64)(x)
        x = Reshape((input_dim, 64, 1))(x)

        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        x = Flatten()(x)
        latent = Dense(output_dim, name='latent_vector')(x)
        return  Model(inputs,
                    latent,
                    name='encoder')        


    def create_cnn_model(self):
        input_dim = 2048
        output_dim = 5000
        input_shape = (input_dim, 62, 1)
        batch_size = 16
        kernel_size = 3

        # encoder/decoder number of CNN layers and filters per layer
        layer_filters = [64, 128]

        # build the autoencoder model
        # first build the encoder model
        inputs = Input(shape=input_shape, name='encoder_input')

        x = inputs
        x = Reshape((input_dim, 62))(x)
        x = Dense(64)(x)
        x = Reshape((input_dim, 64, 1))(x)

        # stack of Conv2D(64)-Conv2D(128)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        # so we don't do hand computation
        # the input to the decoder's first
        # Conv2DTranspose will have this shape
        # shape is (7, 7, 64) which is processed by
        # the decoder back to (28, 28, 1)
        # shape = K.int_shape(x)
        # print(shape)

        # generate latent vector
        x = Flatten()(x)
        latent = Dense(output_dim, name='latent_vector')(x)
        return  Model(inputs,
                    latent,
                    name='encoder')

    def run(self, all_model_data, dataset):
        if self._model == "cnn":
          new_model = self.create_cnn_model()
        else: 
          new_model = self.create_rnn_model(dataset)
        new_model.load_weights(self.encoder_file)
        dataset = new_model.predict(all_model_data)
        clear_session()
        return dataset