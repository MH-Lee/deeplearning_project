import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

class ACBLSTMClassifier(tf.keras.Model):
    def __init__(self, **kargs):
        super().__init__(name=kargs['model_name'])
        if kargs['train_mode'] == 'rand':
            self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                              output_dim=kargs['embedding_size'])
        else:
            self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                             output_dim=kargs['embedding_size'],
                                             weights=[kargs['embedding_matrix']],
                                             trainable=True)
        self.conv_list = [layers.Conv1D(filters=kargs['num_filters'],
                                        kernel_size=kernel_size,
                                        padding='same',
                                        activation=tf.keras.activations.relu,
                                        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
                                        for kernel_size in [3, 4, 5]]
        self.pooling = layers.MaxPooling1D()
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.lstm_layer = layers.Bidirectional(layers.LSTM(kargs['lstm_dimension'], return_sequences=True))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=kargs['hidden_dimension'],
                           activation=tf.keras.activations.relu,
                           kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.)
                           )
        self.fc2 = layers.Dense(units=kargs['output_dimension'],
                           activation=tf.keras.activations.softmax,
                           kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.)
                           )
    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = layers.concatenate([self.pooling(conv(x)) for conv in self.conv_list], axis=2)
        x = self.lstm_layer(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def plot_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string], '')
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()
