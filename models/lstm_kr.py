import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


class RNNClassifier(tf.keras.Model):
    def __init__(self, **kargs):
        super(RNNClassifier, self).__init__(name=kargs['model_name'])
        if kargs['train_mode'] == 'rand':
            self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                              output_dim=kargs['embedding_size'])
        else:
            self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                              output_dim=kargs['embedding_size'],
                                              weights=[kargs['embedding_matrix']])

        self.lstm_1_layer = layers.Bidirectional(layers.LSTM(kargs['lstm_dimension'], return_sequences=True))
        self.lstm_2_layer = layers.Bidirectional(layers.LSTM(kargs['lstm_dimension']))
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.fc1 = layers.Dense(units=kargs['dense_dimension'],
                                activation=tf.keras.activations.tanh)
        self.fc2 = layers.Dense(units=kargs['output_dimension'],
                                activation=tf.keras.activations.softmax)

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.lstm_1_layer(x)
        x = self.lstm_2_layer(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def plot_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string], '')
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()
