import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
sys.path.append('..\\..')
from models.ac_blstm_kr import ACBLSTMClassifier
from models.word_embedding import get_embedding_matrix

DATA_IN_PATH = './assets/data/npy_data/2020-05-31/'
DATA_OUT_PATH = './model_save/acblstm_model/'
DATA_OUT_PATH_SMALL = './model_save/acblstm_model_small/'
# Data save label
TRAIN_INPUT_DATA = 'train_input.npy'
TEST_INPUT_DATA = 'test_input.npy'
DATA_CONFIGS = 'data_configs.json'
SEQ_CONFIGS = 'seq_configs_bt.json'

# Train label save file name
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_LABEL_SMALL = 'train_label_small.npy'
TEST_LABEL_DATA = 'test_label.npy'
TEST_LABEL_DATA_SMALL = 'test_label_small.npy'


class ACBLSTMTrain:
    def __init__(self,
                 embedding_matrix=None,
                 vocab_size=None,
                 lr_schedule=None,
                 pre_trained_mode=None,
                 embedding_size=300,
                 num_filters=128,
                 lstm_dimension=256,
                 hidden_dimension=128,
                 word_index = None,
                 dropout_rate=0.5,
                 output_dim=43,
                 batch_size = 64,
                 num_epochs = 1000,
                 valid_split = 0.2,
                 optimizer = 'adam',
                 train_mode='rand'):
        if (train_mode == 'rand') and (embedding_matrix is not None):
            raise Exception('rand는 embedding matrix를 가질 수 없습니다.')
        if output_dim > 43:
            print('small_class')
            print('output_dim : ', output_dim)
            self.DATA_OUT_PATH = DATA_OUT_PATH_SMALL
            print(self.DATA_OUT_PATH)
        else:
            print('big_class')
            print('output_dim : ', output_dim)
            self.DATA_OUT_PATH = DATA_OUT_PATH
            print(self.DATA_OUT_PATH)

        self.kargs = {'vocab_size'      :  vocab_size,
                      'embedding_size'  :  embedding_size,
                      'num_filters'     :  num_filters,
                      'dropout_rate'    :  dropout_rate,
                      'lstm_dimension'  :  lstm_dimension,
                      'hidden_dimension':  hidden_dimension,
                      'train_mode'      :  train_mode,
                      'output_dimension':  output_dim}

        if (train_mode == 'rand') and (embedding_matrix is not None):
            raise Exception('rand는 embedding matrix를 가질 수 없습니다.')
        if pre_trained_mode == 'w2v':
            self.kargs['model_name'] = 'AC-BLSTM_{}_{}_{}'.format(train_mode, optimizer, pre_trained_mode)
            word_vectorizer = embedding_matrix
            emb_matrix = get_embedding_matrix(word_vectorizer, word_index)
            self.kargs['embedding_matrix'] = emb_matrix
        elif pre_trained_mode == 'pt_w2v':
            self.kargs['model_name'] = 'AC-BLSTM_{}_{}_{}'.format(train_mode, optimizer, pre_trained_mode)
            word_vectorizer = embedding_matrix
            emb_matrix = get_embedding_matrix(word_vectorizer, word_index)
            self.kargs['embedding_matrix'] = emb_matrix
        else:
            self.kargs['model_name'] = 'AC-BLSTM_{}_{}'.format(train_mode, optimizer, pre_trained_mode)

        self.kargs['train_mode'] = train_mode
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.valid_split = valid_split
        print('train_mode : {}, optimizer : {}, pre-trained_mode : {}, model_name : {}'.format(
            train_mode,
            self.optimizer,
            pre_trained_mode,
            self.kargs['model_name']
        ))

    def train(self, train_X, train_Y, lr_schedule=None):
        if self.optimizer == 'radam':
            opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
        else:
            opt = self.optimizer

        print(self.kargs)
        f1_score = tfa.metrics.F1Score(
            num_classes= self.kargs['output_dimension'],
            average='macro'
        )
        model = ACBLSTMClassifier(**self.kargs)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', f1_score])

        checkpoint_path = self.DATA_OUT_PATH + self.kargs['model_name'] + '/weights.h5'
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create path if exists
        if os.path.exists(checkpoint_dir):
            print("{} -- Folder already exists \n".format(checkpoint_dir))
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("{} -- Folder create complete \n".format(checkpoint_dir))

        earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)
        cp_callback = ModelCheckpoint(checkpoint_path,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True)

        if self.lr_schedule is not None:
            print('lr_schedule_set')
            callback_list = [earlystop_callback, cp_callback, lr_schedule]
        else:
            print('callback set')
            callback_list = [earlystop_callback, cp_callback]

        history = model.fit(train_X, train_Y,
                            batch_size = self.batch_size,
                            epochs = self.num_epochs,
                            validation_split = self.valid_split,
                            callbacks = callback_list)

        return model, history

    def train_plot(self, model, history):
        model.plot_graphs(history, 'accuracy')
        model.plot_graphs(history, 'loss')
        model.plot_graphs(history, 'f1_score')

    def evaluation(self, model, test_X, test_Y):
        SAVE_FILE_NM = 'weights.h5' #저장된 best model 이름
        model.load_weights(os.path.join(self.DATA_OUT_PATH, self.kargs['model_name'], SAVE_FILE_NM))
        model.evaluate(test_X, test_Y)
