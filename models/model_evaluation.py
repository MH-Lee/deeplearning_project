import sys
import os
import numpy as np
import tensorflow_addons as tfa
sys.path.append(os.pardir)
from models.cnn_kr import CNNClassifier
from models.lstm_kr import RNNClassifier
from models.word_embedding import get_embedding_matrix

def model_build(train_X, word_index, embedding_matrix,**kargs):
    f1_score = tfa.metrics.F1Score(
        num_classes= kargs['output_dimension'],
        average='macro'
    )

    if kargs['optimizer'] == 'radam':
        optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)
    else:
        optimizer =  kargs['optimizer']
    kargs['embedding_matrix'] = get_embedding_matrix(embedding_matrix, word_index)
    if 'lstm' in kargs['model_name']:
        print('rnn')
        if kargs['output_dimension'] > 43:
            print('small_class')
            DATA_OUT_PATH = '../model_save/rnn_model_small/'
        else:
            print('big_class')
            DATA_OUT_PATH = '../model_save/rnn_model/'
        model = RNNClassifier(**kargs)
    elif 'cnn' in kargs['model_name']:
        print('cnn')
        if kargs['output_dimension'] > 43:
            print('small_class')
            DATA_OUT_PATH = '../model_save/cnn_model_small/'
        else:
            print('big_class')
            DATA_OUT_PATH = '../model_save/cnn_model/'
        model = CNNClassifier(**kargs)
    else:
        raise Exception('cnn과 rnn 중에 선택하십시오')
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', f1_score])
    model.build(train_X.shape)
    print(model.summary())

    SAVE_FILE_NM = 'weights.h5'
    print(os.path.join(DATA_OUT_PATH, kargs['model_name'], SAVE_FILE_NM))
    model.load_weights(os.path.join(DATA_OUT_PATH, kargs['model_name'], SAVE_FILE_NM))
    return model

def return_two_calss_acc(model, x_test, y_test):
    y_pred_prob = model.predict(x_test)
    L = np.argsort(-y_pred_prob, axis=1)
    two_pred = L[:,0:2]
    score = []
    score = []
    for i in range(len(y_test)):
        label = two_pred[i]
        if y_test[i] in label :
            score.append(1)
        else :
            score.append(0)
    acc = sum(score)/len(y_test)
    return acc
