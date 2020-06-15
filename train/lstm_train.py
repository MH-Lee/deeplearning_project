import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import argparse
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
sys.path.append(os.getcwd())
from models.lstm_kr import RNNClassifier
from models.word_embedding import get_embedding_matrix

label_size = input('select big or small')
DATA_IN_PATH = './assets/data/npy_data/2020-05-31/'
DATA_OUT_PATH = './model_save/rnn_model/'
# Data save label
TRAIN_INPUT_DATA = 'train_input.npy'
TEST_INPUT_DATA = 'test_input.npy'
DATA_CONFIGS = 'data_configs.json'
SEQ_CONFIGS = 'seq_configs_bt.json'

# Train label save file name
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_LABEL_SMALL = 'train_label_small.npy'
TEST_LABEL_DATA = 'test_label.npy'
TEST_LABEL_SMALL = 'test_label_small.npy'

# pre-trained model load
d2v_model_name = './model_save/embedding_model/Doc2vec_new.model'
w2v_model_name = './model_save/embedding_model/Word2vec1.model'
pre_trained_name = './model_save/embedding_model/trained_word2vec1.model'

doc_vectorizer = Doc2Vec.load(d2v_model_name)
word_vectorizer = Word2Vec.load(w2v_model_name)
pre_trained_w2v = Word2Vec.load(pre_trained_name)

train_X = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
test_X = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))

if label_size == 'big':
    train_Y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
    train_YS = tf.one_hot(train_Y, 43)
    test_Y = np.load(open(DATA_IN_PATH + TEST_LABEL_DATA, 'rb'))
    test_YS = tf.one_hot(test_Y, 43)
else:
    train_Y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_SMALL, 'rb'))
    train_YS = tf.one_hot(train_Y, 455)
    test_Y = np.load(open(DATA_IN_PATH + TEST_LABEL_SMALL, 'rb'))
    test_YS = tf.one_hot(test_Y, 455)

data_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
print("vacab_size : ", data_configs['vocab_size'])
word_index = data_configs['vocab']

def parse_args():
    # set crawler parser
    parser = argparse.ArgumentParser(description='lstm 학습 parameter 정하기')
    parser.add_argument('--batch_size', help='batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', help='train epochs', default=1000, type=int)
    parser.add_argument('--embedding_size', help='embedding size', default=300, type=int)
    parser.add_argument('--lstm_dimension', help='lstm_dimension', default=128, type=int)
    parser.add_argument('--dense_dimension', help='dense_dimension', default=128, type=int)
    parser.add_argument('--valid_split', help='valid split', default=0.2, type=float)
    parser.add_argument('--dropout_rate', help='dropout rate', default=0.5, type=float)
    parser.add_argument('--pre_trained_mode', help='(1) d2v (2) w2v (3) pt_w2v (4) None', default=None, type=str)
    parser.add_argument('--train_mode', help='(1) rand (2) pt (using pre-trained)', default='rand', type=str)
    parser.add_argument('--output_dimension', help='set output_dimension', default=43, type=int)
    parser.add_argument('--optimizer', help='select optimizer adam or radam(tfa) or others', default='adam', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if (args.train_mode == 'rand') and (args.pre_trained_mode != None):
        raise Exception('rand는 pretrained vector를 가질 수 없습니다.')
    f1_score = tfa.metrics.F1Score(
        num_classes= args.output_dimension,
        average='macro'
    )
    kargs = {'vocab_size'     :  data_configs['vocab_size']+1,
            'embedding_size'  :  args.embedding_size,
            'lstm_dimension'  :  args.lstm_dimension,
            'dense_dimension' :  args.dense_dimension,
            'dropout_rate'    :  args.dropout_rate,
            'train_mode'      :  args.train_mode,
            'output_dimension':  args.output_dimension}
    model_name = 'lstm_{}_{}'.format(args.train_mode, args.optimizer)
    if args.optimizer == 'radam':
        optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)
    else:
        optimizer = args.optimizer
    if args.train_mode == 'rand':
        kargs['model_name'] = model_name
    else:
        if args.pre_trained_mode == 'w2v':
            emb_matrix = get_embedding_matrix(word_vectorizer, word_index)
        elif args.pre_trained_mode == 'pt_w2v':
            emb_matrix = get_embedding_matrix(pre_trained_w2v, word_index)
        else:
            emb_matrix = get_embedding_matrix(doc_vectorizer, word_index)
        kargs['model_name'] = model_name + '_' + args.pre_trained_mode
        kargs['embedding_matrix'] = emb_matrix
    print(kargs)
    model = RNNClassifier(**kargs)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1_score])
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.001,patience=5)

    checkpoint_path = DATA_OUT_PATH + kargs['model_name'] + '/weights.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create path if exists
    if os.path.exists(checkpoint_dir):
        print("{} -- Folder already exists \n".format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("{} -- Folder create complete \n".format(checkpoint_dir))

    cp_callback = ModelCheckpoint(checkpoint_path,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True)

    history = model.fit(train_X, train_YS,
                       batch_size=args.batch_size,
                       epochs=args.num_epochs,
                       validation_split=args.valid_split,
                       callbacks=[earlystop_callback, cp_callback])

    # 결과 평가하기
    SAVE_FILE_NM = 'weights.h5' #저장된 best model 이름
    model.load_weights(os.path.join(DATA_OUT_PATH, kargs['model_name'], SAVE_FILE_NM))
    model.evaluate(test_X, test_YS)

    # train_loss, accuracy 확인
    model.plot_graphs(history, 'accuracy')
    model.plot_graphs(history, 'loss')
    model.plot_graphs(history, 'f1_score')
