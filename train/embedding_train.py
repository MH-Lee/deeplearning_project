import logging
import gensim
import pandas as pd
import sys
import os
import multiprocessing
from ast import literal_eval
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from sklearn.model_selection import train_test_split
import json
import argparse
sys.path.append(os.getcwd())
from models.word_embedding import word2vec_model, doc2vec_model

def parse_args():
    # set crawler parser
    parser = argparse.ArgumentParser(description='w2v or d2v 학습 parameter 정하기')
    parser.add_argument('--function', help='select w2v or d2v', default='w2v', type=str)
    parser.add_argument('--num_features', help='word embedding dimension', default=300, type=int)
    parser.add_argument('--window', help='word embedding window size', default=8, type=int)
    parser.add_argument('--min_count', help='word embedding min_word_count', default=5, type=int)
    parser.add_argument('--min_alpha', help='word embedding min_alpha', default=0.025, type=float)
    parser.add_argument('--iter', help='word embedding train iteration', default=30, type=int)
    parser.add_argument('--using_pretrained', help='using korean_pretrained model(True or False)', default=False, type=bool)
    args = parser.parse_args()
    return args

def main(**kwargs):
    data = pd.read_excel("./assets/data/doc_set_final_version3.xlsx")
    data.token = data.token.apply(lambda x : literal_eval(x))
    data = data.sample(frac=1, random_state=1234)

    token_list = data.token.tolist()
    target = data[['new_class', 'new_small_class']]
    train_x_data, test_x_data, train_y, test_y = train_test_split(token_list, target,
                                                                    test_size=0.3,
                                                                    stratify=target,
                                                                    shuffle=True,
                                                                    random_state=1234)

    model_select = kwargs['function']
    if model_select == 'w2v':
        word2vec_kargs = {'num_features':kwargs['num_features'],
                          'num_workers':4,
                          'window':kwargs['window'],
                          'seed':1234,
                          'min_word_count':kwargs['min_count'],
                          'min_alpha':kwargs['min_alpha'],
                          'iter': kwargs['iter']}
        if kwargs['using_pretrained'] == True:
            print("사전훈련모델 학습")
            model = Word2Vec.load('./model_save/embedding_model/pretrained_word2vec1.model')
            model.build_vocab(train_x_data, update=True)
            total_examples = model.corpus_count
            model.train(train_x_data, total_examples = total_examples, epochs=30)
            model_name = './model_save/embedding_model/trained_word2vec1.model'
        else:
            print("모델 학습")
            model = word2vec_model(train_x_data, **word2vec_kargs)
            model_name = './model_save/embedding_model/word2vec1.model'
        print("모델 저장")
        model.save(model_name)

    elif model_select == 'd2v':
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')
        tagged_train_docs = [TaggedDocument(d, [c[1]['new_class'], c[1]['new_small_class']]) for d, c in zip(train_x_data, train_y.iterrows())]
        print("모델 학습")
        doc2vec_kargs = {'size':kwargs['num_features'], #300
                         'window':kwargs['window'], #10
                         'min_count':kwargs['min_count'], # 3
                         'alpha':0.025,
                         'min_alpha':kwargs['min_alpha'], # 0.025
                         'workers':4,
                         'seed':1234,
                         'iter':kwargs['iter']} # 30
        model = doc2vec_model(tagged_train_docs, **doc2vec_kargs)
        print("모델 저장")
        model.save('./model_save/embedding_model/Doc2vec_new.model')

    else:
        print("2가지 방식 중에 고르시오")

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    kwargs = dict(args._get_kwargs())
    print(kwargs)
    main(**kwargs)
