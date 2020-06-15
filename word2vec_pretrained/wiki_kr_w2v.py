import logging
import gensim
import pandas as pd
import sys
import os
import multiprocessing
from ast import literal_eval
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts, get_tmpfile
from glob import glob
import json
import argparse
import re
from tqdm import tqdm
sys.path.append(os.getcwd())
from models.word_embedding import word2vec_model, doc2vec_model
from utils.nlp_tools import noun_corpus, stopwords_remove

def parse_args():
    # set crawler parser
    parser = argparse.ArgumentParser(description='w2v 학습 parameter 정하기')
    parser.add_argument('--num_features', help='word embedding dimension', default=300, type=int)
    parser.add_argument('--window', help='word embedding window size', default=8, type=int)
    parser.add_argument('--min_count', help='word embedding min_word_count', default=5, type=int)
    parser.add_argument('--min_alpha', help='word embedding min_alpha', default=0.025, type=float)
    parser.add_argument('--iter', help='word embedding train iteration', default=30, type=int)
    args = parser.parse_args()
    return args

def main(**kwargs):
    dir_list = os.listdir('./word2vec_pretrained/text/')
    stopwords_list = pd.read_csv('./word2vec_pretrained/korean_stopwords.txt')['stopwords'].tolist()
    tmp_text = []
    for dir in dir_list:
        print(dir)
        file_list = glob('./word2vec_pretrained/text/{}/*'.format(dir))
        for file_name in tqdm(file_list):
            with open(file_name, mode='r', encoding="utf8") as f:
                data = f.read()
            f.close()

            data = re.sub('</doc>', '',data)
            split_data  = re.split('<doc [^<>]*>', data)
            split_data = list(filter(lambda x: x != '', split_data))
            split_data = [re.sub(r'[^가-힣A-z0-9\s]', '',x) for x in split_data]
            split_data = [re.sub(r'\s{2,}', ' ',x) for x in split_data]
            split_data = [re.sub(r'\n', ' ',x).strip() for x in split_data]
            tmp_text.extend(split_data)
    print("문서개수", len(tmp_text))
    corpus=noun_corpus(tmp_text)
    print("토큰개수", len(corpus))
    sw_remove = stopwords_remove(corpus, stopwords_list)
    word2vec_kargs = {'num_features':kwargs['num_features'],
                      'num_workers':4,
                      'window':kwargs['window'],
                      'seed':1234,
                      'min_word_count':kwargs['min_count'],
                      'min_alpha':kwargs['min_alpha'],
                      'iter': kwargs['iter']}
    model = word2vec_model(sw_remove, **word2vec_kargs)
    print("모델 저장")
    model_name = './model_save/embedding_model/pretrained_word2vec1.model'
    model.save(model_name)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    kwargs = dict(args._get_kwargs())
    print(kwargs)
    main(**kwargs)
