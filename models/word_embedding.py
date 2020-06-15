import logging
import gensim
import pandas as pd
import numpy as np
from collections import namedtuple
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def word2vec_model(text_data, **kwargs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(text_data,
                     window=kwargs['window'],
                     size=kwargs['num_features'],
                     min_alpha=kwargs['min_alpha'], # min learning-rate
                     min_count=kwargs['min_word_count'],
                     workers=kwargs['num_workers'],
                     seed=kwargs['seed'],
                     iter=kwargs['iter'],
                     sg=1,
                     hs=1,
                     negative = 10)
    return model

def doc2vec_model(text_data, **kwargs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Doc2Vec(size=kwargs['size'],
                                  window=kwargs['window'],
                                  min_count=kwargs['min_count'],
                                  alpha=kwargs['alpha'],
                                  min_alpha=kwargs['min_alpha'],
                                  workers=kwargs['workers'],
                                  seed=kwargs['seed'],
                                  iter=kwargs['iter'])
    model.build_vocab(text_data)
    model.train(text_data,
                epochs=model.iter,
                total_examples=model.corpus_count)
    return model

def get_embedding_matrix(embedding_model, word_index, EMBEDDING_DIM=300):
    vocabulary_size= len(word_index)+1
    NUM_WORDS = len(word_index)+1
    print("word index size : ", vocabulary_size)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    j, k = 0, 0
    for word, idx in word_index.items():
        if idx >= NUM_WORDS:
            continue
        try:
            embedding_vector = embedding_model[word]
            embedding_matrix[idx] = embedding_vector
            j += 1
        except KeyError as e:
            embedding_matrix[idx]=np.random.normal(0, np.sqrt(0.25),EMBEDDING_DIM)
            k += 1

    print(j, k)
    return embedding_matrix
