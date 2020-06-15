import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import konlpy
from ast import literal_eval
from konlpy.tag import Kkma, Okt, Hannanum, Twitter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from tqdm import tqdm

def stopwords_remove(corpus_data, sw_list):
    docs=[]
    for corpus_list in tqdm(corpus_data):
        words=[]
        for w in corpus_list:
            if w.isdecimal():
                continue
            if (w not in sw_list):
                words.append(w)
        docs.append(words)
    return docs

if __name__ == '__main__':
    data = pd.read_excel("./doc_set_final_version.xlsx")
    zipfs_law_sw =  pd.read_csv('./stopwords/zipfs_law_sw_3.txt', delimiter='\t')['term'].tolist()
    big_corpus = data['token'].apply(literal_eval).tolist()
    data['token'] = stopwords_remove(big_corpus, zipfs_law_sw)

    data_tokens = [ t for d in data['token'] for t in d]
    data_text = nltk.Text(data_tokens, name='NMSC')
    data_fdist = data_text.vocab()

    data_fdist = pd.DataFrame.from_dict(data_fdist, orient='index')
    data_fdist.columns = ['frequency']
    data_fdist['term'] = list(data_fdist.index)
    data_fdist = data_fdist.reset_index(drop=True)
    data_fdist = data_fdist.sort_values(["frequency"], ascending=[False])
    data_fdist = data_fdist.reset_index(drop=True)
    data_fdist.to_excel('zipf_law6.xlsx')
    data.to_excel("doc_set_final_version2.xlsx", index=False, encoding='cp949')
