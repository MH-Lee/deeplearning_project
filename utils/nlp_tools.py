import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import konlpy
from konlpy.tag import Kkma, Okt, Hannanum, Twitter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from tqdm import tqdm
from multiprocessing import Process

def text_cleaner(sents):
    text_data = re.sub("[^가-힣ㄱ-하-ㅣA-z\\s0-9\-]", " ", sents)
    text_data = re.sub("[\r.*?\n]", " ", text_data)
    text_data = re.sub("\x0c", " ", text_data)
    text_data = re.sub(r"\s{2,}", " ", text_data)
    text_data = text_data.replace('-', ' ')
    text_data = text_data.replace("SW", " 소프트웨어 ")
    text_data = text_data.replace("ARVR", " AR VR ")
    text_data = text_data.replace("비정형정형", "비정형 정형")
    for sys_ in ["system", "systems", "systems", "System", "Systems", "SYSTEM"]:
        text_data = text_data.replace(sys_, "")
    text_data = text_data.replace("Systems미국", "")
    return text_data

def noun_corpus(sents):
    noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
    noun_extractor.train(sents)
    nouns = noun_extractor.extract()

    noun_scores = {noun:score[0] for noun, score in nouns.items() if len(noun) > 1}
    tokenizer = NounLMatchTokenizer(noun_scores)
    corpus = [tokenizer.tokenize(sent) for sent in sents]
    return corpus

def stopwords_remove(corpus_data, sw_list):
    docs=[]
    for corpus_list in tqdm(corpus_data):
        words=[]
        for w in corpus_list:
            if w.isdecimal():
                continue
            if (w not in sw_list) & (len(w) > 1):
                words.append(w)
        docs.append(words)
    return docs
