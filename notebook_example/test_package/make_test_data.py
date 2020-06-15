################################################################################
########  1. package load   ####################################################
################################################################################
import pandas as pd
import numpy as np
import re, os, json, sys
import tensorflow as tf
import argparse
import datetime
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
sys.path.append('..\\..')
from utils.nlp_tools import text_cleaner, noun_corpus, stopwords_remove

Today = datetime.date.today().strftime("%Y-%m-%d")

data = pd.read_excel('./notebook_example/application_data/professor2.xlsx')
stop_words = pd.read_csv('./assets/data/korean_stopwords.txt')['stopwords'].tolist()
add_sw = ['교수', '연구팀', '유튜브', '사람들', '기술','최근', '연구', '개발',  '진행',
          '연구비', '지원', '소속', '구성', '대학원생', '미국', '게재', '교수신문', '무단전재',
          '재배', '금지', '세계적', '권위', '미국', '온라인', '속보' , '분야', '활용', '편리',
          '관련', '사용', '의미', '것으로', '기대', '의미', '특훈교수', '과정',  '대전', '이용',
          '산업', '향후', '방법', '생산', '바이', '최초', '방식', '경제성', '기반', '사진제공', '가능',
          '다방면', '여러', '주로', '의의', '단순',  '환경', '문제', '때문', '우려', '여러','소개', '박사후연구원',
          '박사과정', '공동', '저자', '참여', '다방면', '기법']
media = data['신문사'].tolist()
stop_words.extend(add_sw)
stop_words.extend(media)
data['content'] = data['content'].apply(lambda x: text_cleaner(str(x)).strip())

noun_list = noun_corpus(data['content'] )
data['token'] = stopwords_remove(noun_list, stop_words)

data['len'] = data.token.apply(len)
data = data[data['len'] > np.percentile(data['len'], 25)]
data = data[data['len'] < np.percentile(data['len'], 90)]
print('전처리 후 명사 길이 최대 값: {}'.format(np.max(data['len'])))
print('전처리 후 명사 길이 최소 값: {}'.format(np.min(data['len'])))
print('전처리 후 명사 길이 평균 값: {:.2f}'.format(np.mean(data['len'])))
print('전처리 후 명사 길이 표준편차: {:.2f}'.format(np.std(data['len'])))
print('전처리 후 명사 길이 중간 값: {}'.format(np.median(data['len'])))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('전처리 후 명사 길이 제 1 사분위: {}'.format(np.percentile(data['len'], 25)))
print('전처리 후 명사 길이 제 3 사분위: {}'.format(np.percentile(data['len'], 90)))
DATA_OUT_PATH = './notebook_example/application_data/'
data.to_excel(DATA_OUT_PATH + 'application_benchmark.xlsx', index=False)
MAX_SEQUENCE_LENGTH = int(np.median(data['len']))

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(data['token'])

low_count_words = [w for w,c in tokenizer.word_counts.items() if c < 5]
for w in low_count_words:
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]
test_sequence = tokenizer.texts_to_sequences(data['token'])
test_input = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

TEST_INPUT_DATA = 'professor.npy'
np.save(open(DATA_OUT_PATH + TEST_INPUT_DATA, 'wb'), test_input)
