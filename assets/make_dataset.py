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

Today = datetime.date.today().strftime("%Y-%m-%d")

################################################################################
######## File Name Setting #####################################################
################################################################################
# Data save label
TRAIN_INPUT_DATA = 'train_input.npy'
TEST_INPUT_DATA = 'test_input.npy'
DATA_CONFIGS = 'data_configs.json'
SEQ_CONFIGS = 'seq_configs.json'
TOK_CONFIG = 'tokenizer_config.json'

# Train label save file name
TRAIN_LABEL = 'train_label.npy'
# TRAIN_LABEL_SPARSE = 'train_label_sparse.npy'
TRAIN_LABEL_SMALL = 'train_label_small.npy'
# TRAIN_LABEL_SMALL_SPARSE = 'train_label_small_sparse.npy'

# Test label save file name
TEST_LABEL = 'test_label.npy'
# TEST_LABEL_SPARSE = 'test_label_sparse.npy'
TEST_LABEL_SMALL = 'test_label_small.npy'
# TEST_LABEL_SMALL_SPARSE = 'test_label_small_sparse.npy'

#File name
LABEL_JSON_NAME = './assets/label_data/label.json'
LABEL_JSON_NAME_SMALL = './assets/label_data/label_small.json'

## Data Load
total_data = pd.read_excel('./assets/data/doc_set_final_version3.xlsx')
stop_words = pd.read_csv('./assets/data/korean_stopwords.txt')['stopwords'].tolist()
total_data['token'] = total_data['token'].apply(lambda x: literal_eval(x))
total_data = total_data.sample(frac=1, random_state=1234)

data = total_data[['token', 'new_small_class']]
token_list = total_data['token'].tolist()
target = total_data['new_class'].tolist()

### target class (big and small) encoding
lbl_e = LabelEncoder()
target_label = lbl_e.fit_transform(total_data['new_class'])
le_name_mapping = dict(zip(lbl_e.transform(lbl_e.classes_), lbl_e.classes_))
le_dict = dict()
for k, v in le_name_mapping.items():
    le_dict[str(k)] = v
json.dump(le_dict, open(LABEL_JSON_NAME, 'w'), ensure_ascii=True)

lbl_e2 = LabelEncoder()
target_label_small = lbl_e2.fit_transform(total_data['new_small_class'])
le_small_name_mapping = dict(zip(lbl_e2.transform(lbl_e2.classes_), lbl_e2.classes_))
data['new_small_class_le'] = target_label_small
le_dict2 = dict()
for k, v in le_small_name_mapping.items():
    le_dict2[str(k)] = v
json.dump(le_dict2, open(LABEL_JSON_NAME_SMALL, 'w'), ensure_ascii=True)

train_x_data, test_x_data, train_y, test_y = train_test_split(data, target_label,
                                                             test_size=0.3,
                                                             stratify=target,
                                                             shuffle=True,
                                                             random_state=1234)

print(train_x_data.shape, test_x_data.shape, len(train_y), len(test_y))
print(len(train_x_data.iloc[:, 0]), len(test_x_data.iloc[:, 0]), len(np.unique(train_y)), len(np.unique(test_y)))

label_number = len(le_dict.keys())
print("big 클래스 개수 : ", label_number)
print("학습데이터 클래스 개수 : ", len(np.unique(train_y)))
print("검증데이터 클래스 개수 : ",len(np.unique(test_y)))

label_number_small = len(le_dict2.keys())
print("small 클래스 개수 : ", label_number_small)
print("학습데이터 클래스 개수 : ", len(np.unique(train_x_data['new_small_class_le'])))
print("검증데이터 클래스 개수 : ", len(np.unique(test_x_data['new_small_class_le'])))

after_len = [len(word) for word in train_x_data['token']]
print('전처리 후 명사 길이 최대 값: {}'.format(np.max(after_len)))
print('전처리 후 명사 길이 최소 값: {}'.format(np.min(after_len)))
print('전처리 후 명사 길이 평균 값: {:.2f}'.format(np.mean(after_len)))
print('전처리 후 명사 길이 표준편차: {:.2f}'.format(np.std(after_len)))
print('전처리 후 명사 길이 중간 값: {}'.format(np.median(after_len)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('전처리 후 명사 길이 제 1 사분위: {}'.format(np.percentile(after_len, 25)))
print('전처리 후 명사 길이 제 3 사분위: {}'.format(np.percentile(after_len, 75)))

# plt.hist(after_len, bins=10)
# plt.title("The histogram of token length")
# plt.show()

## make one-hot encoding label data
# train_y_s = tf.keras.utils.to_categorical(train_y, num_classes = label_number)
# test_y_s= tf.keras.utils.to_categorical(test_y, num_classes = label_number)
# train_y_ss = tf.keras.utils.to_categorical(train_x_data['new_small_class_le'], num_classes = label_number_small)
# test_y_ss= tf.keras.utils.to_categorical(test_x_data['new_small_class_le'], num_classes = label_number_small)

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(train_x_data['token'])

low_count_words = [w for w,c in tokenizer.word_counts.items() if c < 5]
for w in low_count_words:
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]
train_sequence = tokenizer.texts_to_sequences(train_x_data['token'])
test_sequence = tokenizer.texts_to_sequences(test_x_data['token'])

sequence_data = dict()
sequence_data['train_seq'] = train_sequence
sequence_data['test_seq'] = test_sequence
sequence_data['train_token_list'] = train_x_data['token'].tolist()
sequence_data['test_token_list'] = test_x_data['token'].tolist()
sequence_data['tokenizer_config'] = tokenizer.get_config()

word_idx = tokenizer.word_index
MAX_SEQUENCE_LENGTH = int(np.median(after_len))
DATA_OUT_PATH = './assets/data/npy_data/{}/'.format(Today)
## Make output save directory
if os.path.exists(DATA_OUT_PATH):
    print("{} -- Folder already exists \n".format(DATA_OUT_PATH))
else:
    os.makedirs(DATA_OUT_PATH, exist_ok=True)
    print("{} -- Folder create complete \n".format(DATA_OUT_PATH))

train_input = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(train_y)
train_small_labels = np.array(train_x_data['new_small_class_le'])
test_input = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(test_y)
test_small_labels = np.array(test_x_data['new_small_class_le'])

data_configs = {}
data_configs['vocab'] = word_idx
data_configs['vocab_size'] = len(word_idx)

### DATA SAVE
# 전처리 된 데이터를 넘파이 형태로 저장
np.save(open(DATA_OUT_PATH + TRAIN_INPUT_DATA, 'wb'), train_input)
np.save(open(DATA_OUT_PATH + TEST_INPUT_DATA, 'wb'), test_input)

# save label numpy file
np.save(open(DATA_OUT_PATH + TRAIN_LABEL, 'wb'), train_labels)
np.save(open(DATA_OUT_PATH + TEST_LABEL, 'wb'), test_labels)
# np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SPARSE, 'wb'), train_y_s)
# np.save(open(DATA_OUT_PATH + TEST_LABEL_SPARSE, 'wb'), test_y_s)

# save small label numpy file
np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SMALL, 'wb'), train_small_labels)
np.save(open(DATA_OUT_PATH + TEST_LABEL_SMALL, 'wb'), test_small_labels)
# np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SMALL_SPARSE, 'wb'), train_y_ss)
# np.save(open(DATA_OUT_PATH + TEST_LABEL_SMALL_SPARSE, 'wb'), test_y_ss)

json.dump(data_configs, open(DATA_OUT_PATH + DATA_CONFIGS, 'w'), ensure_ascii=True)
json.dump(sequence_data, open(DATA_OUT_PATH + SEQ_CONFIGS, 'w'), ensure_ascii=True)
