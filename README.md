# MGE-51101 Final Project Code (temporary)


## 1. Environment

### 1-1. Package version

```
tensorflow-gpu ~= 2.2.x
tensorflow-addons ~= 0.10.x
gensim ~= 3.8.x
nltk ~= 3.5
pandas ~= 1.0.x
sckit-learn ~= 0.22.x
numpy ~= 1.18.x
matplotlib ~= 3.1.x
soynlp == 0.0.493 # korean nlp package
```

## 2. Make dataset

### data preprocessing

I eliminated manually stop-words and apply Zipf’s Law to the corpus with noun words, a method to remove either too common or too rare words based on the tokens in a corpus.

Zipf's law and preprocess code can be checked in the following folder.(`./utils/nlp_tools` and `./utils/zipfs_law.py`)

### Make dataset
```
python ./assets/make_dataset.py
```
This code creates a folder with the name of today's date in the `assets/data/` folder, and contains the data.


## 3. Make Pre-trained model

### 3-1. Train korean wikipedia pre-trained model

I trained the word2vec pre-trained model using a total of 460,000 Korean wikipedia data.

You can download Korean Wikipedia data through the link below
(https://dumps.wikimedia.org/kowiki/latest/)

And Then, in the `./word2vec_pretrained/text/` folder, extract the text file through the code at the link below and run the following code to run the word2vec pre-trained model.

```
call ./word2vec_pretrained/wiki_kr_w2v.py
```
You can set detail parameter of word2vec model
```
python ./word2vec_pretrained/wiki_kr_w2v.py --num_features [int] \
                                            --window [int] \
                                            --min_count [int] \
                                            --min_alpha [float] \
                                            --iter [int]
```
+ --num_features : [int] [default : 300] \
 Word embedding vector dimension
+ --window       : [int] [default : 5] \
Maximum distance between the current and predicted word within a sentence.
+ --min_count    : [int] [default : 2] \
Ignores all words with total frequency lower than this.
+ --min_alpha    : [float] [default : 0.025] \
 Learning rate will linearly drop to min_alpha as training progresses.
+ --iter         : [int] [default : 30]
  Number of iterations (epochs) over the corpus.

training technology document corpus in a pre-trained model

```
python ./train/embedding_train.py --function [str]
                                  --num_features [int] \
                                  --window [int] \
                                  --min_count [int] \
                                  --min_alpha [float] \
                                  --iter [int]
```
+ word2vec or doc2vec parameter is same above
+ --function : [str] [default : w2v]  
  select word or document embedding model (w2v or doc2vec)
+ --using_pretrained : [bool] [default : False]  
 select using pre-trained model

```
python .\train\embedding_train.py --using_pretrained True
```

### 3-2 train only technology document with word2vec model

```
python .\train\embedding_train.py --function w2v
```

## 4. Train and evaluate deep learning model

### Train CNN model

```
python .\train\cnn_train.py --num_epoch [int] \
                            --hidden_dimension [int] \
                            --batch_size [int] \
                            --embedding_size [int] \
                            --num_filters [int] \
                            --dropout_rate [float] \
                            --valid_split [float] \
                            --pre_trained_mode [str] \
                            --train_mode [str] \
                            --optimizer [str] \
                            --output_dimension [int]
```

+ --num_epoch [int] [default : 1000] : Maximum number of training epoch
+ --hidden_dimension [int] [default : 1000] : the number of dimension in the hidden layer.
+ --batch_size [int] [default : 512] : The number of training samples present in a single batch.
+ --embedding_size [int] [default : 300] : The dimensionality of word embedding matrix.
(input_size = vocaburary size , output_size = embedding_size)
+ --num_filters [int] [default : 128] : the number of filters in the convolution.
+ --dropout_rate [float] [default : 0.5] : Set dropout_rate.
+ --valid_split [float] [default : 0.2] : the ratio of validation data in training data.
+ --pre_trained_mode [str] [default : w2v] :  Select pre-trained embedding model ((1) w2v : word2vec with technology document, (2) d2v : doc2vec with technology document, (3) pt_w2v : word2vec with korean wikipedia + technology document)
+ --train_mode [str] [default : non_static] : Select train mode.((1) rand : randomly weight initialize. (2) static : model with pre-trained vector. (3) model with pre-trained vectors and update each task )
+ --optimizer [str] [default : adam] : Select optimizer ((1) Adam (2) Radam)
+ --output_dimension [int] [default]

### Train RNN(bi-direction LSTM) model

```
python .\train\lstm_train.py --num_epoch [int] \
                             --batch_size [int] \
                             --embedding_size [int] \
                             --lstm_dimension [int] \
                             --dense_dimension [int] \
                             --dropout_rate [float] \
                             --valid_split [float] \
                             --pre_trained_mode [str] \
                             --train_mode [str] \
                             --optimizer [str] \
                             --output_dimension [int]
```
+ --num_epoch [int] [default : 1000] : Maximum number of training epoch.
+ --batch_size [int] [default : 512] : The number of training samples present in a single batch.
+ --embedding_size [int] [default : 300] : The dimensionality of word embedding matrix.
+ --lstm_dimension [int] [default : 128] : The dimensionality of the LSTM layer output space. The size of LSTM layer units.
+ --dense_dimension [int] [default : 128] : The size of FC layer units.
+ --dropout_rate [float] [default : 0.5] : Set dropout_rate.
+ --valid_split [float] [default : 0.2] : the ratio of validation data in training data.
+ --pre_trained_mode [str] [default : w2v] : Select pre-trained embedding model ((1) w2v : word2vec with technology document, (2) d2v : doc2vec with technology document, (3) pt_w2v : word2vec with wikipedia + technology document)
+ --train_mode [str] [default : pt] : Select train mode ((1) rand : randomly weight initialize (2) pt : reflect the pre-trained model's weights)
+ --optimizer [str] [default : adam] : Select optimizer ((1) Adam (2) Radam)

* * *

## 5. Jupyter Notebook link

1. Make dataset jupyer notebook link
[01.make_dataset.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/01.make_dataset.ipynb)

2. train word_embedding
[02.word_embedding_train.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/02.word_embedding_train.ipynb)

3. Train CNN model
[03.CNN_model.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/03.CNN_model.ipynb)

3-1. Train CNN model(small_class)
[03-1.CNN_model_small.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/03-1.CNN_model_small.ipynb)

4. Train LSTM model
[04.LSTM_model.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/04.LSTM_model.ipynb)

4-1. Train LSTM model(small_class)
[04-1.LSTM_model_small.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/04-1.LSTM_model_small.ipynb)

6. Model evaluation
[05.evaluation.ipynb](https://github.com/leemh012/mge51101-20196013/blob/master/final_project/notebook_example/05.evaluation.ipynb)
"# deeplearning_project"
