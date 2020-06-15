call venv.bat

python .\word2vec_pretrained\wiki_kr_w2v.py

python .\train\embedding_train.py --using_pretrained True
