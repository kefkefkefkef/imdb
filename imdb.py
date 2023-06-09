import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import streamlit as st
import re
import string
from collections import Counter

from gensim.models import Word2Vec
from string import punctuation
import transformers
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchutils as tu
from torchmetrics.classification import BinaryAccuracy
from data.rnn_preprocessing import (
                                data_preprocessing, 
                                preprocess_single_string
                                )

def main():
    device = 'cpu'
    df = pd.read_csv('data/imdb.csv')
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    reviews = df['review'].tolist()
    preprocessed = [data_preprocessing(review) for review in reviews]

    wv = Word2Vec.load('models/word2vec32.model')

    words_list = [word for review in preprocessed for word in review.lower().split()]
    for i in words_list:
        ''.join([j for j in i if j not in punctuation])
        
    # делаем множество уникальных слов.
    unique_words = set(words_list)

    # word -> index
    vocab_to_int = {word: idx+1 for idx, word in enumerate(sorted(unique_words))}

    word_seq = [i.split() for i in preprocessed]
    VOCAB_SIZE = len(vocab_to_int) + 1  # add 1 for the padding token
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 64
    SEQ_LEN = 32

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

    for word, i in vocab_to_int.items():
        try:
            embedding_vector = wv.wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass

    embedding_layer32 = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))


    class LSTMClassifierBi32(nn.Module):
        def __init__(self, embedding_dim: int, hidden_size:int = 32) -> None:
            super().__init__()

            self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size
            self.embedding = embedding_layer32
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True
            )
            self.clf = nn.Sequential(nn.Linear(self.hidden_size*2, 128), 
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            embeddings = self.embedding(x)
            out, (_, _) = self.lstm(embeddings)
            out = self.clf(out[:,-1,:])
            return out
        
    model = LSTMClassifierBi32(embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_DIM)
    model.load_state_dict(torch.load('models/ltsm_bi1.pt'))
    model.eval()

    def predict_sentence(text:str, model: nn.Module):
        result = model.to(device)(preprocess_single_string(text, seq_len=SEQ_LEN, vocab_to_int=vocab_to_int).unsqueeze(0)).sigmoid().round().item()
        return 'negative' if result == 0.0 else 'positive'
    
    #Bag Tfidf
    # bagvectorizer = CountVectorizer(max_df=0.5,
    # min_df=5,
    # stop_words="english",)
    # bvect = bagvectorizer.fit(preprocessed)
    # X_bag = bvect.transform(preprocessed)

    tfid_vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5)
    vect = tfid_vectorizer.fit(preprocessed)
    X_tfidf = vect.transform(preprocessed)
    
    tfidf_model = pickle.load(open('models/modeltfidf.sav', 'rb'))
    # bag_model = pickle.load(open('models/modelbag.sav', 'rb'))
    # def predictbag(text):
    #     result = bag_model.predict(vect.transform([text]))
    #     return 'negative' if result == [0] else 'positive'

    def predicttf(text):
        result = tfidf_model.predict(vect.transform([text]))
        return 'negative' if result == [0] else 'positive'
    
        


    
    
    
    
    
    review = st.text_input('Enter review')

    start1 = time.time()
    
    automodel = transformers.AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
    )
    autotoken = transformers.AutoTokenizer.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
    )
    

    input_tokens = autotoken(
        review, 
        return_tensors='pt', 
        padding=True, 
        max_length=10
    )
    outputs = automodel(**input_tokens)
    st.write('Sentiment Predictions')
    st.write(f'\nBERT: {[automodel.config.id2label[i.item()] for i in outputs.logits.argmax(-1)]}')
    end1 = time.time()
    st.write(f'{(end1 - start1):.2f} sec')
    start2 = time.time()

    st.write(f'LTSM: {predict_sentence(review, model)}')
    end2 = time.time()
    st.write(f'{(end2 - start2):.2f} sec')
    # start3 = time.time()
    # st.write(f'bag+log: {predictbag(review)}')
    # end3 = time.time()
    # st.write(f'{(end3 - start3):.2f} sec')
    start4 = time.time()
    st.write(f'tfidf+log: {predicttf(review)}')
    end4 = time.time()
    st.write(f'{(end4 - start4):.2f} sec')


    

if __name__ == '__main__':
    main()