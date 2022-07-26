import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
data_path = '/Users/zhangxiaoheng/Desktop/data/IMDB_Dataset.csv'
imdb_data=pd.read_csv(data_path)

#print(imdb_data.shape)
#(50000, 2)

#print(imdb_data.head(10))

#print(imdb_data.keys())
#Index(['review', 'sentiment'], dtype='object')
#print("2",imdb_data['sentiment'].value_counts())
#2 positive    25000 negative    25000  Name: sentiment, dtype: int64

labels=[]
texts=[]
for sentiment in imdb_data['sentiment']:
    labels.append(sentiment)
for review in imdb_data['review']:
    texts.append(review)

maxlen = 100    #只读取评论的前100个单词
max_words = 10000   #只考虑数据集中最常见的前10000个单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts) #将texts文本转换成整数序列

word_index = tokenizer.word_index #单词和数字的字典
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen) #将data填充为一个(sequences, maxlen)的二维矩阵

data = np.asarray(data).astype('int64')
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, labels)))
labels = np.asarray(y).astype('float32')

#print(data.shape, labels.shape)
train_reviews = data[: 40000]
train_sentiments = labels[:40000]

test_reviews = data[40000:]
test_sentiments = labels[40000:]

#show train datasets and test datasets shape
#print(train_reviews.shape,train_sentiments.shape)
#print(test_reviews.shape,test_sentiments.shape)
#使用glove中10000个单词的向量

glove_dir = '/Users/zhangxiaoheng/Desktop/data/glove.6B'

embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs

f.close()

print('Found %s word vectors' % len(embedding_index))

#准备GloVe词嵌入矩阵


embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#print(embedding_matrix)
from keras import Sequential
from keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.utils import plot_model

network = Sequential()
network.add(Embedding(max_words, embedding_dim, input_length=100))
network.add(Flatten())
network.add(Dense(32, activation='relu'))
network.add(Dense(1, activation='sigmoid'))
network.summary()



plot_model(network, show_shapes = True)

#将预训练的的词嵌入加载到Embedding层中
network.layers[0].set_weights([embedding_matrix])
network.layers[0].trainable = False

network.summary()


network.compile('rmsprop', 'binary_crossentropy', 'accuracy')

history = network.fit(train_reviews, train_sentiments,
                    epochs=10,
                    batch_size=32,
                    validation_data=(test_reviews, test_sentiments))



