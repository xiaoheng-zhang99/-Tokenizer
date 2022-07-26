from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer

text1='some thing to eat'
text2='some some thing to drink'
text3='thing to eat food'
texts=[text1, text2, text3]

# keras.preprocessing.text.text_to_word_sequence(text,
#         filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
#         lower=True,
#         split=" ")
print(text.text_to_word_sequence(text3))
print(text.one_hot(text2,20))  #n表示编码值在1到n之间
print(text.one_hot(text2,5))

tokenizer = Tokenizer(num_words=5) #num_words:None或整数,个人理解就是对统计单词出现数量后选择次数多的前n个单词，后面的单词都不做处理。
tokenizer.fit_on_texts(texts)
print( "output1",tokenizer.texts_to_sequences(texts)) # 使用字典将对应词转成index。shape为 (文档数，每条文档的长度)
print( "output2",tokenizer.texts_to_matrix(texts)) # 转成one-hot，与前面的不同。shape为[len(texts),num_words]
print( "output3",tokenizer.word_counts) #单词在所有文档中的总数量，如果num_words=4，应该选择some thing to
print("output4", tokenizer.word_docs) #单词出现在文档中的数量
print( "output5",tokenizer.word_index) #单词对应的index
print( "output6",tokenizer.index_docs) #index对应单词出现在文档中的数量

# padding 填充位置pre/post truncating 超过maxlen后截取位置pre/post
# keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
#     padding='pre', truncating='pre', value=0.)
from keras_preprocessing import sequence
print(sequence.pad_sequences([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4]],15, padding='post', value=0, truncating='pre'))
print("padding2",sequence.pad_sequences([[1,2,3,4,5,6,7,8,9,10]],5, padding='pre', value=-1, truncating='pre'))

"""
from keras.preprocessing import text, sequence
# 生成的字典取频数高的max_feature个word对文本进行处理，其他的word都会忽略
max_feature = 30000
# 每个文本处理后word长度为maxlen
maxlen = 100
# 词嵌入长度取facebook fasttext的300长度
embedding_size = 300

# 使用keras进行分词，word转成对应index
tokenizer = text.Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(list(train_f)+list(test_f))
train_feature = tokenizer.texts_to_sequences(train_f)
test_feature = tokenizer.texts_to_sequences(test_f)
# 将每个文本转成固定长度maxlen，长的截取，短的填充0
train_feature = sequence.pad_sequences(train_feature, maxlen)
test_feature = sequence.pad_sequences(test_feature, maxlen)

# 词嵌入向量转dict的
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict( get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

# 文本中词的index映射对应的词嵌入
word_index = tokenizer.word_index
nb_words = min(max_feature, len(word_index)) # 基于文本的词典总长为len(word_index)，
# 由于使用max_feature进行了筛选，则最终使用的词典由max_feature, len(word_index)决定
embedding_matrix = np.zeros((nb_words, embedding_size))
for word, i in word_index.items():
    if i>= max_feature: continue #如果i大于max_feature,说明该词已经超出max_feature范围，无需处理
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
"""