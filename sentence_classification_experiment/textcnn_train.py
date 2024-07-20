import numpy as np 
import pandas as pd 
import re 
import jieba  
from tensorflow.keras.preprocessing.text import * 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout

cate_list = ['市场监管', '金融行业', '社会民生', '教育行业', '食品安全', '社会治安', '刑事事件', '医疗卫生']
cate_dict = dict(zip(cate_list, [i for i in range(len(cate_list))]))
stop_word_set = set()
with open('stop_word.txt') as files:
    for line in files:
        line = line.strip('\n')
        stop_word_set.add(line)

class TextCNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 kernel_sizes=[3, 4, 5],
                 class_num=len(cate_list),
                 last_activation='sigmoid'):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(Conv1D(128, kernel_size, activation='relu'))
            self.max_poolings.append(GlobalMaxPooling1D())
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        # Embedding part can try multichannel as same as origin paper
        embedding = self.embedding(inputs)
        convs = []
        for i in range(len(self.kernel_sizes)):
            c = self.convs[i](embedding)
            c = self.max_poolings[i](c)
            convs.append(c)
        x = Concatenate()(convs)
        output = self.classifier(x)
        return output


def read_and_train_data():
    """
        读取数据
    """
    df = pd.read_csv('textcnn_predict_result', delimiter='\t')[['query', 'label', 'flag']]
    # df.columns = ['query', 'label']
    df['text'] = df['query'].apply(lambda x: str(x)[:26])
    print(f"the data is : ", df.head())


    def process_text(text):
        text = str(text)
        text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", text)
        text = ' '.join([i for i in jieba.lcut(text) if i not in stop_word_set])
        return text
    
    df.text = df.text.apply(process_text)
    df['text_len'] = df['text'].apply(lambda x: len(x))
    print(f"the data shape is: {df.shape} !!!")
    # df = df.query('text_len != 0')
    print(f"the filter data shape is: {df.shape}!!!")

    train_set = df.query('flag == "train"')
    test_set = df.query('flag == "test"')
    print(f"the train data shape is : {train_set.shape}, and the test data shape is: {test_set.shape}")
    max_document_length = 0

    # train_set['text'] = train_set['text'].apply(lambda x: str(x)[:32])
    # test_set['text'] = test_set['text'].apply(lambda x: str(x)[:32])
    

    print(test_set.head())
    
    def get_input_data(data):
        """
            获取模型的输入格式
        """
        labels = []
        x_datas = []
        max_length = 0
        for i, v in data.iterrows():
            # if(len(v['text'].strip()) == 0):
            #     continue
            x_datas.append(v['text'])
            labels.extend([v['label']])
            max_length = max(max_length, len(v['text'].split(' ')))
        return x_datas, labels, max_length
    train_x, train_y, max_len = get_input_data(train_set)
    max_document_length = max(max_document_length, max_len)
    test_x, test_y, max_len = get_input_data(test_set)
    max_document_length = max(max_document_length, max_len)


    train_all_x, train_all_y, _ = get_input_data(df)

    # 进行编码
    tk = Tokenizer()    # create Tokenizer instance
    tk.fit_on_texts(train_x)    # tokenizer should be fit with text data in advance
    word_size = max(tk.index_word.keys())


    sen = tk.texts_to_sequences(train_x)
    train_xx = sequence.pad_sequences(sen, padding='post', maxlen=max_document_length)
    train_yy = np.array(train_y)

    sen1 = tk.texts_to_sequences(test_x)
    test_xx = sequence.pad_sequences(sen1, padding='post', maxlen=max_document_length)
    test_yy = np.array(test_y)

    print('Build model...')
    # model = TextRNN(max_document_length, word_size+1, embedding_dims)
    model = TextCNN(max_document_length, word_size+1, embedding_dims=64)
    # model = TextAttBiRNN(max_document_length, word_size+1, embedding_dims)
    import tensorflow as tf 
    model.compile('adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    model.fit(train_xx, train_yy,
            batch_size=128,
            epochs=1,
            callbacks=[early_stopping],
            )

    print('Test...')
    # print(model.predict(test_xx))
    # exit(1)
    result = [','.join([str(j) for j in i]) for i in model.predict(test_xx)]


    test_set['textcnn_vec'] = result
    test_set['flag'] = 'test'
    # test_set.to_csv('test_textcnn_predict_new', index=False, sep='\t')

    train_result = [','.join([str(j) for j in i]) for i in model.predict(train_xx)]
    train_set['textcnn_vec'] = train_result
    train_set['flag'] = 'train'

    pd.concat([train_set, test_set]).drop('text', axis=1).to_csv('textcnn_predict_result', index=False, sep='\t')

    print(classification_report(test_yy, [np.argmax(i) for i in model.predict(test_xx)], digits=3))

if __name__ == "__main__":
    read_and_train_data()

