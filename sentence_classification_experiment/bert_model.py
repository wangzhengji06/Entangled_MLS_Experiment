import re
import logging
import jieba 
logging.basicConfig(level=logging.ERROR)
# from transformers import TFBertPreTrainedModel,TFBertMainLayer,BertTokenizer
from transformers import TFBertForSequenceClassification,BertTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')



cate_list = ['市场监管', '金融行业', '社会民生', '教育行业', '食品安全', '社会治安', '刑事事件', '医疗卫生']
cate_dict = dict(zip(cate_list, [i for i in range(len(cate_list))]))
# stop_word_set = set()
# with open('stop_word') as files:
#     for line in files:
#         line = line.strip('\n')
#         stop_word_set.add(line)

def convert_example_to_feature(review):
  
    # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
    return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
            truncation=True
              )
# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
  
    for index, row in ds.iterrows():
        review = row["text"]
        label = row["label"]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)



def split_dataset(df):
    train_set, x = train_test_split(df, 
        stratify=df['label'],
        test_size=0.1, 
        random_state=42)
    val_set, test_set = train_test_split(x, 
        stratify=x['label'],
        test_size=0.5, 
        random_state=43)

    return train_set,val_set, test_set


def read_data():
    """
        读取数据
    """
    df = pd.read_csv('textcnn_predict_result', sep='\t').dropna()[['query', 'label', 'flag']]
    df['text'] = df['query'].apply(lambda x: re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",x))
    df['text'] = df['text'].apply(lambda x: str(x)[:23])

    print(f"the data is : ", df.shape)

    train_set = df.query('flag =="train"')
    test_set = df.query('flag =="test"')
    return train_set, test_set

if __name__ == '__main__': 

    # parameters
    model_path = "./bert-base-chinese" #模型路径
    max_length = 32
    batch_size = 128
    learning_rate = 2e-5
    number_of_epochs = 1
    num_classes = len(cate_list) # 类别数

    train_data, test_data = read_data()
    print(train_data.shape, test_data.shape)
    print(test_data.head())

#     # tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # train dataset
    ds_train_encoded = encode_examples(train_data).batch(batch_size)

    # val dataset
    # ds_val_encoded = encode_examples(val_data).batch(batch_size)
    # test dataset
    ds_test_encoded = encode_examples(test_data).batch(batch_size)

    # model initialization
    model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

    # optimizer Adam recommended
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    # fit model
    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs)
    # evaluate test_set
    
    
      
    # 对数据进行预测，需要对数据处理成模型的输入方式即可
    def predict(model, data):
        """
            对测试数据进行预测
        """
        # model = models.load_model("bert_base_chinese_new", compile=False)
        note_list = []
        pre_label_list = []
        real_label_list = []
        for i in list(data.as_numpy_iterator()):
            note = [''.join(tokenizer.convert_ids_to_tokens(i, skip_special_tokens=True)) for i in i[0]['input_ids']]
            pre_res = model.predict(i[0])
            # pre_label = [np.argmax(i) for i in list(pre_res[0])]
            pre_label = [i for i in list(pre_res[0])]
            real_label = [i[0] for i in i[1]]
            # print(pre_label)
            note_list.extend(note)
            pre_label_list.extend(pre_label)
            real_label_list.extend(real_label)
        df_res = pd.DataFrame({'text':note_list, 'real_label':real_label_list, 'bert_vec':pre_label_list})
        df_res['bert_vec'] = df_res['bert_vec'].apply(lambda x: ','.join([str(i) for i in x]))
        return df_res 
    
    # 对数据进行预测
    df_test = predict(model, ds_test_encoded)
    df_train = predict(model, ds_train_encoded)

    print(df_train.head())
    print(df_test.head())
    print(df_train.shape, df_test.shape, df_train.columns, df_test.columns)
    df_test.to_csv('bert_test', index=False, sep='\t')

    df_test = pd.read_csv('bert_test', delimiter='\t')
    print(f"the predict test data is: {df_test.shape}")
    train_data['bert_vec'] = df_train['bert_vec']
    test_data['bert_vec'] = df_test['bert_vec']

    print(test_data.head())

    # df_result = pd.concat([df_train, df_test])
    # print(f"the predict result all data is: {df_result.shape} !!!")
    # df_result = df_result.dropna().drop_duplicates(subset=['text'])
    # print(df_result.head())
    # print(f"the predict result is: {df_result.shape} !!!")

    df_all = pd.concat([train_data, test_data])
    print(f"the all train data shape is: {df_all.shape} !!!")

    # df_all = df_all.merge(df_result, on='text', how='left')
    df_all.to_csv('bert_predict_result_new1', index=False, sep='\t')

    # print("# evaluate test_set:",model.evaluate(ds_test_encoded))
 
    df_test['bert_vec'] = df_test['bert_vec'].apply(lambda x: np.argmax(str(x).split(',')))
    print(classification_report(df_test['real_label'].tolist(), df_test['bert_vec'].tolist(),  digits=3))
