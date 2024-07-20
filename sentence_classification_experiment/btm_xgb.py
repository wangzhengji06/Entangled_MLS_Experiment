import numpy as np 
import pandas as pd 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import bitermplus as btm
import jieba 


cate_list = ['市场监管', '金融行业', '社会民生', '教育行业', '食品安全', '社会治安', '刑事事件', '医疗卫生']
cate_dict = dict(zip(cate_list, [i for i in range(len(cate_list))]))
stop_word_set = set()
with open('stop_word.txt') as files:
    for line in files:
        line = line.strip('\n')
        stop_word_set.add(line)


df = pd.read_csv('text_all', sep='\t').dropna()
df.columns = ['text', 'label']
df['text'] = df['text'].apply(lambda x: str(x)[:32])
df['label'] = df['label'].map(cate_dict)
print(f"the data is : ", df.head())


def process_text(text):
    text = str(text)
    text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", text)
    text = ' '.join([i for i in jieba.lcut(text) if i not in stop_word_set])
    return text

df.text = df.text.apply(process_text)
df['text_len'] = df['text'].apply(lambda x: len(x))
print(f"the data shape is: {df.shape} !!!")
df = df.query('text_len != 0')
print(f"the filter data shape is: {df.shape}!!!")
df = df.reset_index(drop=True)
train_set, test_set = train_test_split(df, 
    stratify=df['label'],
    test_size=0.2, 
    random_state=43)
texts = df['text'].str.strip().tolist()
# 获取评论的词频和向量化
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
tf = np.array(X.sum(axis=0)).ravel()


# 向量化文本
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))
# Generating biterms
biterms = btm.get_biterms(docs_vec)

perplexity_list, coherence_list, model_list = [], [], []

# 对多个主题进行建模，查看不同主题下的一致性和困惑度

# INITIALIZING AND RUNNING MODEL
model = btm.BTM(
    X, vocabulary, seed=12321, T=8, M=20, alpha=50/8, beta=0.01)
model.fit(biterms, iterations=20)
model_list.append(model)
p_zd = model.transform(docs_vec)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 创建XGBoost分类器
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=8, n_estimators=20, min_child_weight=1, subsample=0.9)
train_x = p_zd[train_set.index.tolist(),:]
train_y = train_set['label'].tolist()
test_x = p_zd[test_set.index.tolist(),:]
test_y = test_set['label'].tolist()
xgb_model.fit(train_x, train_y, verbose=2,
              eval_set=[(train_x, train_y),(test_x, test_y)])

# 在测试集上进行预测
y_pred = xgb_model.predict(test_x)

# 输出分类报告
print(classification_report(test_y, y_pred))


train_res = pd.DataFrame(train_x, columns=[f'vec_{i}' for i in range(8)])
train_res['btm_label'] = xgb_model.predict(train_x)


test_res = pd.DataFrame(test_x, columns=[f'vec_{i}' for i in range(8)])
test_res['btm_label'] = xgb_model.predict(test_x)

test_res.to_csv('xgb_btm_test', index=False, sep='\t')
train_res.to_csv('xgb_btm_train', index=False, sep='\t')