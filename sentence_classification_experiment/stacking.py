import numpy as np 
import pandas as pd 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    classification_report
import lightgbm as lgb
import xgboost as xgb
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer


#引入停用词
infile = open("stop_word.txt",encoding='utf-8')
stopwords_lst = infile.readlines()
stopwords = set([x.strip() for x in stopwords_lst])

# textcnn预测结果特征
textcnn_df = pd.read_csv('textcnn_predict_result', delimiter='\t')[['query', 'label', 'flag', 'textcnn_vec']]
textcnn_df['text'] = textcnn_df['query'].apply(lambda x: ' '.join([i for i in jieba.lcut(str(x)[:64]) if i not in stopwords]))
bert_df = pd.read_csv('bert_predict_result_new', delimiter='\t')[['query', 'bert_vec', 'flag']]
bert_test = pd.read_csv('bert_test', delimiter='\t')
textcnn_df['bert_vec'] = bert_df.query('flag == "train"')['bert_vec'].tolist() + bert_test['bert_vec'].tolist() 

print(f"the data shape is: {textcnn_df.shape} !!!")
print(textcnn_df.head())

textcnn_train_feature = textcnn_df.query('flag == "train"')
textcnn_test_feature = textcnn_df.query('flag == "test"')
print(f"the train data is : {textcnn_train_feature.shape}, and the test data is: {textcnn_test_feature.shape}")


def get_cnt(data):
    words = pseg.cut(data) 
    person_cnt = 0
    loc_cnt = 0
    time_cnt = 0
    other_cnt = 0 
    sport_cnt = 0
    for word, flag in words:
        if flag in ['PER', 'nr']:
            person_cnt += 1
        elif flag in ['LOC', 'ORG', 'ns', 'nt']:
            loc_cnt += 1
        elif flag in ['t', 'TIME']:
            time_cnt += 1
        elif flag in ['n', 'f']:
            other_cnt += 1
        elif flag in ['v', 'vn']:
            sport_cnt += 1
    return ','.join([str(i) for i in [person_cnt,loc_cnt,time_cnt,other_cnt,sport_cnt]])

textcnn_df['cnt_feature'] = textcnn_df['query'].apply(lambda x: get_cnt(str(x)))

cate_list = ['市场监管', '金融行业', '社会民生', '教育行业', '食品安全', '社会治安', '刑事事件', '医疗卫生']
cate_dict = dict(zip(cate_list, [i for i in range(len(cate_list))]))

def get_label(data):
    """
        统计相关特征
    """
    sub_str = re.sub(u"([^\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",data['query'])
    if (len(sub_str) / len(data['query']) >= 0.5) or re.search(r"民生", data['query']):
        return cate_dict['社会民生']
    if re.search(r"金融|股票|交易|财报|股价", data['query']): 
        return cate_dict['金融行业']
    elif re.search(r"食品|食物", data['query']): 
        return cate_dict['食品安全']
    elif re.search(r"教育|学校|学生|校园", data['query']):
        return cate_dict['教育行业']
    elif re.search(r"刑事|案件|犯罪|判刑", data['query']):
        return cate_dict['刑事事件']
    elif re.search(r"医院|医疗|医生", data['query']): 
        return cate_dict['医疗卫生']
    else:
        return data['label']

textcnn_df['query'] = textcnn_df['query'].astype('str')
textcnn_df['label'] = textcnn_df.apply(lambda x: get_label(x), axis=1)
def get_other_feature(data):
    fince_flag = 0
    food_flag = 0
    endutry_flag = 0
    xishi_flag = 0
    host_flag = 0
    social_falg = 0

    sub_str = re.sub(u"([^\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",data['query'])
    ratio = round(len(sub_str) / len(data['query']), 2)
    if ratio >= 0.5 or re.search(r"民生", data['query']):
        social_falg = 1
    if (re.search(r"金融|股票|交易|财报|股价", data['query'])) and (social_falg == 0):
        fince_flag = 1
    elif re.search(r"食品|食物", data['query']) and (social_falg == 0) and (fince_flag == 0):
        food_flag = 1
    elif (re.search(r"教育|学校|学生|校园", data['query'])) \
        and (social_falg == 0) and (fince_flag == 0) and (food_flag == 0):
        endutry_flag = 1
    elif (re.search(r"刑事|案件|犯罪|判刑", data['query'])) \
        and (social_falg == 0) and (fince_flag == 0) and (food_flag == 0) and (endutry_flag == 0):
        xishi_flag = 1
    elif (re.search(r"医院|医疗|医生", data['query']))  and \
        (social_falg == 0) and (fince_flag == 0) and (food_flag == 0) and (endutry_flag == 0) and (xishi_flag == 0):
        host_flag = 1
    return ','.join([str(i) for i in [social_falg,fince_flag,food_flag,endutry_flag,xishi_flag,host_flag,len(data['query']),ratio]])
textcnn_df['other_feature'] = textcnn_df.apply(lambda x: get_other_feature(x), axis=1)
# textcnn_df = textcnn_df.dropna() 
textcnn_df = textcnn_df.fillna(method='ffill')

# 各个特征
tv = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
tv.fit(textcnn_df['text'])
text_feature = tv.transform(textcnn_df['text']).toarray()


split_index = textcnn_df.query('flag == "train"').shape[0]

y_train = textcnn_df['label'].tolist()[:split_index]
y_test = textcnn_df['label'].tolist()[split_index:]

bert_vec = np.array(textcnn_df['bert_vec'].apply(lambda x: [round(abs(float(i)), 3) for i in str(x).split(',')[:-2]]).tolist())
# bert_vec_train_x =  bert_vec[:split_index]
# bert_vec_test_x =  bert_vec[split_index:]
textcnn_vec = np.array(textcnn_df['textcnn_vec'].apply(lambda x: [round(abs(float(i)), 3) for i in str(x).split(',')[:-2]]).tolist())
# textcnn_vec_train_x =  textcnn_vec[:split_index]
# textcnn_vec_test_x =  textcnn_vec[split_index:]

cnt_feature_vec = np.array(textcnn_df['cnt_feature'].apply(lambda x: [int(i) for i in str(x).split(',')]).tolist())
other_feature_vec = np.array(textcnn_df['other_feature'].apply(lambda x: [float(i) for i in str(x).split(',')]).tolist())

bert_all_vec = np.concatenate((text_feature, bert_vec, cnt_feature_vec, other_feature_vec), axis=1)
bert_all_vec_train_x =  bert_all_vec[:split_index]
bert_all_vec_test_x =  bert_all_vec[split_index:]

textcnn_all_vec = np.concatenate((text_feature, textcnn_vec, cnt_feature_vec, other_feature_vec), axis=1)
textcnn_all_vec_train_x =  textcnn_all_vec[:split_index]
textcnn_all_vec_test_x =  textcnn_all_vec[split_index:]


model = LGBMClassifier(
    boosting_type='gbdt',  # 基学习器 gbdt:传统的梯度提升决策树; dart:Dropouts多重加性回归树
    n_estimators=100,  # 迭代次数
    learning_rate=0.05,  # 步长
    max_depth=4,  # 树的最大深度
    min_child_weight=1,  # 决定最小叶子节点样本权重和
    # min_split_gain=0.1,  # 在树的叶节点上进行进一步分区所需的最小损失减少
    subsample=1,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
    colsample_bytree=1,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
    random_state=27,  # 指定随机种子，为了复现结果
    importance_type='gain',  # 特征重要性的计算方式，split:分隔的总数; gain:总信息增益
    objective='multiclass',
)

print("the bert model result is: !!!!")
model.fit(bert_all_vec_train_x, y_train, eval_metric="multi_logloss", \
                          eval_set=[(bert_all_vec_train_x, y_train), (bert_all_vec_test_x, y_test)], \
                         )
print(classification_report(model.predict(bert_all_vec_test_x), y_test, digits=4))


model = LGBMClassifier(
    boosting_type='gbdt',  # 基学习器 gbdt:传统的梯度提升决策树; dart:Dropouts多重加性回归树
    n_estimators=100,  # 迭代次数
    learning_rate=0.05,  # 步长
    max_depth=4,  # 树的最大深度
    min_child_weight=1,  # 决定最小叶子节点样本权重和
    # min_split_gain=0.1,  # 在树的叶节点上进行进一步分区所需的最小损失减少
    subsample=1,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
    colsample_bytree=1,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
    random_state=27,  # 指定随机种子，为了复现结果
    importance_type='gain',  # 特征重要性的计算方式，split:分隔的总数; gain:总信息增益
    objective='multiclass',
)

print("the textcnn model result is: !!!!")
model.fit(textcnn_all_vec_train_x, y_train, eval_metric="multi_logloss", \
                          eval_set=[(textcnn_all_vec_train_x, y_train), (textcnn_all_vec_test_x, y_test)], \
                         )
print(classification_report(model.predict(textcnn_all_vec_test_x), y_test, digits=4))