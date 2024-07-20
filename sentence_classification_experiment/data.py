import os
import sys
import pandas as pd 



# path_root = os.getcwd()
# cate_list = ['市场监管', '金融行业', '社会民生', '教育行业', '食品安全', '社会治安', '刑事事件', '医疗卫生']
# # df_all = None
# # for i in cate_list:
# #     file_name = os.listdir(os.path.join(path_root, i))[:4]
# #     for j in file_name:
# #         file_path = os.path.join(path_root, i, j)
# #         df_tmp = pd.read_excel(file_path)[['摘要']]
# #         df_tmp['cate'] = i 
# #         df_all = pd.concat([df_all, df_tmp])
# # print(df_all.shape)
# # df_all.to_csv('text_all', index=False, sep='\t')
# print(dict(zip(cate_list, [i for i in range(len(cate_list))])))

import pandas as pd 
pd.read_json('data').to_csv('fengkong_word', index=False, header=None)