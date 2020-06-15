import pandas as pd
from ast import literal_eval

doc_set = pd.read_excel("./doc_set_final_version2.xlsx")
doc_set.head()

doc_set['token'] = doc_set.token.apply(literal_eval)
doc_set['token_len'] = doc_set.token.apply(lambda x: len(x))
doc_set[['token','token_len']]

import matplotlib.pyplot as plt
import numpy as np

np.percentile(doc_set['token_len'], 20)
np.percentile(doc_set['token_len'] , 10)
plt.hist(doc_set['token_len'], bins=50)
plt.title('token len distribution')
plt.show()

doc_set2 = doc_set.drop(doc_set[doc_set['token_len'] < 70].index)
small_class = pd.DataFrame(doc_set2.new_small_class.value_counts())
small_class.to_excel("small_class_refine.xlsx")

np.percentile(small_class['new_small_class'], 25)
np.percentile(small_class['new_small_class'] , 15)
plt.hist(small_class['new_small_class'], bins=50)
plt.title('class len distribution')
plt.show()

small_class_list = small_class[small_class['new_small_class'] < 16].index.tolist()
doc_set3 = doc_set2.drop(doc_set2[doc_set2.new_small_class.isin(small_class_list)].index)
doc_set3.new_class.value_counts()
doc_set3.to_excel("./doc_set_final_version2.xlsx")
