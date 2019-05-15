import pandas as pd
import jieba
import numpy as np
import re
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import heapq

mutual_information = False   # Please set this to True when you want to do mutual information

stopWordsFile = (r'stop words.txt')
stopwords = open(stopWordsFile,'rb')
stops = stopwords.read()
stops=str(stops, encoding = "gbk")
stopList = stops.splitlines()
stopwords.close()

df = pd.read_csv(r"10000 tweets after 7 class labeling.csv", encoding='UTF-8').astype(str)
# How to choose file is depends on which data set you want

def chinese_word_cut(mytext):
    newtext = ""
    for i in jieba.cut(mytext):
        if re.match('^[\u4e00-\u9fa5]{0,}$', i):
            newtext = newtext+i
        elif re.match('^[0-9]*$',i):
            pass
        elif re.match('^\w+$',i):
            pass
        else:
            pass
    result = " ".join(jieba.cut(newtext))
    return result


df["content_cutted"] = df.post_plaintext.apply(chinese_word_cut)

CountVectorizer = CountVectorizer(strip_accents = 'unicode',
                                analyzer='word',
                                stop_words= stopList,
                                max_df=0.15,  # previous parameter 0.2    #0.15
                                min_df= 30)   # previous parameter 10(27)   #30


tf = CountVectorizer.fit_transform(df.content_cutted.values.astype('U')).toarray()

featureArray = np.array([CountVectorizer.get_feature_names()])
featureArray_full = np.column_stack((featureArray,['class']))
# for binary classification(Health or Not Health), please use "is_health_related?"

train = np.column_stack((tf,df['class']))
# for binary classification(Health or Not Health), please use "is_health_related?"

train = np.row_stack((featureArray_full,train))


np.savetxt('train.csv',train, fmt="%s",delimiter = ',',encoding='UTF-8')


if mutual_information == True:
    y = pd.read_csv(r"train.csv", encoding='UTF-8')
    y = y[['class']]
    res = dict(zip(CountVectorizer.get_feature_names(),mutual_info_classif(tf,y,discrete_features=True)))
    nlargestList = heapq.nlargest(800, res.keys())
    nlargestList_value=[]
    for key in nlargestList:
        nlargestList_value = nlargestList_value+[key]
    keepList = pd.read_csv(r"train.csv", encoding='UTF-8')
    f_MI = keepList[nlargestList_value]
    f_MI.to_csv("train_after_MI.csv", index=False)



