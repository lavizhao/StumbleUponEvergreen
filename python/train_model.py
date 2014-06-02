#coding: utf-8

from read_conf import config
from read_file import data,gen_submission
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def get_tfidf(train,test):
    vectorizer = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    length_train = len(train)
    x_all = train + test
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    return x,t

def train_model(train,test,y):
    print "tf-idf"    
    train,test = get_tfidf(train,test)
    y = np.array(y)

    print "tf-idf后数据维度"
    print "train:",train.shape
    print "test:",test.shape

    print "交叉验证"
    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=True,C=2,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=0.1)
    cv = cross_validation.cross_val_score(clf,train,y,cv=3,scoring='roc_auc',n_jobs=3)
    print "cv score:",cv
    print "cv mean:",np.mean(cv)

    print "训练并预测"
    clf.fit(train,y)
    result = clf.predict_proba(test)[:,1]

    return result
    

def main():
    #读入配置文件
    dp = config("../conf/dp.conf")
    #读入数据
    train,test,y,label = data(dp)
    print "train 大小",len(train)
    print "test 大小",len(test)

    result = train_model(train,test,y)
    print "产生结果"
    gen_submission(dp,result,label)

if __name__ == '__main__':
    main()

    
