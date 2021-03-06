#coding: utf-8

#util
from read_conf import config
from read_file import data,gen_submission,topic,save_pred
from optparse import OptionParser
import sys

#numpy
import numpy as np
from scipy import sparse

#sklearn
#dim reduction
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import PCA

#preprocessing
from sklearn import preprocessing

#feature selection
from sklearn.linear_model import RandomizedLogisticRegression as RLR

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

#classfier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.linear_model import LinearRegression as LReg

#cv
from sklearn import cross_validation


def get_tfidf(train,test,LSA=False):
    if LSA==False:
        vectorizer = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
        length_train = len(train)
        x_all = train + test
        x_all = vectorizer.fit_transform(x_all)
        x = x_all[:length_train]
        t = x_all[length_train:]

    else:
        vectorizer = TfidfVectorizer(max_features=300000,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
        length_train = len(train)
        x_all = train + test
        x_all = vectorizer.fit_transform(x_all)
        x = x_all[:length_train]
        t = x_all[length_train:]

        print "LSA"
        lsa = SVD(n_components=600,algorithm="arpack")
        lsa.fit(x)
        x = lsa.transform(x)
        t = lsa.transform(t)

    return x,t

def get_textf(train,test):
    vectorizer = TfidfVectorizer(max_features=300000,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    length_train = len(train)
    x_all = train + test
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    print "LSA"
    lsa = SVD(n_components=600,algorithm="arpack")
    lsa.fit(x)
    x1 = lsa.transform(x)
    t1 = lsa.transform(t)

    return x,t,x1,t1

    
def get_lsa(x,t):
    print "LSA"
    lsa = SVD(n_components=600,algorithm="arpack")
    lsa.fit(x)
    x = lsa.transform(x)
    t = lsa.transform(t)
    return x,t
    
def feature_selection(train,test,y):
    print "特征选择"
    clf = RLR(C=10,scaling=0.5,sample_fraction=0.6,n_resampling=200,selection_threshold=0.4,n_jobs=3)
    clf.fit(train,y)
    train = clf.transform(train)
    test = clf.transform(test)

    return train,test
    
def train_model(train,test,y,options,train_topic,test_topic,train_nontext,test_nontext):
    model_name = options.model
    if options.nontext==True:
        train = np.array(train_nontext)
        test = np.array(test_nontext)

        scaler = preprocessing.StandardScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        
    elif model_name != "LDA":
        print "tf-idf"    
        train,test = get_tfidf(train,test,options.LSA)
        print "tf-idf后数据维度"
        print "train:",train.shape
        print "test:",test.shape

    else:
        train_topic = np.array(train_topic)
        test_topic = np.array(test_topic)
        #train,test = get_tfidf(train,test,False)#options.LSA)
        #train = sparse.hstack((train,train_topic)).tocsr()
        #test = sparse.hstack((test,test_topic)).tocsr()
        train = train_topic
        test = test_topic
        print "train:",train.shape
        print "test:",test.shape

    y = np.array(y)

    if options.fs == True:
        train,test = feature_selection(train,test,y)
        
        print "特征选择后数据维度"
        print "train:",train.shape
        print "test:",test.shape
        

    clf = get_clf(model_name)
        
    print "交叉验证"
    cv = cross_validation.cross_val_score(clf,train,y,cv=3,scoring='roc_auc',n_jobs=3)
    print "cv score:",cv
    print "cv mean:",np.mean(cv)

 
    print "训练并预测"
    clf.fit(train,y)
    result = clf.predict_proba(test)[:,1]
    if model_name == "GBDT":
        print clf.feature_importances_ 
    
    return result

def get_clf(model_name):
    #选择模型
    clf = " "
    print "你选择的模型是",model_name
    if model_name == "LR" :
        clf =  LogisticRegression(penalty='l2',dual=True,fit_intercept=True,C=2,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=0.1)
    elif model_name == "NB":
        clf = NB()
    elif model_name == "KNN":
        clf = KNN(n_neighbors=40,weights='distance')
    elif model_name == "RF":
        clf = RF(n_estimators=1500,max_features="sqrt",max_depth=8,min_samples_split=10,min_samples_leaf=2)
        #clf = RF(n_estimators=1500,max_features="sqrt")
    elif model_name == "GBDT":
        clf = GBDT(n_estimators=800,max_features="sqrt",max_depth=8,min_samples_split=10,min_samples_leaf=2)
    elif model_name == "LDA":
        clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=True,C=1,tol=1e-8,class_weight=None, random_state=None, intercept_scaling=0.1)
    elif model_name == "LReg":
        clf = LReg()
    else:
        print "你只能从LR,NB,RF几种模型里选择"
        sys.exit(1)    

    return clf

def get_pred(clf,train,test,y):
    clf.fit(train,y)
    x = clf.predict_proba(train)[:,1]
    t = clf.predict_proba(test)[:,1]
    return x,t

    
def combine_model(dp,train,test,y,label,train_topic,test_topic,train_nontext,test_nontext):
    print "模型融合"

    print "计算tf-idf"
    train_raw,test_raw = get_tfidf(train,test,LSA=False)
    print "tfidf数据维度"
    print "train ",train_raw.shape
    print "test ",test_raw.shape

    print "计算LR + tfidf"
    clf = get_clf("LR")
    pred_train,pred_test = get_pred(clf,train_raw,test_raw,y)
    pred_train,pred_test = [pred_train],[pred_test]

    print "计算LDA"
    clf = get_clf("LDA")
    pred_train1,pred_test1 = get_pred(clf,train_topic,test_topic,y)
    pred_train.append(pred_train1)
    pred_test.append(pred_test1)

    print "计算LSA"
    train_lsa,test_lsa = get_tfidf(train,test,LSA=True)
    
    print "计算LR+LSA"
    clf = get_clf("LR")
    pred_train1,pred_test1 = get_pred(clf,train_lsa,test_lsa,y)
    pred_train.append(pred_train1)
    pred_test.append(pred_test1)

    print "计算RF+LSA"
    clf = get_clf("RF")
    pred_train1,pred_test1 = get_pred(clf,train_lsa,test_lsa,y)
    pred_train.append(pred_train1)
    pred_test.append(pred_test1)
    
    print "计算RF+nontext"
    clf = get_clf("RF")
    pred_train1,pred_test1 = get_pred(clf,train_nontext,test_nontext,y)
    pred_train.append(pred_train1)
    pred_test.append(pred_test1)
    
    print "计算GBDT+nontext"
    clf = get_clf("GBDT")
    pred_train1,pred_test1 = get_pred(clf,train_nontext,test_nontext,y)
    pred_train.append(pred_train1)
    pred_test.append(pred_test1)

    print "所有模型计算完成"
    
    pred_train = np.array(pred_train).T
    pred_test = np.array(pred_test).T
    print "pred train",pred_train.shape
    print "pred test",pred_test.shape

    print "保存"
    save_pred(dp,pred_train,pred_test)
    
    
    
    
def main():
    parser = OptionParser()  
    parser.add_option("-m", "--model", dest="model",  \
                      help=u"选择模型:可选择的有LR，RF，NB", metavar="your_model",default="LR")
    parser.add_option("-t","--tokenize",dest="tokenize",action="store_true",\
                      help=u"选择是否进行tokenize，tokenize会得到稍微高一点的准确率，但是效率会慢很多,默认是true",\
                      metavar="your_tokenize",default=False)

    parser.add_option("-n","--nontext",dest="nontext",action="store_true",\
                      help=u"选择是否利用非文本特征，默认是false",default=False)

    parser.add_option("-l","--LSA",dest="LSA",action="store_true",\
                      help=u"选择是否LSA，注意当选用非LR模型的时候，LSA是必须默认开着的，这个在后来我会强制一下逻辑，现在没写",\
                      default=False)
    parser.add_option("-s","--fselect",dest="fs",action="store_true",
                      help=u"选择是否进行特征选择，默认是否，加上-s后会进行选择",default=False)
    parser.add_option("-p","--topic",dest="topic",action="store_true",\
                      help=u"选择是否读取主题分布，默认是否，加上-tp后会进行读取",default=False)

    parser.add_option("-c","--combine",dest="combine",action="store_true",\
                      help=u"选择是否进行模型融合，默认是否",default=False)
    (options, args) = parser.parse_args()
    print options
    
    #读入配置文件
    dp = config("../conf/dp.conf")
    #读入数据
    print "读取数据集"
    train,test,y,label,train_nontext,test_nontext = data(dp,options.tokenize)
    print "train 大小",len(train)
    print "test 大小",len(test)

    print "读取主题"
    total_topic = topic(dp)
    train_topic = total_topic[:len(train)]
    test_topic = total_topic[len(train):]
    print "train 大小",len(train_topic)
    print "test 大小",len(test_topic)

    if options.combine==False:
        result = train_model(train,test,y,options,train_topic,test_topic,train_nontext,test_nontext)
        print "产生结果"
        gen_submission(dp,result,label)

    else:
        combine_model(dp,train,test,y,label,train_topic,test_topic,train_nontext,test_nontext)

def test():
    print "for test"
    parser = OptionParser()  
    parser.add_option("-m", "--model", dest="model",  \
                      help=u"选择模型:可选择的有LR，RF，NB", metavar="your_model",default="LR")

  
    (options, args) = parser.parse_args()
    
if __name__ == '__main__':
    main()
    #test()

    
