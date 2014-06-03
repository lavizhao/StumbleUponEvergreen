#coding: utf-8

#util
from read_conf import config
from read_file import data,gen_submission
from optparse import OptionParser
import sys

#numpy
import numpy as np

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD as PCA

def get_tfidf(train,test,LSA=False):
    if LSA==False:
        vectorizer = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
        length_train = len(train)
        x_all = train + test
        x_all = vectorizer.fit_transform(x_all)
        x = x_all[:length_train]
        t = x_all[length_train:]

        return x,t

    else:
        vectorizer = TfidfVectorizer(max_features=300000,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
        length_train = len(train)
        x_all = train + test
        x_all = vectorizer.fit_transform(x_all)
        x = x_all[:length_train]
        t = x_all[length_train:]

        print "LSA"
        pca = PCA(n_components=600,algorithm="arpack")
        pca.fit(x)
        x = pca.transform(x)
        t = pca.transform(t)

        #print "to dense"
        #x = x.toarray()
        #t = t.toarray()
        
        return x,t

def train_model(train,test,y,options):
    model_name = options.model    
    print "tf-idf"    
    train,test = get_tfidf(train,test,options.LSA)
    y = np.array(y)

    print "tf-idf后数据维度"
    print "train:",train.shape
    print "test:",test.shape

    #选择模型
    print "你选择的模型是",model_name
    if model_name == "LR":
        clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=True,C=2,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=0.1)
    else:
        print "你只能从LR,NB,RF几种模型里选择"
        sys.exit(1)
    print "交叉验证"
    cv = cross_validation.cross_val_score(clf,train,y,cv=3,scoring='roc_auc',n_jobs=3)
    print "cv score:",cv
    print "cv mean:",np.mean(cv)

    print "训练并预测"
    clf.fit(train,y)
    result = clf.predict_proba(test)[:,1]

    return result
    

def main():
    parser = OptionParser()  
    parser.add_option("-m", "--model", dest="model",  \
                      help=u"选择模型:可选择的有LR，RF，NB", metavar="your_model",default="LR")
    parser.add_option("-t","--tokenize",dest="tokenize",action="store_true",\
                      help=u"选择是否进行tokenize，tokenize会得到稍微高一点的准确率，但是效率会慢很多,默认是true",\
                      metavar="your_tokenize",default=False)

    parser.add_option("-l","--LSA",dest="LSA",action="store_true",\
                      help=u"选择是否LSA，注意当选用非LR模型的时候，LSA是必须默认开着的，这个在后来我会强制一下逻辑，现在没写",\
                      default=False)
    (options, args) = parser.parse_args()
    print options
    
    #读入配置文件
    dp = config("../conf/dp.conf")
    #读入数据
    train,test,y,label = data(dp,options.tokenize)
    print "train 大小",len(train)
    print "test 大小",len(test)

    result = train_model(train,test,y,options)
    print "产生结果"
    gen_submission(dp,result,label)

def test():
    print "for test"
    parser = OptionParser()  
    parser.add_option("-m", "--model", dest="model",  \
                      help=u"选择模型:可选择的有LR，RF，NB", metavar="your_model",default="LR")

  
    (options, args) = parser.parse_args()
    
if __name__ == '__main__':
    main()
    #test()

    
