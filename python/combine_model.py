#coding: utf-8
'''
这个文件的作用是融合模型
'''

#util
from read_conf import config
from read_file import load_pred,data,gen_submission
from optparse import OptionParser
import sys

#numpy
import numpy as np

#math
from math import log,exp
ll = lambda x : exp((x*log(x))+(1-x)*log(1-x))

from train_model import get_clf

def select_one(train,test,which):
    #其实这个函数的作用更像是测试
    result = test[:,which]
    return result

def average(train,test):
    #完全平均
    result = []
    for i in test:
        result.append(i.mean())

    return result

def score_average(train,test):
    score = np.array([0.88275,0.87831,0.88367,0.87735,0.74347,0.75824])
    result = []
    for i in test:
        temp = (i*score).sum() / (score.sum())
        result.append(temp)
    return result

def select_two(train,test):
    result = []
    for i in test:
        result.append((i[0]+i[2])/2)

    return result

def select_three(train,test):
    result = []
    for i in test:
        result.append((i[0]+i[2]+i[1])/3)

    return result

def select_four(train,test):
    result = []
    for i in test:
        result.append((i[0]+i[2]+i[1]+i[3])/3)

    return result    

def avg_art(train,test):
    #手工设定权值，这个需要调一下
    te,nte = 6,1
    score = np.array([te,te,te,te,nte,nte])
    result = []
    for i in test:
        temp = (i*score).sum() / (score.sum())
        result.append(temp)
    return result

def sub_score_average(train,test):
    score = np.array([0.88275,0.87831,0.88367,0.87735,0.74347,0.75824])-0.74
    result = []
    for i in test:
        temp = (i*score).sum() / (score.sum())
        result.append(temp)
    return result

def bma(train,test):
    score = np.array([0.88275,0.87831,0.88367,0.87735,0.74347,0.75824])
    score = [ll(i) for i in score]
    score = np.array(score)
    score = score / score.sum()
    result = []
    test = test[:,0:4]
    score = score[0:4]
    print score
    for i in test:
        temp = (i*score).sum() / (score.sum())
        result.append(temp)
    return result

def ensemble(train,test,y,model_name):
    train = train[:,0:4]
    test = test[:,0:4]
    clf = get_clf(model_name)
    clf.fit(train,y)

    if model_name == "LReg":
        print clf.coef_
        result = clf.predict(test)
    else :
        result = clf.predict_proba(test)[:,1]
    return result

def main(options,conf):
    print "读取数据集"
    train,test,y,label,train_nontext,test_nontext = data(dp,False)
    train,test = load_pred(dp)
    train,test = np.array(train),np.array(test)
    print "融合训练大小",train.shape
    print "融合测试大小",test.shape

    if options.stra == "select_one":
        result = select_one(train,test,5)

    elif options.stra == "average":
        result = average(train,test)

    elif options.stra == "score_avg":
        result = score_average(train,test)

    elif options.stra == "select_2":
        result = select_two(train,test)
        
    elif options.stra == "select_3":
        result = select_three(train,test)

    elif options.stra == "select_4":
        result = select_four(train,test)

    elif options.stra == "avg_art":
        result = avg_art(train,test)

    elif options.stra == "sub_score":
        result = sub_score_average(train,test)

    elif options.stra == "bma":
        result = bma(train,test)

    elif options.stra == "RF" or options.stra=="LR"\
         or options.stra == "GBDT"\
         or options.stra == "LReg"\
         or options.stra == "NB":
        
        result = ensemble(train,test,y,options.stra)
    
    else :
        print "没有这个更新策略"
        sys.exit(1)

    gen_submission(conf,result,label)

if __name__ == '__main__':
    dp = config("../conf/dp.conf")
    parser = OptionParser()
    parser.add_option("-s","--strategy",dest="stra",
                      help = u"更新策略",default="select_one")

    (options,args) = parser.parse_args()
    print options

    main(options,dp)
