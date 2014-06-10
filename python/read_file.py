#coding: utf-8

import csv
from optparse import OptionParser
from nlp import nlp
import sys
from read_conf import config

from sklearn.preprocessing import Imputer
import numpy as np

mnlp = nlp()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

def topic(conf):
    print "读取文件主题"
    f = open(conf["tp"])
    reader = csv.reader(f)
    total = []
    for line in reader:
        s = [float(i) for i in line]
        total.append(s)

    print "整个长度",len(total)
    return total
        

def data(conf,tokenize=True):
    #读入路径
    train_dir = conf["raw_train"]
    test_dir = conf["raw_test"]

    #数据
    train,test,y,label = [],[],[],[]
    train_nontext,test_nontext = [],[]

    f = open(train_dir)
    reader = csv.reader(f,delimiter='\t')
    a = 0

    print "read train"
    for line in reader:
        if a == 0:
            a += 1
            continue
        if tokenize==True:
            train.append(mnlp.token(line[2]))
        else:
            train.append(line[2])
        y.append(int(line[-1]))
        line = line[5:]
        line = line[:-1]
        temp = []
        for item in line:
            if item == "?":
                temp.append(np.nan)
            else:
                temp.append(float(item))
        train_nontext.append(temp)
        a += 1
        if a % 100 == 0:
            print a
    f.close()

    a = 0
    print "read test"
    f = open(test_dir)
    reader = csv.reader(f,delimiter='\t')
    for line in reader:
        if a == 0:
            a += 1
            continue
        label.append(line[1])
        if tokenize==True:
            test.append(mnlp.token(line[2]))
        else:
            test.append(line[2])
        line = line[5:]
        temp = []
        for item in line:
            if item == "?":
                temp.append(np.nan)
            else:
                temp.append(float(item))
        test_nontext.append(temp)
        a += 1
        if a%100==0:
            print a
    f.close()
    imp.fit(train_nontext)
    train_nontext = imp.transform(train_nontext)
    test_nontext = imp.transform(test_nontext)
    return train,test,y,label,train_nontext,test_nontext

def gen_submission(conf,result,label):
    f = open(conf["result"],"w")
    writer = csv.writer(f)

    writer.writerow(["urlid","label"])
    assert(len(result)==len(label))

    for i in range(len(result)):
        writer.writerow([label[i],result[i]])

def save_pred(conf,pred_train,pred_test):
    f = open(conf["pred_train"],"w")
    writer = csv.writer(f)
    for line in pred_train:
        writer.writerow(line)

    f.close()

    f = open(conf["pred_test"],"w")
    writer = csv.writer(f)
    for line in pred_train:
        writer.writerow(line)
        
    f.close()
def merge(conf):
    train,test,y,label,fuck,suck = data(conf,tokenize=False)
    print "train",len(train)
    print "test",len(test)
    print "写文件"
    t = open(conf["total"],"w")

    for line in train:
        t.write(line+"\n")
    for line in test:
        t.write(line+"\n")

    print "done"

def sp(conf,tp):
    f = open(conf["topic"])
    tt = []
    for i in f.readlines():
        tt.append(i)

    tt = tt[1:]
    print "训练数据大小%s"%(len(tt))    

    t = open(conf["tp"],"w")
    writer = csv.writer(t)
    
    for line in tt:
        sp = line.split('\t')
        sp = sp[2:len(sp)-1]
        temp = [0 for i in range(tp)]

        for i in range(tp):
            ind = int(sp[i*2])
            temp[ind] = float(sp[i*2+1])
        st = [str(j) for j in temp]
        writer.writerow(st)

def main(options):
    dp = config("../conf/dp.conf")        
    #1 merge
    if options.merge == True:
        print "合并训练测试集"
        merge(dp)
    elif options.split==True:
        print "将得到的数据分开"
        sp(dp,options.tp)
    else:
        print "error 没有这个选项"
        sys.exit(1)
            
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m","--merge",dest="merge",action="store_true",\
                      help=u"合并两个文件的所有文本",default=False)

    parser.add_option("-s","--split",dest="split",action="store_true",\
                      help=u"将已经合并好的文件排好序，分成两个文件",default=False)

    parser.add_option("-t","--topic",dest="tp",action="store",\
                      help=u"文件的主题数目",type="int")
    (options, args) = parser.parse_args()
    #执行判断
    print options
    main(options)
