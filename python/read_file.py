#coding: utf-8

import csv
from nlp import nlp

mnlp = nlp()

def data(conf,tokenize=True):
    #读入路径
    train_dir = conf["raw_train"]
    test_dir = conf["raw_test"]

    #数据
    train,test,y,label = [],[],[],[]

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
            train.append(line[2].lower())
        y.append(int(line[-1]))
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
        if tokenize==True:
            test.append(mnlp.token(line[2]))
        else:
            test.append(line[2].lower())    
        label.append(line[1])
        a += 1
        if a%100==0:
            print a
    f.close()
    return train,test,y,label

def gen_submission(conf,result,label):
    f = open(conf["result"],"w")
    writer = csv.writer(f)

    writer.writerow(["urlid","label"])
    assert(len(result)==len(label))

    for i in range(len(result)):
        writer.writerow([label[i],result[i]])
