import numpy as np
def sort_index(realvalue):
    x=realvalue.shape[0]
    id1=np.zeros((x,1))
    for i in range(x):
        id1[i][0]=i
    realvalue=realvalue.reshape((x,1))
    c = np.hstack((realvalue, id1))
    for i in range(x):
        for j in range(x-1):
            if c[j][0]>c[j+1][0]:
                temp=c[j].copy()
                c[j]=c[j+1].copy()
                c[j+1]=temp.copy()
    return c
def max(f):
    id=0
    value=f[0]
    for i in range(f.shape[0]):
        if f[i]>value:
            value=f[i]
            id=i
    return value,id
def newmetric(truevalue,predvalue):
    pos_num=np.sum(truevalue==1)
    neg_num=np.sum(truevalue==0)
    m =truevalue.shape[0]
    Index = sort_index(predvalue)
    truevalue1=np.zeros((truevalue.shape[0],1))
    for i in range(Index.shape[0]):
        truevalue1[i] = truevalue[int(Index[i][1])]
    x = np.zeros((m + 1, 1))
    y = np.zeros((m + 1, 1))
    p = np.zeros((m + 1, 1))
    f = np.zeros((m + 1, 1))
    auc=0
    aup=0
    x[0] = 1
    y[0] = 1
    p[0] = 1
    TP = np.sum(truevalue1[1:m+1] == 1)
    FP = np.sum(truevalue1[1:m+1] == 0)
    for i in range(1,m+1):
        if i !=1:
            if truevalue1[i - 1][0] == 1:
                TP = TP - 1
            else:
                FP = FP - 1
        if neg_num == 0:
            x[i][0] = 0
        else:
            x[i][0] = FP / neg_num
        if pos_num == 0:
            y[i][0] = 0
        else:
            y[i][0]= TP / pos_num
        if TP + FP == 0:
            p[i][0] = 0
        else:
            p[i][0] = TP / (TP + FP)
        a=x[i - 1] - x[i]
        b=y[i] + y[i - 1]
        c=y[i - 1] - y[i]
        auc = auc + (y[i][0] + y[i - 1][0]) * (x[i - 1][0] - x[i][0]) / 2
        aup = aup + (p[i][0] + p[i - 1][0]) * (y[i - 1][0] - y[i][0]) / 2
    auc = auc + y[m-1][0] * x[m-1][0] / 2
    aup = aup + p[m-1][0] * y[m-1][0] / 2
    for i in range(m):
        if p[i][0] + y[i][0] == 0:
            f[i][0] = 0
        else:
            f[i][0] = (2 * p[i][0] * y[i][0]) / (p[i][0]+ y[i][0])
    f[0][0] = 0
    p[0][0] = 0
    y[0][0] = 0
    [fmeasure, index] = max(f)
    precision = p[index][0]
    recall = y[index][0]
    x = x[1:2263].T
    y = y[1:2263].T
    p = p[1:2263].T
    # print(auc)
    return auc,aup,precision,recall,fmeasure[0]


