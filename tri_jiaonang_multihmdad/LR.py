import numpy as np
from sklearn import linear_model
a1=2
a2=4
def LR(train_x, train_y,test_x):
    regr = linear_model.LogisticRegression()
    train = np.concatenate((train_x, train_y), axis=1)
    data1 = train[train[:, train.shape[1] - 1] == 1]
    data0 = train[train[:, train.shape[1] - 1] == 0]
    step_train = data0.shape[0] // data1.shape[0] + 1
    for j in range(step_train):
        if j == step_train - 1:
            train_batch_x = np.vstack((data1[:,0:a1],data0[j * data1.shape[0]:data0.shape[0], 0:a1]))
            train_batch_y = np.vstack((data1[:,a1:a2],data0[j * data1.shape[0]:data0.shape[0], a1:a2]))
        if j != step_train - 1:
            train_batch_x = np.vstack((data1[:, 0:a1], data0[j * data1.shape[0]:(j+1)*data1.shape[0], 0:a1]))
            train_batch_y = np.vstack((data1[:, a1:a2], data0[j * data1.shape[0]:(j+1)*data1.shape[0], a1:a2]))
        train_batch_y=np.reshape(train_batch_y[:,1],(train_batch_x.shape[0],1))
        regr.fit(train_batch_x, train_batch_y)
    pre = regr.predict_proba(test_x)
    pre = pre[:, 1:2]
    return pre