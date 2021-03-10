
import numpy as np
import newcap
from newmetric import newmetric
import warnings
import math
from numba import jit
warnings.filterwarnings("ignore")
import math
def fnorm(X1):
    x = X1.shape[0]
    y = X1.shape[1]
    sum=0
    for i in range(x):
        for j in range(y):
            sum=sum+X1[i][j]*X1[i][j]
    return sum

def dchu(X,Y):
    x = X.shape[0]
    y = X.shape[1]
    result=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            result[i][j]=X[i][j]/Y[i][j]
    return result

def sumrow(X):
    x=X.shape[0]
    y=X.shape[1]

    if x==1:
        sum=0
        for j in range(y):
            sum+=X[0][j]
        return sum
    if x>1:
        sum=np.zeros(x)
        for i in range(x):
            for j in range(y):
                sum[i]+=X[i][j]
        return sum


def eye1(X,axis):
    x = X.shape[0]
    y = X.shape[1]
    if axis==1:
        eyecol=np.zeros((y,y))
        for i in range(x):
            eyecol[i][i]=np.round(sum(X[i]),4)
        return eyecol
    if axis==0:
        sumc=sumrow(X)
        eyerow=np.zeros((x,x))
        for i in range(len(sumrow)):
            eyerow[i][i]=np.round(sumc[i],4)
        return eyerow



def sort(realvalue):
    x=realvalue.shape[0]
    y=realvalue.shape[1]
    id=np.zeros((x,y))
    for i in range(x):
        id[i]=i
    c = np.hstack((realvalue, id))
    for i in range(x):
        for j in range(x-1):
            if c[j][0]>c[j+1][0]:
                temp=c[j].copy()
                c[j]=c[j+1].copy()
                c[j+1]=temp.copy()
    return c
def ZeroCompletion(Y,kd,km):
    nd=Y.shape[0]
    nm=Y.shape[1]
    rownumber=0
    for i in range(nd):
        if np.sum(Y[i,:])==0:
            kd[:,i]=0
            rownumber=rownumber+1
    kd=kd-np.diag(np.diag(kd))
    retRow=Y.copy()
    for i in range(nd):
        if np.sum(Y[i,:])==0:
            s=sort(np.reshape(kd[i:i+1,:],(nd,1)))
            indeneigh=int(s[s.shape[0]-1][1])
            retRow[i,:]=Y[indeneigh,:]*kd[i][indeneigh]
    retCol = Y.copy()
    colnumber=0
    for i in range(nm):
        if np.sum(Y[:,i])==0:
            km[:,i]=0
            colnumber=colnumber+1
    km=km-np.diag(np.diag(km))
    for i in range(nm):
        if np.sum(Y[:,i])==0:
            s=sort(np.reshape(km[i:i+1,:],(nm,1)))
            u=s[s.shape[0]-1][0]
            indeneigh=0
            for j in range(s.shape[0]-1):
                if s[s.shape[0]-1-j][0]!=u:
                    break
                if   s[s.shape[0] - 1 - j][0] == u:
                    indeneigh=int(s[s.shape[0]-1-j][1])
            retCol[:,i]=Y[:,indeneigh]*km[i][indeneigh]
    if colnumber>0 and rownumber>0:
        out=retRow.copy()
        for i in range(out.shape[1]):
            if np.sum(out[:,i])==0:
                out[:,i]=retCol[:,i]
    elif colnumber>0:
        out=retCol.copy()
    elif rownumber>0:
        out=retCol.copy()
    else:
        out=Y
    return out
def brwhmda(X,Md1,Mm1,Mm,Md):
    X1 = ZeroCompletion(X, Md1, Mm1)
    sumX = np.sum(np.sum(X1))
    for i in range(nd):
        for j in range(nm):
            X1[i][j] = X1[i][j] / sumX
    Il = 1
    Ir = 3
    alpha = 0.5
    R = X1.copy()
    for i in range(nm):
        sumcoltmp = np.sum(Mm[:, i])
        if sumcoltmp > 0:
            Mm[:, i] = Mm[:, i] / sumcoltmp
    for i in range(nd):
        sumrowtmp = np.sum(Md[:, i])
        if sumrowtmp > 0:
            Md[:, i] = Md[:, i] / sumrowtmp
    for i in range(3):
        rflag = 0
        lflag = 0

        if i < Il:
            lflag = 1
            LR = np.dot((1 - alpha) * Md, R) + alpha * X1

        if i < Ir:
            rflag = 1
            RR = np.dot((1 - alpha) * R, Mm) + alpha * X1
        R = (rflag * RR + lflag * LR) * (1 / (rflag + lflag))
    return R


def comtriple(X,W,L_parameter,D_parameter,U_parameter,V_parameter,k,m,bool):
    print('comptriple')
    nd = interaction.shape[0]
    nm = interaction.shape[1]
    e_parameter = 1
    sd = pow(np.linalg.norm(X, ord=2, keepdims=True, axis=1), 2)
    gamad = np.round(nd / sum(sd) , 4)*0.5
    sm = pow(np.linalg.norm(X, ord=2, keepdims=True, axis=0), 2)
    gamal = np.round(nm / sumrow(sm) , 4)*0.5
    kd = np.zeros((nd, nd))
    km = np.zeros((nm, nm))
    for i in range(nd):
        for j in range(nd):
            kd[i][j] = np.exp(-1 * gamad * pow(np.linalg.norm(X[i] - X[j]), 2))
    for i in range(nm):
        for j in range(nm):
            km[i][j] = np.exp(-1 * gamal * pow(np.linalg.norm(X[:, i] - X[:, j]), 2))

    dd=np.loadtxt('data/disease_features.txt')
    mm=np.loadtxt('data/microbe_features.txt')
    kd=(kd+dd)*0.5
    km=(km+mm)*0.5
    kd = (kd + np.transpose(kd)) * 0.5
    km = (km + np.transpose(km)) * 0.5


    U=np.random.random(size=(nd,k))
    S=np.random.random(size=(k,m))
    V=np.random.random(size=(nm,m))
    a = 1e+6
    D1=eye1(km,axis=1)
    D2 = eye1(kd,axis=1)
    LV=D1-km
    LU=D2-kd
    cost = 0.5 * L_parameter * np.trace(np.dot(np.dot(np.transpose(V), LV), V)) \
           + 0.5 * D_parameter * np.trace(np.dot(np.dot(np.transpose(U), LU), U)) \
           + V_parameter * fnorm(V) + U_parameter * fnorm(U) \
           + e_parameter * fnorm(W*(X-np.dot(np.dot(U, S), np.transpose(V))))
    for i in range(1000):
        U_up=2*e_parameter*np.dot(np.dot(np.multiply(W,X),V),np.transpose(S))+D_parameter*np.dot(kd,U)
        U_down=2*e_parameter*np.dot(np.dot(np.multiply(W,np.dot(np.dot(U,S),np.transpose(V))),V),
                                    np.transpose(S))+D_parameter*np.dot(D2,U)+2*U_parameter*U
        U_down[U_down<1e-10]=1e-10
        U=np.multiply(U,dchu(U_up,U_down))

        S_up=np.dot(np.dot(np.transpose(U),np.multiply(W,X)),V)#+4*S_parameter*S
        S_down=np.dot(np.dot(np.transpose(U),np.multiply(W,np.dot(np.dot(U,S),np.transpose(V)))),V)#+4*S_parameter*np.dot(np.dot(S,np.transpose(S)),S)+2*S1_parameter*S
        S_down[S_down<1e-10]=1e-10
        S=np.multiply(S,dchu(S_up,S_down))

        V_up=2*e_parameter*np.dot(np.dot(np.multiply(np.transpose(W),np.transpose(X)),U),S)+L_parameter*np.dot(km,V)
        V_down=2*e_parameter*np.dot(np.dot(np.multiply(np.transpose(W),np.dot(np.dot(V,np.transpose(S)),
                                    np.transpose(U))),U),S)+L_parameter*np.dot(D1,V)+2*V_parameter*V
        V_down[V_down<1e-10]=1e-10
        V=np.multiply(V,dchu(V_up,V_down))

        cost_pre=cost
        cost =  0.5 * L_parameter * np.trace(np.dot(np.dot(np.transpose(V), LV), V)) \
           + 0.5 * D_parameter * np.trace(np.dot(np.dot(np.transpose(U), LU), U)) \
           + V_parameter * fnorm(V) + U_parameter *fnorm(U) \
           + e_parameter * fnorm(W*(X-np.dot(np.dot(U, S), np.transpose(V))))
        a=abs((cost-cost_pre)/cost)
        if a<10e-6:
            print(i)
            break
    F=np.dot(np.dot(U,S),np.transpose(V))

    Mm = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            Mm[i][j] = 1 / (1 + np.exp(-17 * km[i][j] + np.log(9999)))
    Md1 = kd.copy()
    Mm1 = Mm.copy()
    Md = kd.copy()
    R=brwhmda(X,Md1,Mm1,Mm,Md)

    if (bool == 2):
        for i in range(nd):
            a = np.max(R[i])
            b = np.min(R[i])
            for j in range(nm):
                R[i][j] = (R[i][j] - b) / (a - b)
        for i in range(nd):
            a = np.max(F[i])
            b = np.min(F[i])
            for j in range(nm):
                F[i][j] = (F[i][j] - b) / (a - b)
    else:
        for i in range(nm):
            a = np.max(R[:, i])
            b = np.min(R[:, i])
            for j in range(nd):
                R[j][i] = (R[j][i] - b) / (a - b)
        for i in range(nm):
            a = np.max(F[:, i])
            b = np.min(F[:, i])
            for j in range(nd):
                F[j][i] = (F[j][i] - b) / (a - b)
    return F,R,kd,km

def colum_triple(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,bool):
    nd=interaction.shape[0]
    nm=interaction.shape[1]
    index=[i for i in range(nm)]
    np.random.shuffle(index)
    auc=np.zeros((5,1))
    aup = np.zeros((5,1))
    precition = np.zeros((5,1))
    recall = np.zeros((5,1))
    fmeasure = np.zeros((5,1))
    auc0 = np.zeros((5, 1))
    aup0 = np.zeros((5, 1))
    precition0 = np.zeros((5, 1))
    recall0 = np.zeros((5, 1))
    fmeasure0 = np.zeros((5, 1))
    auc1 = np.zeros((5, 1))
    aup1 = np.zeros((5, 1))
    precition1 = np.zeros((5, 1))
    recall1 = np.zeros((5, 1))
    fmeasure1 = np.zeros((5, 1))
    auc2 = np.zeros((5, 1))
    aup2 = np.zeros((5, 1))
    precition2 = np.zeros((5, 1))
    recall2 = np.zeros((5, 1))
    fmeasure2 = np.zeros((5, 1))
    auc3 = np.zeros((5, 1))
    aup3 = np.zeros((5, 1))
    precition3 = np.zeros((5, 1))
    recall3 = np.zeros((5, 1))
    fmeasure3 = np.zeros((5, 1))
    auc4 = np.zeros((5, 1))
    aup4 = np.zeros((5, 1))
    precition4 = np.zeros((5, 1))
    recall4 = np.zeros((5, 1))
    fmeasure4 = np.zeros((5, 1))
    auc5 = np.zeros((5, 1))
    aup5 = np.zeros((5, 1))
    precition5 = np.zeros((5, 1))
    recall5 = np.zeros((5, 1))
    fmeasure5 = np.zeros((5, 1))
    for i in range(5):
        X = interaction.copy()
        # W=np.ones((nd,nm))
        W = interaction.copy()
        for d in range(nd):
            for g in range(nm):
                if W[d][g]==0:
                    W[d][g]=0.5
        if i!=4:
            test = np.zeros(int(nm / 5))
            for g in range(int(nm/5)):
                test[g]=index[i*int(nm/5)+g]
        if i==4:
            test = np.zeros(nm-4*int(nm / 5))
            for m1 in range(nm-4*int(nm/5)):
                test[m1]=index[i*int(nm/5)+m1]
        for j in range(test.shape[0]):
            for d in range(nd):
                X[d][int(test[j])]=0
                W[d][int(test[j])]=0
        F,R,kd,km=comtriple(X,W,L_parameter,D_parameter,U_parameter,V_parameter,k,m,BOOL)
        captestx1 = []
        captestx2=[]
        captesty=[]
        captrain1x1 = []
        captrain0x1=[]
        captrain1x2 = []
        captrain0x2=[]
        captrain1y = []
        captrain0y = []
        numtest = 0
        numtrain1 = 0
        numtrain0=0
        tmpF = np.hstack((F, np.zeros((nd, 38))))
        tmpR=np.hstack((R,np.zeros((nd,38))))
        tmpX=np.hstack((X,np.zeros(((nd,38)))))
        for b in range(interaction.shape[1]):
            for a in range(interaction.shape[0]):

                if b in test:
                    tmp=np.zeros((4*(nm//nd+1),nd))
                    numf=0
                    numr=0
                    for c in range(4*(nm//nd+1)):
                        if c<2*(nm // nd + 1):
                            a1 = tmpR[:, b].T
                            a1[a] = 0
                            b1 = tmpR[a]
                            b1[b] = 0
                            if c % 2 == 0:
                                tmp[c] = a1
                            else:
                                tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                        elif c<4*(nm // nd + 1):
                            a1 = tmpF[:, b].T
                            a1[a] = 0
                            b1 = tmpF[a]
                            b1[b] = 0
                            if c % 2 == 0:
                                tmp[c] = a1
                            else:
                                tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                numf=numf+1
                    captestx2.append(tmp)
                    # captestx1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                    captesty.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                    # captestx1.append(np.vstack((newF[a+b][0:39],newF[a+b][39:])).T)
                    # captestx2.append(np.hstack((F[a],F[:,b].T)))
                    # captestx2.append(np.hstack((np.hstack((np.hstack((F[a],F[:,b].T)),R[a])),R[:,b].T)))
                    numtest = numtest + 1
                if b not in test:
                    if interaction[a][b]==1:
                        tmp = np.zeros((4 * (nm // nd + 1) , nd))
                        numf = 0
                        numr = 0

                        for c in range(4 * (nm // nd + 1) ):
                            if c < 2 * (nm // nd + 1):
                                a1 = tmpR[:, b].T
                                a1[a] = 0
                                b1 = tmpR[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                            elif c < 4 * (nm // nd + 1):
                                a1 = tmpF[:, b].T
                                a1[a] = 0
                                b1 = tmpF[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                    numf = numf + 1

                        captrain1y.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                        # captrain1x1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                        captrain1x2.append(tmp)
                        numtrain1 = numtrain1 + 1
                    if interaction[a][b]==0:
                        tmp = np.zeros((4 * (nm // nd + 1) , nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1) ):
                            if c < 2 * (nm // nd + 1):
                                a1 = tmpR[:, b].T
                                a1[a] = 0
                                b1 = tmpR[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                            elif c < 4 * (nm // nd + 1):
                                a1 = tmpF[:, b].T
                                a1[a] = 0
                                b1 = tmpF[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                    numf = numf + 1

                        captrain0y.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                        captrain0x2.append(tmp)
                        # captrain0x1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                        # captrain0x2.append(np.hstack((np.hstack((np.hstack((F[a],F[:,b].T)),R[a])),R[:,b].T)))
                        numtrain0 = numtrain0 + 1
        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=IF(captrain[:,0:(captrain.shape[1]-2)],captrain[:,(captrain.shape[1]-2):],captest[:,0:(captest.shape[1]-2)],captest[:,(captest.shape[1]-2):],result)

        captrain0x2=np.array(captrain0x2)
        captrain1x2=np.array(captrain1x2)
        captrain0x1 = np.array(captrain0x1)
        captrain1x1 = np.array(captrain1x1)
        captrain1y = np.array(captrain1y)
        captrain0y = np.array(captrain0y)
        captestx2=np.array(captestx2)
        captesty=np.array(captesty)
        captestx1=np.array(captestx1)
        traindata_tmp_y=np.vstack((captrain0y,captrain1y))
        captrainresult,capresult=newcap.capmain(captrain0x2,captrain0y,captrain1x2,captrain1y,captestx2,captesty)        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=newmetric(captest[:,(captest.shape[1]-1)],result)
        train_tmp_y = np.vstack((captrain0y, captrain1y))
        trainx = np.hstack((train_tmp_y[:, 2:4], captrainresult[:, 1:2]))
        trainy = train_tmp_y[:, 1:2]
        from sklearn import linear_model
        regr = linear_model.LogisticRegression()
        regr.fit(trainx, trainy)
        pre = regr.predict_proba(np.hstack((captesty[:, 2:4], capresult[:, 1:2])))
        pre1 = captesty[:, 1:3]
        pre2 = captesty[:, 2:4]
        pre3 = capresult
        regr = linear_model.LogisticRegression()
        regr.fit(train_tmp_y[:, 2:4], trainy)
        pre4 = regr.predict_proba(captesty[:, 2:4])
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 2:3], captrainresult[:, 1:2])), trainy)
        pre5 = regr.predict_proba(np.hstack((captesty[:, 2:3], capresult[:, 1:2])))
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 3:4], captrainresult[:, 1:2])), trainy)
        pre6 = regr.predict_proba(np.hstack((captesty[:, 3:4], capresult[:, 1:2])))
        if bool == 0:
            for j in range(captesty.shape[0] // nd):
                # result1=sort_index(capresult[i * nd:(i + 1) * nd, 0:1])+sort_index(triresult[i * nd:(i + 1) * nd, 0:1])
                auc_, aup_, precition_, recall_, fmeasure_ = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre[j * nd:(j + 1) * nd, 1:2])
                auc_0, aup_0, precition_0, recall_0, fmeasure_0 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre1[j * nd:(j + 1) * nd, 1:2])
                auc_1, aup_1, precition_1, recall_1, fmeasure_1 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre2[j * nd:(j + 1) * nd, 1:2])
                auc_2, aup_2, precition_2, recall_2, fmeasure_2 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre3[j * nd:(j + 1) * nd, 1:2])
                auc_3, aup_3, precition_3, recall_3, fmeasure_3 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre4[j * nd:(j + 1) * nd, 1:2])
                auc_4, aup_4, precition_4, recall_4, fmeasure_4 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre5[j * nd:(j + 1) * nd, 1:2])
                auc_5, aup_5, precition_5, recall_5, fmeasure_5 = newmetric(
                    captesty[j * nd:(j + 1) * nd, 1:2],
                    pre6[j * nd:(j + 1) * nd, 1:2])
                auc[i] = auc[i] + auc_
                aup[i] = aup[i] + aup_
                precition[i] = precition[i] + precition_
                recall[i] = recall[i] + recall_
                fmeasure[i] = fmeasure[i] + fmeasure_
                auc0[i] = auc0[i] + auc_0
                aup0[i] = aup0[i] + aup_0
                precition0[i] = precition0[i] + precition_0
                recall0[i] = recall0[i] + recall_0
                fmeasure0[i] = fmeasure0[i] + fmeasure_0
                auc1[i] = auc1[i] + auc_1
                aup1[i] = aup1[i] + aup_1
                precition1[i] = precition1[i] + precition_1
                recall1[i] = recall1[i] + recall_1
                fmeasure1[i] = fmeasure1[i] + fmeasure_1
                auc2[i] = auc2[i] + auc_2
                aup2[i] = aup2[i] + aup_2
                precition2[i] = precition2[i] + precition_2
                recall2[i] = recall2[i] + recall_2
                fmeasure2[i] = fmeasure2[i] + fmeasure_2
                auc3[i] = auc3[i] + auc_3
                aup3[i] = aup3[i] + aup_3
                precition3[i] = precition3[i] + precition_3
                recall3[i] = recall3[i] + recall_3
                fmeasure3[i] = fmeasure3[i] + fmeasure_3
                auc4[i] = auc4[i] + auc_4
                aup4[i] = aup4[i] + aup_4
                precition4[i] = precition4[i] + precition_4
                recall4[i] = recall4[i] + recall_4
                fmeasure4[i] = fmeasure4[i] + fmeasure_4
                auc5[i] = auc5[i] + auc_5
                aup5[i] = aup5[i] + aup_5
                precition5[i] = precition5[i] + precition_5
                recall5[i] = recall5[i] + recall_5
                fmeasure5[i] = fmeasure5[i] + fmeasure_5
            auc[i] = auc[i] / (captesty.shape[0] // nd)
            aup[i] = aup[i] / (captesty.shape[0] // nd)
            precition[i] = precition[i] / (captesty.shape[0] // nd)
            recall[i] = recall[i] / (captesty.shape[0] // nd)
            fmeasure[i] = fmeasure[i] / (captesty.shape[0] // nd)
            auc0[i] = auc0[i] / (captesty.shape[0] // nd)
            aup0[i] = aup0[i] / (captesty.shape[0] // nd)
            precition0[i] = precition0[i] / (captesty.shape[0] // nd)
            recall0[i] = recall0[i] / (captesty.shape[0] // nd)
            fmeasure0[i] = fmeasure0[i] / (captesty.shape[0] // nd)
            auc1[i] = auc1[i] / (captesty.shape[0] // nd)
            aup1[i] = aup1[i] / (captesty.shape[0] // nd)
            precition1[i] = precition1[i] / (captesty.shape[0] // nd)
            recall1[i] = recall1[i] / (captesty.shape[0] // nd)
            fmeasure1[i] = fmeasure1[i] / (captesty.shape[0] // nd)
            auc2[i] = auc2[i] / (captesty.shape[0] // nd)
            aup2[i] = aup2[i] / (captesty.shape[0] // nd)
            precition2[i] = precition2[i] / (captesty.shape[0] // nd)
            recall2[i] = recall2[i] / (captesty.shape[0] // nd)
            fmeasure2[i] = fmeasure2[i] / (captesty.shape[0] // nd)
            auc3[i] = auc3[i] / (captesty.shape[0] // nd)
            aup3[i] = aup3[i] / (captesty.shape[0] // nd)
            precition3[i] = precition3[i] / (captesty.shape[0] // nd)
            recall3[i] = recall3[i] / (captesty.shape[0] // nd)
            fmeasure3[i] = fmeasure3[i] / (captesty.shape[0] // nd)
            auc4[i] = auc4[i] / (captesty.shape[0] // nd)
            aup4[i] = aup4[i] / (captesty.shape[0] // nd)
            precition4[i] = precition4[i] / (captesty.shape[0] // nd)
            recall4[i] = recall4[i] / (captesty.shape[0] // nd)
            fmeasure4[i] = fmeasure4[i] / (captesty.shape[0] // nd)
            auc5[i] = auc5[i] / (captesty.shape[0] // nd)
            aup5[i] = aup5[i] / (captesty.shape[0] // nd)
            precition5[i] = precition5[i] / (captesty.shape[0] // nd)
            recall5[i] = recall5[i] / (captesty.shape[0] // nd)
            fmeasure5[i] = fmeasure5[i] / (captesty.shape[0] // nd)
    return np.sum(auc) / 5, np.sum(aup) / 5, np.sum(precition) / 5, np.sum(recall) / 5, np.sum(
            fmeasure) / 5, np.sum(
            auc0) / 5, np.sum(aup0) / 5, np.sum(precition0) / 5, np.sum(recall0) / 5, np.sum(fmeasure0) / 5, np.sum(
            auc1) / 5, \
               np.sum(aup1) / 5, np.sum(precition1) / 5, np.sum(recall1) / 5, np.sum(fmeasure1) / 5, np.sum(
            auc2) / 5, np.sum(aup2) / 5, np.sum(precition2) / 5, np.sum(recall2) / 5, np.sum(fmeasure2) / 5, np.sum(
            auc3) / 5, np.sum(aup3) / 5, np.sum(precition3) / 5, np.sum(recall3) / 5, np.sum(fmeasure3) / 5, np.sum(
            auc4) / 5, np.sum(aup4) / 5, np.sum(precition4) / 5, np.sum(recall4) / 5, np.sum(fmeasure4) / 5, np.sum(
            auc5) / 5, np.sum(aup5) / 5, np.sum(precition5) / 5, np.sum(recall5) / 5, np.sum(fmeasure5) / 5


def row_triple(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,bool):
    nd=interaction.shape[0]
    nm=interaction.shape[1]
    index=[i for i in range(nd)]
    np.random.shuffle(index)
    auc=np.zeros((5,1))
    aup = np.zeros((5,1))
    precition = np.zeros((5,1))
    recall = np.zeros((5,1))
    fmeasure = np.zeros((5,1))
    auc0 = np.zeros((5, 1))
    aup0 = np.zeros((5, 1))
    precition0 = np.zeros((5, 1))
    recall0 = np.zeros((5, 1))
    fmeasure0 = np.zeros((5, 1))
    auc1 = np.zeros((5, 1))
    aup1 = np.zeros((5, 1))
    precition1 = np.zeros((5, 1))
    recall1 = np.zeros((5, 1))
    fmeasure1 = np.zeros((5, 1))
    auc2 = np.zeros((5, 1))
    aup2 = np.zeros((5, 1))
    precition2 = np.zeros((5, 1))
    recall2 = np.zeros((5, 1))
    fmeasure2 = np.zeros((5, 1))
    auc3 = np.zeros((5, 1))
    aup3 = np.zeros((5, 1))
    precition3 = np.zeros((5, 1))
    recall3 = np.zeros((5, 1))
    fmeasure3 = np.zeros((5, 1))
    auc4 = np.zeros((5, 1))
    aup4 = np.zeros((5, 1))
    precition4 = np.zeros((5, 1))
    recall4 = np.zeros((5, 1))
    fmeasure4 = np.zeros((5, 1))
    auc5 = np.zeros((5, 1))
    aup5 = np.zeros((5, 1))
    precition5 = np.zeros((5, 1))
    recall5 = np.zeros((5, 1))
    fmeasure5 = np.zeros((5, 1))
    for i in range(5):
        X = interaction.copy()
        # W=np.ones((nd,nm))
        W = interaction.copy()
        for d in range(nd):
            for g in range(nm):
                if W[d][g]==0:
                    W[d][g]=0.5
        if i!=4:
            test = np.zeros(int(nd / 5))
            for g in range(int(nd/5)):
                test[g]=index[i*int(nd/5)+g]
        if i==4:
            test = np.zeros(nd-4*int(nd / 5))
            for m1 in range(nd-4*int(nd/5)):
                test[m1]=index[i*int(nd/5)+m1]
        for j in range(test.shape[0]):
            for d in range(nm):
                X[int(test[j])][d]=0
                W[int(test[j])][d]=0
        F,R,kd,km=comtriple(X,W,L_parameter,D_parameter,U_parameter,V_parameter,k,m,BOOL)
        captestx1 = []
        captestx2=[]
        captesty=[]
        captrain1x1 = []
        captrain0x1=[]
        captrain1x2 = []
        captrain0x2=[]
        captrain1y = []
        captrain0y = []
        numtest = 0
        numtrain1 = 0
        numtrain0=0
        tmpF = np.hstack((F, np.zeros((nd, 38))))
        tmpR=np.hstack((R,np.zeros((nd,38))))
        tmpX=np.hstack((X,np.zeros(((nd,38)))))
        for a in range(nd):
            for b in range(nm):
                if a in test:
                    tmp=np.zeros((4*(nm//nd+1),nd))
                    numf=0
                    numr=0
                    for c in range(4*(nm//nd+1)):
                        if c<2*(nm // nd + 1):
                            a1 = tmpR[:, b].T
                            a1[a] = 0
                            b1 = tmpR[a]
                            b1[b] = 0
                            if c % 2 == 0:
                                tmp[c] = a1
                            else:
                                tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                        elif c<4*(nm // nd + 1):
                            a1 = tmpF[:, b].T
                            a1[a] = 0
                            b1 = tmpF[a]
                            b1[b] = 0
                            if c % 2 == 0:
                                tmp[c] = a1
                            else:
                                tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                numf=numf+1
                    captestx2.append(tmp)
                    # captestx1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                    captesty.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                    # captestx1.append(np.vstack((newF[a+b][0:39],newF[a+b][39:])).T)
                    # captestx2.append(np.hstack((F[a],F[:,b].T)))
                    # captestx2.append(np.hstack((np.hstack((np.hstack((F[a],F[:,b].T)),R[a])),R[:,b].T)))
                    numtest = numtest + 1
                if a not in test:
                    if interaction[a][b]==1:
                        tmp = np.zeros((4 * (nm // nd + 1) , nd))
                        numf = 0
                        numr = 0

                        for c in range(4 * (nm // nd + 1) ):
                            if c < 2 * (nm // nd + 1):
                                a1 = tmpR[:, b].T
                                a1[a] = 0
                                b1 = tmpR[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                            elif c < 4 * (nm // nd + 1):
                                a1 = tmpF[:, b].T
                                a1[a] = 0
                                b1 = tmpF[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                    numf = numf + 1

                        captrain1y.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                        # captrain1x1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                        captrain1x2.append(tmp)
                        numtrain1 = numtrain1 + 1
                    if interaction[a][b]==0:
                        tmp = np.zeros((4 * (nm // nd + 1) , nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1) ):
                            if c < 2 * (nm // nd + 1):
                                a1 = tmpR[:, b].T
                                a1[a] = 0
                                b1 = tmpR[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                            elif c < 4 * (nm // nd + 1):
                                a1 = tmpF[:, b].T
                                a1[a] = 0
                                b1 = tmpF[a]
                                b1[b] = 0
                                if c % 2 == 0:
                                    tmp[c] = a1
                                else:
                                    tmp[c] = b1[numf * nd:(numf + 1) * nd]
                                    numf = numf + 1

                        captrain0y.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])),F[a][b])),R[a][b])))
                        captrain0x2.append(tmp)
                        # captrain0x1.append(np.hstack((np.hstack((np.hstack((U[a],np.hstack((V[b],np.hstack((F[a],F[:,b].T)))))),a1)),b1)))
                        # captrain0x2.append(np.hstack((np.hstack((np.hstack((F[a],F[:,b].T)),R[a])),R[:,b].T)))
                        numtrain0 = numtrain0 + 1
        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=IF(captrain[:,0:(captrain.shape[1]-2)],captrain[:,(captrain.shape[1]-2):],captest[:,0:(captest.shape[1]-2)],captest[:,(captest.shape[1]-2):],result)

        captrain0x2=np.array(captrain0x2)
        captrain1x2=np.array(captrain1x2)
        captrain0x1 = np.array(captrain0x1)
        captrain1x1 = np.array(captrain1x1)
        captrain1y = np.array(captrain1y)
        captrain0y = np.array(captrain0y)
        captestx2=np.array(captestx2)
        captesty=np.array(captesty)
        captestx1=np.array(captestx1)
        traindata_tmp_y=np.vstack((captrain0y,captrain1y))
        captrainresult,capresult=newcap.capmain(captrain0x2,captrain0y,captrain1x2,captrain1y,captestx2,captesty)        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=newmetric(captest[:,(captest.shape[1]-1)],result)
        train_tmp_y = np.vstack((captrain0y, captrain1y))
        trainx = np.hstack((train_tmp_y[:, 2:4], captrainresult[:, 1:2]))
        trainy = train_tmp_y[:, 1:2]
        from sklearn import linear_model
        regr = linear_model.LogisticRegression()
        regr.fit(trainx, trainy)
        pre = regr.predict_proba(np.hstack((captesty[:, 2:4], capresult[:, 1:2])))
        pre1 = captesty[:, 1:3]
        pre2 = captesty[:, 2:4]
        pre3 = capresult
        regr = linear_model.LogisticRegression()
        regr.fit(train_tmp_y[:, 2:4], trainy)
        pre4 = regr.predict_proba(captesty[:, 2:4])
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 2:3], captrainresult[:, 1:2])), trainy)
        pre5 = regr.predict_proba(np.hstack((captesty[:, 2:3], capresult[:, 1:2])))
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 3:4], captrainresult[:, 1:2])), trainy)
        pre6 = regr.predict_proba(np.hstack((captesty[:, 3:4], capresult[:, 1:2])))
        if bool == 2:
            for j in range(captesty.shape[0] // nm):
                auc_, aup_, precition_, recall_, fmeasure_ = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre[j * nm:(j + 1) * nm, 1:2])
                auc_0, aup_0, precition_0, recall_0, fmeasure_0 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre1[j * nm:(j + 1) * nm, 1:2])
                auc_1, aup_1, precition_1, recall_1, fmeasure_1 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre2[j * nm:(j + 1) * nm, 1:2])
                auc_2, aup_2, precition_2, recall_2, fmeasure_2 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre3[j * nm:(j + 1) * nm, 1:2])
                auc_3, aup_3, precition_3, recall_3, fmeasure_3 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre4[j * nm:(j + 1) * nm, 1:2])
                auc_4, aup_4, precition_4, recall_4, fmeasure_4 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre5[j * nm:(j + 1) * nm, 1:2])
                auc_5, aup_5, precition_5, recall_5, fmeasure_5 = newmetric(
                    captesty[j * nm:(j + 1) * nm, 1:2],
                    pre6[j * nm:(j + 1) * nm, 1:2])
                auc[i] = auc[i] + auc_
                aup[i] = aup[i] + aup_
                precition[i] = precition[i] + precition_
                recall[i] = recall[i] + recall_
                fmeasure[i] = fmeasure[i] + fmeasure_
                auc0[i] = auc0[i] + auc_0
                aup0[i] = aup0[i] + aup_0
                precition0[i] = precition0[i] + precition_0
                recall0[i] = recall0[i] + recall_0
                fmeasure0[i] = fmeasure0[i] + fmeasure_0
                auc1[i] = auc1[i] + auc_1
                aup1[i] = aup1[i] + aup_1
                precition1[i] = precition1[i] + precition_1
                recall1[i] = recall1[i] + recall_1
                fmeasure1[i] = fmeasure1[i] + fmeasure_1
                auc2[i] = auc2[i] + auc_2
                aup2[i] = aup2[i] + aup_2
                precition2[i] = precition2[i] + precition_2
                recall2[i] = recall2[i] + recall_2
                fmeasure2[i] = fmeasure2[i] + fmeasure_2
                auc3[i] = auc3[i] + auc_3
                aup3[i] = aup3[i] + aup_3
                precition3[i] = precition3[i] + precition_3
                recall3[i] = recall3[i] + recall_3
                fmeasure3[i] = fmeasure3[i] + fmeasure_3
                auc4[i] = auc4[i] + auc_4
                aup4[i] = aup4[i] + aup_4
                precition4[i] = precition4[i] + precition_4
                recall4[i] = recall4[i] + recall_4
                fmeasure4[i] = fmeasure4[i] + fmeasure_4
                auc5[i] = auc5[i] + auc_5
                aup5[i] = aup5[i] + aup_5
                precition5[i] = precition5[i] + precition_5
                recall5[i] = recall5[i] + recall_5
                fmeasure5[i] = fmeasure5[i] + fmeasure_5
            auc[i] = auc[i] / (captesty.shape[0] // nm)
            aup[i] = aup[i] / (captesty.shape[0] // nm)
            precition[i] = precition[i] / (captesty.shape[0] // nm)
            recall[i] = recall[i] / (captesty.shape[0] // nm)
            fmeasure[i] = fmeasure[i] / (captesty.shape[0] // nm)
            auc0[i] = auc0[i] / (captesty.shape[0] // nm)
            aup0[i] = aup0[i] / (captesty.shape[0] // nm)
            precition0[i] = precition0[i] / (captesty.shape[0] // nm)
            recall0[i] = recall0[i] / (captesty.shape[0] // nm)
            fmeasure0[i] = fmeasure0[i] / (captesty.shape[0] // nm)
            auc1[i] = auc1[i] / (captesty.shape[0] // nm)
            aup1[i] = aup1[i] / (captesty.shape[0] // nm)
            precition1[i] = precition1[i] / (captesty.shape[0] // nm)
            recall1[i] = recall1[i] / (captesty.shape[0] // nm)
            fmeasure1[i] = fmeasure1[i] / (captesty.shape[0] // nm)
            auc2[i] = auc2[i] / (captesty.shape[0] // nm)
            aup2[i] = aup2[i] / (captesty.shape[0] // nm)
            precition2[i] = precition2[i] / (captesty.shape[0] // nm)
            recall2[i] = recall2[i] / (captesty.shape[0] // nm)
            fmeasure2[i] = fmeasure2[i] / (captesty.shape[0] // nm)
            auc3[i] = auc3[i] / (captesty.shape[0] // nm)
            aup3[i] = aup3[i] / (captesty.shape[0] // nm)
            precition3[i] = precition3[i] / (captesty.shape[0] // nm)
            recall3[i] = recall3[i] / (captesty.shape[0] // nm)
            fmeasure3[i] = fmeasure3[i] / (captesty.shape[0] // nm)
            auc4[i] = auc4[i] / (captesty.shape[0] // nm)
            aup4[i] = aup4[i] / (captesty.shape[0] // nm)
            precition4[i] = precition4[i] / (captesty.shape[0] // nm)
            recall4[i] = recall4[i] / (captesty.shape[0] // nm)
            fmeasure4[i] = fmeasure4[i] / (captesty.shape[0] // nm)
            auc5[i] = auc5[i] / (captesty.shape[0] // nm)
            aup5[i] = aup5[i] / (captesty.shape[0] // nm)
            precition5[i] = precition5[i] / (captesty.shape[0] // nm)
            recall5[i] = recall5[i] / (captesty.shape[0] // nm)
            fmeasure5[i] = fmeasure5[i] / (captesty.shape[0] // nm)
    return np.sum(auc) / 5, np.sum(aup) / 5, np.sum(precition) / 5, np.sum(recall) / 5, np.sum(
            fmeasure) / 5, np.sum(
            auc0) / 5, np.sum(aup0) / 5, np.sum(precition0) / 5, np.sum(recall0) / 5, np.sum(fmeasure0) / 5, np.sum(
            auc1) / 5, \
               np.sum(aup1) / 5, np.sum(precition1) / 5, np.sum(recall1) / 5, np.sum(fmeasure1) / 5, np.sum(
            auc2) / 5, np.sum(aup2) / 5, np.sum(precition2) / 5, np.sum(recall2) / 5, np.sum(fmeasure2) / 5, np.sum(
            auc3) / 5, np.sum(aup3) / 5, np.sum(precition3) / 5, np.sum(recall3) / 5, np.sum(fmeasure3) / 5, np.sum(
            auc4) / 5, np.sum(aup4) / 5, np.sum(precition4) / 5, np.sum(recall4) / 5, np.sum(fmeasure4) / 5, np.sum(
            auc5) / 5, np.sum(aup5) / 5, np.sum(precition5) / 5, np.sum(recall5) / 5, np.sum(fmeasure5) / 5


def random1_triple(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,bool):
    nd = interaction.shape[0]
    nm = interaction.shape[1]
    f1=open('disbiomeknowninteraction.txt','r')
    txt=f1.read()
    txt = txt.split('\n')
    f1.close()
    auc=np.zeros((5,1))
    aup = np.zeros((5, 1))
    precition = np.zeros((5, 1))
    recall = np.zeros((5, 1))
    fmeasure = np.zeros((5, 1))
    auc0 = np.zeros((5, 1))
    aup0 = np.zeros((5, 1))
    precition0 = np.zeros((5, 1))
    recall0 = np.zeros((5, 1))
    fmeasure0 = np.zeros((5, 1))
    auc1 = np.zeros((5, 1))
    aup1 = np.zeros((5, 1))
    precition1 = np.zeros((5, 1))
    recall1 = np.zeros((5, 1))
    fmeasure1 = np.zeros((5, 1))
    auc2 = np.zeros((5, 1))
    aup2 = np.zeros((5, 1))
    precition2 = np.zeros((5, 1))
    recall2 = np.zeros((5, 1))
    fmeasure2 = np.zeros((5, 1))
    auc3 = np.zeros((5, 1))
    aup3 = np.zeros((5, 1))
    precition3 = np.zeros((5, 1))
    recall3 = np.zeros((5, 1))
    fmeasure3 = np.zeros((5, 1))
    auc4 = np.zeros((5, 1))
    aup4 = np.zeros((5, 1))
    precition4 = np.zeros((5, 1))
    recall4 = np.zeros((5, 1))
    fmeasure4 = np.zeros((5, 1))
    auc5 = np.zeros((5, 1))
    aup5 = np.zeros((5, 1))
    precition5 = np.zeros((5, 1))
    recall5 = np.zeros((5, 1))
    fmeasure5 = np.zeros((5, 1))
    index = [i for i in range(len(txt)-1)]
    np.random.shuffle(index)
    zeros1=np.zeros((nm*nd-len(txt)+1,2))
    temp=0

    for i in range(nd):
        for j in range(nm):
            if interaction[i][j]==0:
                zeros1[temp][0]=i
                zeros1[temp][1]=j
                temp+=1
    np.random.shuffle(zeros1)
    for i in range(5):
        X = interaction.copy()
        W = interaction.copy()
        # W=np.ones((nd,nm))
        test0=[]
        for d in range(nd):
            for h in range(nm):
                if W[d][h]==0:
                    W[d][h]=0.5
        if i!=4:
            test=np.zeros((int((len(txt)-1)/5),2))
            for j in range(int((len(txt)-1)/5)):
                test[j][0]=int(txt[index[i*int((len(txt)-1)/5)+j]].split('\t')[0])-1
                test[j][1]=int(txt[index[i*int((len(txt)-1)/5)+j]].split('\t')[1])-1
        if i==4:
            test=np.zeros((len(txt)-1-4*int((len(txt)-1)/5),2))
            for j in range(len(txt)-1-4*int((len(txt)-1)/5)):
                test[j][0]=int(txt[index[i*int((len(txt)-1)/5)+j]].split('\t')[0])-1
                test[j][1]=int(txt[index[i*int((len(txt)-1)/5)+j]].split('\t')[1])-1
        colum1=np.unique(test[:,1])
        colum=[]
        for j in range(len(colum1)):
            num=0
            for u in range(nd):
                if X[u][int(colum1[j])]==0:
                    # W[u][int(colum1[j])] =0
                    num=num+1
                for m1 in range(test.shape[0]):
                    if test[m1][1]==colum1[j] and u==test[m1][0]:
                        X[u][int(colum1[j])]=0
                        W[u][int(colum1[j])] = 0
                        num=num+1
            colum.append([int(colum1[j]),num])
        colum=np.array((colum))
        captest = []
        captest1=[]
        captesty=[]
        captrain0x1 = []
        captrain0x2 = []
        captrain0y = []
        captrain1x1=[]
        captrain1x2 = []
        captrain1y = []
        F,R,kd,km=comtriple(X, W, L_parameter, D_parameter, U_parameter, V_parameter, k, m,BOOL)
        numtest = 0
        tmpF=np.hstack((F,np.zeros((nd,38))))
        tmpR=np.hstack((R,np.zeros((nd,38))))
        for b in range(interaction.shape[1]):
            for a in range(interaction.shape[0]):
                if b in colum1:
                    if X[a][b]==0:
                        tmp = np.zeros((4 * (nm // nd + 1), nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1)):
                            if c < 2 * (nm // nd + 1):
                                if c % 2 == 0:
                                    tmp[c] = tmpF[:, b].T
                                else:
                                    tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                    numf = numf + 1
                            else:
                                if c % 2 == 0:
                                    tmp[c] = tmpR[:, b]
                                else:
                                    tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                    captest.append(tmp)
                    captest1.append(np.hstack((F[a],F[:,b].T)))
                    captesty.append(np.hstack((np.hstack((np.hstack((1-interaction[a][b],interaction[a][b])),F[a][b])),R[a][b])))
                else:
                    if  X[a][b]==0 and interaction[a][b]==0:
                        if X[a][b]==0:
                            tmp = np.zeros((4 * (nm // nd + 1), nd))
                            numf = 0
                            numr = 0
                            for c in range(4 * (nm // nd + 1)):
                                if c < 2 * (nm // nd + 1):
                                    if c % 2 == 0:
                                        tmp[c] = tmpF[:, b].T
                                    else:
                                        tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                        numf = numf + 1
                                else:
                                    if c % 2 == 0:
                                        tmp[c] = tmpR[:, b]
                                    else:
                                        tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                        numr = numr + 1
                            captrain0x1.append(tmp)
                            # captest.append(tmp)
                            # captesty.append(np.hstack((np.hstack((np.hstack((1-interaction[a][b],interaction[a][b])),F[a][b])),R[a][b])))
                            captrain0x2.append(np.hstack((F[a],F[:,b].T)))
                            captrain0y.append(np.hstack((np.hstack((np.hstack((1-interaction[a][b],interaction[a][b])),F[a][b])),R[a][b])))
                    if  X[a][b]==1 and interaction[a][b]==1:
                        tmp = np.zeros((4 * (nm // nd + 1), nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1)):
                            if c < 2 * (nm // nd + 1):
                                if c % 2 == 0:
                                    tmp[c] = tmpF[:, b].T
                                else:
                                    tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                    numf = numf + 1
                            else:
                                if c % 2 == 0:
                                    tmp[c] = tmpR[:, b]
                                else:
                                    tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                        captrain1x1.append(tmp)
                        captrain1x2.append(np.hstack((F[a], F[:, b].T)))
                        captrain1y.append(np.hstack((np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])),R[a][b])))
        captesty=np.array(captesty)
        captest=np.array(captest)
        captest1=np.array(captest1)
        captrain0y=np.array(captrain0y)
        captrain0x2=np.array(captrain0x2)
        captrain0x1 = np.array(captrain0x1)
        captrain1x2 = np.array(captrain1x2)
        captrain1x1 = np.array(captrain1x1)
        captrain1y = np.array(captrain1y)
        captrainresult,capresult=newcap.capmain(captrain0x1,captrain0y,captrain1x1,captrain1y,captest,captesty)        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=newmetric(captest[:,(captest.shape[1]-1)],result)
        train_tmp_y=np.vstack((captrain0y,captrain1y))
        trainx=np.hstack((train_tmp_y[:,2:4],captrainresult[:,1:2]))
        # train_tmp_y=np.vstack((captrain0y,captrain1y))
        # trainx=train_tmp_y[:,2:4]
        trainy=train_tmp_y[:,1:2]
        from sklearn import linear_model
        regr = linear_model.LogisticRegression()
        regr.fit(trainx,trainy)
        pre = regr.predict_proba(np.hstack((captesty[:,2:4],capresult[:,1:2])))
        pre1 = captesty[:, 1:3]
        pre2 = captesty[:, 2:4]
        pre3 = capresult
        regr = linear_model.LogisticRegression()
        regr.fit(train_tmp_y[:, 2:4], trainy)
        pre4 = regr.predict_proba(captesty[:, 2:4])
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 2:3], captrainresult[:, 1:2])), trainy)
        pre5 = regr.predict_proba(np.hstack((captesty[:, 2:3], capresult[:, 1:2])))
        regr = linear_model.LogisticRegression()
        regr.fit(np.hstack((train_tmp_y[:, 3:4], captrainresult[:, 1:2])), trainy)
        pre6 = regr.predict_proba(np.hstack((captesty[:, 3:4], capresult[:, 1:2])))
        auc[i], aup[i], precition[i], recall[i], fmeasure[i] = newmetric(
            captesty[:, 1:2], pre[:, 1:2])
        auc1[i], aup1[i], precition1[i], recall1[i], fmeasure1[i] = newmetric(
            captesty[:, 1:2], pre2[:, 1:2])
        auc2[i], aup2[i], precition2[i], recall2[i], fmeasure2[i] = newmetric(
            captesty[:, 1:2], pre3[:, 1:2])
        auc3[i], aup3[i], precition3[i], recall3[i], fmeasure3[i] = newmetric(
            captesty[:, 1:2], pre4[:, 1:2])
        auc4[i], aup4[i], precition4[i], recall4[i], fmeasure4[i] = newmetric(
            captesty[:, 1:2], pre5[:, 1:2])
        auc5[i], aup5[i], precition5[i], recall5[i], fmeasure5[i] = newmetric(
            captesty[:, 1:2], pre6[:, 1:2])
        auc0[i], aup0[i], precition0[i], recall0[i], fmeasure0[i] = newmetric(
            captesty[:, 1:2], pre1[:, 1:2])

    return np.sum(auc) / 5, np.sum(aup) / 5, np.sum(precition) / 5, np.sum(recall) / 5, np.sum(
        fmeasure) / 5, np.sum(
        auc0) / 5, np.sum(aup0) / 5, np.sum(precition0) / 5, np.sum(recall0) / 5, np.sum(fmeasure0) / 5, np.sum(
        auc1) / 5, \
           np.sum(aup1) / 5, np.sum(precition1) / 5, np.sum(recall1) / 5, np.sum(fmeasure1) / 5, np.sum(
        auc2) / 5, np.sum(aup2) / 5, np.sum(precition2) / 5, np.sum(recall2) / 5, np.sum(fmeasure2) / 5, np.sum(
        auc3) / 5, np.sum(aup3) / 5, np.sum(precition3) / 5, np.sum(recall3) / 5, np.sum(fmeasure3) / 5, np.sum(
        auc4) / 5, np.sum(aup4) / 5, np.sum(precition4) / 5, np.sum(recall4) / 5, np.sum(fmeasure4) / 5, np.sum(
        auc5) / 5, np.sum(aup5) / 5, np.sum(precition5) / 5, np.sum(recall5) / 5, np.sum(fmeasure5) / 5


def LOOCV(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,BOOL):
    nd=interaction.shape[0]
    nm=interaction.shape[1]
    onesindex=[]
    for i in range(nd):
        for j in range(nm):
            if interaction[i][j]==1:
                onesindex.append([i,j])
    onesindex=np.array(onesindex)
    # np.random.shuffle(onesindex)
    # X=interaction.copy()
    auc = np.zeros((450, 1))
    aup = np.zeros((450, 1))
    precition = np.zeros((450, 1))
    recall = np.zeros((450, 1))
    fmeasure = np.zeros((450, 1))
    for cv in range(10):
        X = interaction.copy()
        W=X.copy()
        for i in range(nd):
            for j in range(nm):
                if W[i][j]==0:
                    W[i][j]=0.5
        X[onesindex[cv][0]][onesindex[cv][1]]=0
        W[onesindex[cv][0]][onesindex[cv][1]]=0
        captest = []
        captest1 = []
        captesty = []
        captrain0x1 = []
        captrain0x2 = []
        captrain0y = []
        captrain1x1 = []
        captrain1x2 = []
        captrain1y = []
        F, R ,kd,km= comtriple(X, W, L_parameter, D_parameter, U_parameter, V_parameter, k, m)
        numtest = 0
        tmpF = np.hstack((F, np.zeros((nd, 20))))
        tmpR = np.hstack((R, np.zeros((nd, 20))))
        for b in range(interaction.shape[1]):
            for a in range(interaction.shape[0]):
                # if onesindex[cv][1]==b:
                if X[a][b] == 0 and interaction[a][b]==1:
                    tmp = np.zeros((4 * (nm // nd + 1), nd))
                    numf = 0
                    numr = 0
                    for c in range(4 * (nm // nd + 1)):
                        if c < 2 * (nm // nd + 1):
                            if c % 2 == 0:
                                tmp[c] = tmpF[:, b].T
                            else:
                                tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                numf = numf + 1
                        else:
                            if c % 2 == 0:
                                tmp[c] = tmpR[:, b]
                            else:
                                tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                    captest.append(tmp)
                    captest1.append(np.hstack((F[a], F[:, b].T)))
                    captesty.append(np.hstack(
                        (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
                    
                if X[a][b] == 0 and interaction[a][b]==0:
                    if X[a][b] == 0:
                        tmp = np.zeros((4 * (nm // nd + 1), nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1)):
                            if c < 2 * (nm // nd + 1):
                                if c % 2 == 0:
                                    tmp[c] = tmpF[:, b].T
                                else:
                                    tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                    numf = numf + 1
                            else:
                                if c % 2 == 0:
                                    tmp[c] = tmpR[:, b]
                                else:
                                    tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                        captrain0x1.append(tmp)
                        captest.append(tmp)
                        captest1.append(np.hstack((F[a], F[:, b].T)))
                        captesty.append(np.hstack(
                            (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))

                        captrain0x2.append(np.hstack((F[a], F[:, b].T)))
                        captrain0y.append(np.hstack(
                            (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
                if X[a][b] == 1:
                    tmp = np.zeros((4 * (nm // nd + 1), nd))
                    numf = 0
                    numr = 0
                    for c in range(4 * (nm // nd + 1)):
                        if c < 2 * (nm // nd + 1):
                            if c % 2 == 0:
                                tmp[c] = tmpF[:, b].T
                            else:
                                tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                numf = numf + 1
                        else:
                            if c % 2 == 0:
                                tmp[c] = tmpR[:, b]
                            else:
                                tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                    captrain1x1.append(tmp)
                    captrain1x2.append(np.hstack((F[a], F[:, b].T)))
                    captrain1y.append(np.hstack(
                        (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
        captesty = np.array(captesty)
        captest = np.array(captest)
        captest1 = np.array(captest1)
        captrain0y = np.array(captrain0y)
        captrain0x2 = np.array(captrain0x2)
        captrain0x1 = np.array(captrain0x1)
        captrain1x2 = np.array(captrain1x2)
        captrain1x1 = np.array(captrain1x1)
        captrain1y = np.array(captrain1y)
        print(captrain0x1.shape)
        print(captrain1x1.shape)
        captrainresult, capresult = newcap.capmain(captrain0x1, captrain0y, captrain1x1, captrain1y, captest, captesty)
        train_tmp_y=np.vstack((captrain0y,captrain1y))
        trainx = np.hstack((train_tmp_y[:, 2:4], captrainresult[:, 1:2]))
        # train_tmp_y=np.vstack((captrain0y,captrain1y))
        # trainx =train_tmp_y[:, 2:4]
        testx=np.hstack((captesty[:,2:4],capresult[:,1:2]))
        trainy = train_tmp_y[:, 1:2]
        from sklearn import linear_model
        regr = linear_model.LogisticRegression()
        regr.fit(trainx, trainy)
        pre = regr.predict_proba(testx)
        auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
            captesty[:, 1:2], pre[:, 1:2])
        auc[cv] = auc1
        aup[cv] = aup1
        precition[cv] = precition1
        recall[cv] = recall1
        fmeasure[cv] = fmeasure1
        print(auc[cv])
        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=newcap.capmain(captrain0x1,captrain0x2,captrain0y,captrain1x1,captrain1x2,captrain1y,captest,captest1,captesty,bool,colum)        # auc[i][0],aup[i][0],precition[i][0],recall[i][0],fmeasure[i][0]=newmetric(captest[:,(captest.shape[1]-1)],result)

    return np.sum(auc) / 10, np.sum(aup) / 10, np.sum(precition) / 10, np.sum(recall) / 10, np.sum(fmeasure) / 10


def case(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,bool):
    nd = interaction.shape[0]
    nm = interaction.shape[1]
    onesindex = []
    for i in range(nd):
        for j in range(nm):
            if interaction[i][j] == 1:
                onesindex.append([i, j])
    onesindex = np.array(onesindex)
    numid=35
   
    for cv in range(1):
        X = interaction.copy()
        W = X.copy()
        for i in range(nd):
            for j in range(nm):
                if W[i][j] == 0:
                    W[i][j] = 0.5
        X[numid] = 0
        W[numid] = 0
        captest = []
        captest1 = []
        captesty = []
        captrain0x1 = []
        captrain0x2 = []
        captrain0y = []
        captrain1x1 = []
        captrain1x2 = []
        captrain1y = []
        F, R, kd, km = comtriple(X, W, L_parameter, D_parameter, U_parameter, V_parameter,  k,
                                 m)
        numtest = 0
        tmpF = np.hstack((F, np.zeros((nd, 20))))
        tmpR = np.hstack((R, np.zeros((nd, 20))))
        for b in range(interaction.shape[1]):
            for a in range(interaction.shape[0]):
                if a==numid:
                    tmp = np.zeros((4 * (nm // nd + 1), nd))
                    numf = 0
                    numr = 0
                    for c in range(4 * (nm // nd + 1)):
                        if c < 2 * (nm // nd + 1):
                            if c % 2 == 0:
                                tmp[c] = tmpF[:, b].T
                            else:
                                tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                numf = numf + 1
                        else:
                            if c % 2 == 0:
                                tmp[c] = tmpR[:, b]
                            else:
                                tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                    captest.append(tmp)
                    captest1.append(np.hstack((F[a], F[:, b].T)))
                    captesty.append(np.hstack(
                        (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
                   
                if a!=numid and X[a][b] == 0:
                    if X[a][b] == 0:
                        tmp = np.zeros((4 * (nm // nd + 1), nd))
                        numf = 0
                        numr = 0
                        for c in range(4 * (nm // nd + 1)):
                            if c < 2 * (nm // nd + 1):
                                if c % 2 == 0:
                                    tmp[c] = tmpF[:, b].T
                                else:
                                    tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                    numf = numf + 1
                            else:
                                if c % 2 == 0:
                                    tmp[c] = tmpR[:, b]
                                else:
                                    tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                    numr = numr + 1
                        captrain0x1.append(tmp)
                        captrain0x2.append(np.hstack((F[a], F[:, b].T)))
                        captrain0y.append(np.hstack(
                            (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
                if a!=numid and X[a][b] == 1:
                    tmp = np.zeros((4 * (nm // nd + 1), nd))
                    numf = 0
                    numr = 0
                    for c in range(4 * (nm // nd + 1)):
                        if c < 2 * (nm // nd + 1):
                            if c % 2 == 0:
                                tmp[c] = tmpF[:, b].T
                            else:
                                tmp[c] = tmpF[a, numf * nd:(numf + 1) * nd]
                                numf = numf + 1
                        else:
                            if c % 2 == 0:
                                tmp[c] = tmpR[:, b]
                            else:
                                tmp[c] = tmpR[a, numr * nd:(numr + 1) * nd]
                                numr = numr + 1
                    captrain1x1.append(tmp)
                    captrain1x2.append(np.hstack((F[a], F[:, b].T)))
                    captrain1y.append(np.hstack(
                        (np.hstack((np.hstack((1 - interaction[a][b], interaction[a][b])), F[a][b])), R[a][b])))
        captesty = np.array(captesty)
        captest = np.array(captest)
        captest1 = np.array(captest1)
        captrain0y = np.array(captrain0y)
        captrain0x2 = np.array(captrain0x2)
        captrain0x1 = np.array(captrain0x1)
        captrain1x2 = np.array(captrain1x2)
        captrain1x1 = np.array(captrain1x1)
        captrain1y = np.array(captrain1y)
        print(captrain0x1.shape)
        print(captrain1x1.shape)
        captrainresult, capresult = newcap.capmain(captrain0x1, captrain0y, captrain1x1, captrain1y, captest, captesty)
        train_tmp_y = np.vstack((captrain0y, captrain1y))
        trainx = np.hstack((train_tmp_y[:, 2:4], captrainresult[:, 1:2]))
        # train_tmp_y=np.vstack((captrain0y,captrain1y))
        # trainx =train_tmp_y[:, 2:4]
        testx = np.hstack((captesty[:, 2:4], capresult[:, 1:2]))
        trainy = train_tmp_y[:, 1:2]
        from sklearn import linear_model
        regr = linear_model.LogisticRegression()
        regr.fit(trainx, trainy)
        pre = regr.predict_proba(testx)
        print(sort(pre[:,1:2]))

if __name__ == "__main__":
    nd=218
    nm=1052
    f1=open('data/disbiomeknowninteraction.txt','r')
    interaction=np.zeros((nd,nm))
    line=f1.readline()
    while line!='':
        interaction[int(line.replace('\n','').split('\t')[0])-1][int(line.replace('\n','').split('\t')[1])-1]=1
        line=f1.readline()
    f1.close()
    BOOL=1
    n=10
    auc=np.zeros(n)
    aup = np.zeros(n)
    precition = np.zeros(n)
    recall = np.zeros(n)
    fmeasure = np.zeros(n)
    auc0 = np.zeros(n)
    aup0 = np.zeros(n)
    precition0 = np.zeros(n)
    recall0 = np.zeros(n)
    fmeasure0 = np.zeros(n)
    auc1 = np.zeros(n)
    aup1 = np.zeros(n)
    precition1 = np.zeros(n)
    recall1 = np.zeros(n)
    fmeasure1 = np.zeros(n)
    auc2 = np.zeros(n)
    aup2 = np.zeros(n)
    precition2 = np.zeros(n)
    recall2 = np.zeros(n)
    fmeasure2 = np.zeros(n)
    auc3 = np.zeros((5, 1))
    aup3 = np.zeros((5, 1))
    precition3 = np.zeros((5, 1))
    recall3 = np.zeros((5, 1))
    fmeasure3 = np.zeros((5, 1))
    auc4 = np.zeros((5, 1))
    aup4 = np.zeros((5, 1))
    precition4 = np.zeros((5, 1))
    recall4 = np.zeros((5, 1))
    fmeasure4 = np.zeros((5, 1))
    auc5 = np.zeros((5, 1))
    aup5 = np.zeros((5, 1))
    precition5 = np.zeros((5, 1))
    recall5 = np.zeros((5, 1))
    fmeasure5 = np.zeros((5, 1))
    L_parameter = 0.05
    D_parameter = 0.05
    U_parameter = 0.8
    V_parameter = 0.7
    k = 400
    m=480
    if BOOL==0:
        
        for i in range(n):
            auc[i], aup[i], precition[i], recall[i], fmeasure[i], auc0[i], aup0[i], precition0[i], recall0[i],fmeasure0[i],auc1[i], aup1[i], precition1[i], recall1[i], fmeasure1[i], auc2[i], aup2[i], precition2[i], recall2[i],fmeasure2[i], auc3[i], aup3[i], precition3[i], recall3[i], fmeasure3[i], \
            auc4[i], aup4[i], precition4[i],recall4[i], fmeasure4[i],auc5[i], aup5[i], precition5[i], recall5[i], fmeasure5[i]=colum_triple(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,BOOL)




    if BOOL==1:
        for i in range(n):
            auc[i], aup[i], precition[i], recall[i], fmeasure[i], auc0[i], aup0[i], precition0[i], recall0[i],fmeasure0[i],auc1[i], aup1[i], precition1[i], recall1[i], fmeasure1[i], auc2[i], aup2[i], precition2[i], recall2[i],fmeasure2[i], auc3[i], aup3[i], precition3[i], recall3[i], fmeasure3[i], \
            auc4[i], aup4[i], precition4[i],recall4[i], fmeasure4[i],auc5[i], aup5[i], precition5[i], recall5[i], fmeasure5[i]=random1_triple(interaction,L_parameter,D_parameter,U_parameter,V_parameter,k,m,BOOL)
    if BOOL==2:
        for i in range(n):
            auc[i], aup[i], precition[i], recall[i], fmeasure[i], auc0[i], aup0[i], precition0[i], recall0[i],fmeasure0[i],auc1[i], aup1[i], precition1[i], recall1[i], fmeasure1[i], auc2[i], aup2[i], precition2[i], recall2[i],fmeasure2[i], auc3[i], aup3[i], precition3[i], recall3[i], fmeasure3[i],\
            auc4[i], aup4[i], precition4[i],recall4[i], fmeasure4[i],auc5[i], aup5[i], precition5[i], recall5[i], fmeasure5[i]= row_triple(interaction, L_parameter, D_parameter,
                                                                                  U_parameter, V_parameter,  k, m, BOOL)

    print('MVFA' + str(BOOL) + ',auc=' + str(np.sum(auc) / n) + ',varauc=' + str(np.var(auc)) + ',aup=' + str(
        np.sum(aup) / n) + ',varaup=' + str(np.var(aup)) + ',precition=' + str(
        np.sum(precition) / n) + ',varprecition=' + str(np.var(precition)) + ',recall=' + str(
        np.sum(recall) / n) + ',varrecall=' + str(np.var(recall)) + ',fmeasure=' + str(
        np.sum(fmeasure) / n) + ',varfmeasure=' + str(np.var(fmeasure)) + '\n')
    print('NMTF' + str(BOOL) + ',auc=' + str(np.sum(auc0) / n) + ',varauc=' + str(np.var(auc0)) + ',aup=' + str(
        np.sum(aup0) / n) + ',varaup=' + str(np.var(aup0)) + ',precition=' + str(
        np.sum(precition0) / n) + ',varprecition=' + str(np.var(precition0)) + ',recall=' + str(
        np.sum(recall0) / n) + ',varrecall=' + str(np.var(recall0)) + ',fmeasure=' + str(
        np.sum(fmeasure0) / n) + ',varfmeasure=' + str(np.var(fmeasure0)) + '\n')
    print('BIRW' + str(BOOL) + ',auc=' + str(np.sum(auc1) / n) + ',varauc=' + str(np.var(auc1)) + ',aup=' + str(
        np.sum(aup1) / n) + ',varaup=' + str(np.var(aup1)) + ',precition=' + str(
        np.sum(precition1) / n) + ',varprecition=' + str(np.var(precition1)) + ',recall=' + str(
        np.sum(recall1) / n) + ',varrecall=' + str(np.var(recall1)) + ',fmeasure=' + str(
        np.sum(fmeasure1) / n) + ',varfmeasure=' + str(np.var(fmeasure1)) + '\n')
    print('CAPNN' + str(BOOL) + ',auc=' + str(np.sum(auc2) / n) + ',varauc=' + str(np.var(auc2)) + ',aup=' + str(
        np.sum(aup2) / n) + ',varaup=' + str(np.var(aup2)) + ',precition=' + str(
        np.sum(precition2) / n) + ',varprecition=' + str(np.var(precition2)) + ',recall=' + str(
        np.sum(recall2) / n) + ',varrecall=' + str(np.var(recall2)) + ',fmeasure=' + str(
        np.sum(fmeasure2) / n) + ',varfmeasure=' + str(np.var(fmeasure2)) + '\n')
    print('MVFA-NMTF-BI' + str(BOOL) + ',auc=' + str(np.sum(auc3) / n) + ',varauc=' + str(np.var(auc3)) + ',aup=' + str(
        np.sum(aup3) / n) + ',varaup=' + str(np.var(aup3)) + ',precition=' + str(
        np.sum(precition3) / n) + ',varprecition=' + str(np.var(precition3)) + ',recall=' + str(
        np.sum(recall3) / n) + ',varrecall=' + str(np.var(recall3)) + ',fmeasure=' + str(
        np.sum(fmeasure3) / n) + ',varfmeasure=' + str(np.var(fmeasure3)) + '\n')
    print('MVFA-NMTF-CAPNN' + str(BOOL) + ',auc=' + str(np.sum(auc4) / n) + ',varauc=' + str(
        np.var(auc4)) + ',aup=' + str(
        np.sum(aup4) / n) + ',varaup=' + str(np.var(aup4)) + ',precition=' + str(
        np.sum(precition4) / n) + ',varprecition=' + str(np.var(precition4)) + ',recall=' + str(
        np.sum(recall4) / n) + ',varrecall=' + str(np.var(recall4)) + ',fmeasure=' + str(
        np.sum(fmeasure4) / n) + ',varfmeasure=' + str(np.var(fmeasure4)) + '\n')
    print(
        'MVFA-BI-CAPNN' + str(BOOL) + ',auc=' + str(np.sum(auc5) / n) + ',varauc=' + str(np.var(auc5)) + ',aup=' + str(
            np.sum(aup5) / n) + ',varaup=' + str(np.var(aup5)) + ',precition=' + str(
            np.sum(precition5) / n) + ',varprecition=' + str(np.var(precition5)) + ',recall=' + str(
            np.sum(recall5) / n) + ',varrecall=' + str(np.var(recall5)) + ',fmeasure=' + str(
            np.sum(fmeasure5) / n) + ',varfmeasure=' + str(np.var(fmeasure5)) + '\n')
