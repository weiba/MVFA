# -*- coding: utf-8 -*
"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...

Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
import warnings
import keras
warnings.filterwarnings("ignore")
from newmetric import newmetric
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
np.set_printoptions(threshold=np.inf)
import IF
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from PIL import Image
from newcapsulelayers import CapsuleLayer, PrimaryCap, Length, Mask, squash
import multi
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
K.set_image_data_format('channels_last')
import tensorflow as tf
from sklearn import linear_model
def myslice(x,a,b):
    return x[:,a:b]
# input_shape=[26,1],n_class=2,routings=3
def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)
    # x1 = layers.Lambda(myslice, output_shape=[256], arguments={'a': 0, 'b': 256})(x)
    # x3 = layers.Lambda(myslice, output_shape=[331], arguments={'a': 256, 'b': 587})(x)
    # x2 = layers.Lambda(myslice, output_shape=[331], arguments={'a': 587, 'b': 918})(x)
    # # x1 = layers.concatenate([x1, x3], axis=1)
    # tmpX2 = layers.Dense(128, activation='sigmoid')(x2)
    # tmpX2=layers.Dropout(0.05)(tmpX2)
    # tmpx2 = layers.Dense(64, activation='relu')(tmpX2)
    # tmpx2=layers.Dropout(0.05)(tmpx2)
    # tmpX1=layers.Dense(128, activation='relu')(x1)
    # tmpX1 = layers.Dropout(0.05)(tmpX1)
    # tmpx1 = layers.Dense(64, activation='relu')(tmpX1)
    # tmpx1 = layers.Dropout(0.05)(tmpx1)
    # tmp=layers.concatenate([tmpx2,tmpx1])
    # tmp=layers.concatenate([tmp,x3])
    # tmp=layers.Dense(256,activation='relu')(tmp)
    # tmp=layers.Reshape(target_shape=[-1,1])(tmp)
    conv1 = layers.convolutional.Convolution2D(filters=32, kernel_size=(2,218), strides=1, padding='valid',
                                               activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(19,1), strides=1, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings,
                             name='digitcaps')(primarycaps)

    out_caps = Length(name='capsnet')(digitcaps)
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)
    #
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=8 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    # model=models.Model(x,out_caps)
    # return train_model, eval_model
    # , manipulate_model
    return train_model,eval_model

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def sort_index(realvalue):
    x = realvalue.shape[0]
    id1 = np.zeros((x, 1))
    for i in range(x):
        id1[i][0] = i
    realvalue = realvalue.reshape((x, 1))
    c = np.hstack((realvalue, id1))
    for i in range(x):
        for j in range(x - 1):
            if c[j][0] > c[j + 1][0]:
                temp = c[j].copy()
                c[j] = c[j + 1].copy()
                c[j + 1] = temp.copy()
    result=np.zeros((x,1))
    for i in range(x):
        result[int(c[i][1])]=i/x
    return result


def sigmoid(x):
    return dchu(np.ones((x.shape[0], x.shape[1])), (1 + np.exp(-x)))


def dchu(X, Y):
    x = X.shape[0]
    y = X.shape[1]
    result = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            result[i][j] = X[i][j] / Y[i][j]
    return result

def softmax(train_batch_x, train_batch_y, test_x,test_y):
    x = tf.placeholder("float", [None, 10])
    y_ = tf.placeholder("float", [None, 2])
    W = tf.Variable(tf.zeros([10, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess.run(train_step, feed_dict={x: train_batch_x, y_: train_batch_y})
    # train_accuracy = accuracy.eval(feed_dict={x: train[:,0:10], y_: train[:,10:12]})
    numbers_y, numbers_y_ = sess.run([y, tf.argmax(y_, 1)], feed_dict={x: test_x, y_: test_y})


    return numbers_y
def kmeans1(captrain0x2,captrain0y,captrain1x2,captrain1y,SD,KM,A,F):
    from sklearn.cluster import KMeans
    import random
    unknown = []
    known = []
    nd=A.shape[0]
    nm=A.shape[1]
    for x in range(39):
        for y in range(292):
            if np.sum(A[:,y])!=0:
                if A[x, y] == 0:
                    unknown.append((x, y))
                else:
                    known.append((x, y))
    major = []
    for z in range(captrain0y.shape[0]):
        q = SD[unknown[z][0], :].tolist() + KM[unknown[z][1], :].tolist()
        major.append(q)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(major)
    center = kmeans.cluster_centers_
    center_x = []
    center_y = []
    for j in range(len(center)):
        center_x.append(center[j][0])
        center_y.append(center[j][1])
    labels = kmeans.labels_
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    # type6_x = []
    # type6_y = []
    # type7_x = []
    # type7_y = []
    # type8_x = []
    # type8_y = []
    # type9_x = []
    # type9_y = []
    # type10_x = []
    # type10_y = []
    # type11_x = []
    # type11_y = []
    # type12_x = []
    # type12_y = []
    # type13_x = []
    # type13_y = []
    # type14_x = []
    # type14_y = []
    # type15_x = []
    # type15_y = []
    # type16_x = []
    # type16_y = []
    # type17_x = []
    # type17_y = []
    # type18_x = []
    # type18_y = []
    # type19_x = []
    # type19_y = []
    # type20_x = []
    # type20_y = []
    # type21_x = []
    # type21_y = []
    # type22_x = []
    # type22_y = []
    # type23_x = []
    # type23_y = []
    for i in range(len(labels)):
        if labels[i] == 0:
            type1_x.append(unknown[i][0])
            type1_y.append(unknown[i][1])
        if labels[i] == 1:
            type2_x.append(unknown[i][0])
            type2_y.append(unknown[i][1])
        if labels[i] == 2:
            type3_x.append(unknown[i][0])
            type3_y.append(unknown[i][1])
        if labels[i] == 3:
            type4_x.append(unknown[i][0])
            type4_y.append(unknown[i][1])
        if labels[i] == 4:
            type5_x.append(unknown[i][0])
            type5_y.append(unknown[i][1])
        # if labels[i] == 5:
        #     type6_x.append(unknown[i][0])
        #     type6_y.append(unknown[i][1])
        # if labels[i] == 6:
        #     type7_x.append(unknown[i][0])
        #     type7_y.append(unknown[i][1])
        # if labels[i] == 7:
        #     type8_x.append(unknown[i][0])
        #     type8_y.append(unknown[i][1])
        # if labels[i] == 8:
        #     type9_x.append(unknown[i][0])
        #     type9_y.append(unknown[i][1])
        # if labels[i] == 9:
        #     type10_x.append(unknown[i][0])
        #     type10_y.append(unknown[i][1])
        # if labels[i] == 10:
        #     type11_x.append(unknown[i][0])
        #     type11_y.append(unknown[i][1])
        # if labels[i] == 11:
        #     type12_x.append(unknown[i][0])
        #     type12_y.append(unknown[i][1])
        # if labels[i] == 12:
        #     type13_x.append(unknown[i][0])
        #     type13_y.append(unknown[i][1])
        # if labels[i] == 13:
        #     type14_x.append(unknown[i][0])
        #     type14_y.append(unknown[i][1])
        # if labels[i] == 14:
        #     type15_x.append(unknown[i][0])
        #     type15_y.append(unknown[i][1])
        # if labels[i] == 15:
        #     type16_x.append(unknown[i][0])
        #     type16_y.append(unknown[i][1])
        # if labels[i] == 16:
        #     type17_x.append(unknown[i][0])
        #     type17_y.append(unknown[i][1])
        # if labels[i] == 17:
        #     type18_x.append(unknown[i][0])
        #     type18_y.append(unknown[i][1])
        # if labels[i] == 18:
        #     type19_x.append(unknown[i][0])
        #     type19_y.append(unknown[i][1])
        # if labels[i] == 19:
        #     type20_x.append(unknown[i][0])
        #     type20_y.append(unknown[i][1])
        # if labels[i] == 20:
        #     type21_x.append(unknown[i][0])
        #     type21_y.append(unknown[i][1])
        # if labels[i] == 21:
        #     type22_x.append(unknown[i][0])
        #     type22_y.append(unknown[i][1])
        # if labels[i] == 22:
        #     type23_x.append(unknown[i][0])
        #     type23_y.append(unknown[i][1])
    type = [[], [], [], [], []]
    mtype = [[], [], [], [], []]
    dataSet = []
    for k1 in range(len(type1_x)):
        type[0].append((type1_x[k1], type1_y[k1]))
    for k2 in range(len(type2_x)):
        type[1].append((type2_x[k2], type2_y[k2]))
    for k3 in range(len(type3_x)):
        type[2].append((type3_x[k3], type3_y[k3]))
    for k4 in range(len(type4_x)):
        type[3].append((type4_x[k4], type4_y[k4]))
    for k5 in range(len(type5_x)):
        type[4].append((type5_x[k5], type5_y[k5]))
    # for k6 in range(len(type6_x)):
    #     type[5].append((type6_x[k6], type6_y[k6]))
    # for k7 in range(len(type7_x)):
    #     type[6].append((type7_x[k7], type7_y[k7]))
    # for k8 in range(len(type8_x)):
    #     type[7].append((type8_x[k8], type8_y[k8]))
    # for k9 in range(len(type9_x)):
    #     type[8].append((type9_x[k9], type9_y[k9]))
    # for k10 in range(len(type10_x)):
    #     type[9].append((type10_x[k10], type10_y[k10]))
    # for k11 in range(len(type11_x)):
    #     type[10].append((type11_x[k11], type11_y[k11]))
    # for k12 in range(len(type12_x)):
    #     type[11].append((type12_x[k12], type12_y[k12]))
    # for k13 in range(len(type13_x)):
    #     type[12].append((type13_x[k13], type13_y[k13]))
    # for k14 in range(len(type14_x)):
    #     type[13].append((type14_x[k14], type14_y[k14]))
    # for k15 in range(len(type15_x)):
    #     type[14].append((type15_x[k15], type15_y[k15]))
    # for k16 in range(len(type16_x)):
    #     type[15].append((type16_x[k16], type16_y[k16]))
    # for k17 in range(len(type17_x)):
    #     type[16].append((type17_x[k17], type17_y[k17]))
    # for k18 in range(len(type18_x)):
    #     type[17].append((type18_x[k18], type18_y[k18]))
    # for k19 in range(len(type19_x)):
    #     type[18].append((type19_x[k19], type19_y[k19]))
    # for k20 in range(len(type20_x)):
    #     type[19].append((type20_x[k20], type20_y[k20]))
    # for k21 in range(len(type21_x)):
    #     type[20].append((type21_x[k21], type21_y[k21]))
    # for k22 in range(len(type22_x)):
    #     type[21].append((type22_x[k22], type22_y[k22]))
    # for k23 in range(len(type23_x)):
    #     type[22].append((type23_x[k23], type23_y[k23]))
    for k in range(5):
        mtype[k] = random.sample(type[k], 100)
    for m2 in range(39):
        for n2 in range(292):
            for z2 in range(5):
                if (m2, n2) in mtype[z2]:
                    dataSet.append((m2, n2))
    for m3 in range(39):
        for n3 in range(292):
            if A[m3, n3] == 1:
                dataSet.append((m3, n3))
    trainx = []
    trainy = []
    for xx in dataSet:
        if np.sum(A[:,xx[1]])!=0:
            tmp = np.zeros(( 2*(nm//nd+1), nd))
            tmpF = np.hstack((F, np.zeros(((nd, 20)))))
            numf = 0
            for c in range(2*(nm // nd+1)):
                if c % 2 == 0:
                    tmp[c] = tmpF[:, xx[1]].T
                else:
                    tmp[c] = tmpF[xx[0], numf * nd:(numf + 1) * nd]
                    numf = numf + 1
            trainx.append(tmp)
        if (xx[0], xx[1]) in known:
            trainy.append([0,1])
        else:
            trainy.append([1,0])
    trainy = np.array(trainy)
    trainx=np.array(trainx)
    return trainx,trainy
def train(model,eval_model,  data, args):
    traindata_tmp,traindata_tmp_y,test,test_y= data
    regr = linear_model.LogisticRegression()
    # cls = RandomForestClassifier(n_estimators=2000)
    # if bool == 1:
    #     train0 = captrain0.copy()
    #     np.random.shuffle(train0)
    #     test0 = train0[0:captest.shape[0]]
    #     test = np.vstack((captest, test0))
    #     data1 = captrain1.copy()
    #     data0 = captrain0[captest.shape[0]:captrain0.shape[0]]
    # if bool == 0:
    #     test = captest.copy()
    #     data1 = captrain1.copy()
    #     data0 = captrain0.copy()
    # n = data0.shape[0] // data1.shape[0] + 1
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', batch_size=args.batch_size,
                               histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss,'mse'],
                  loss_weights=[1.,args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    early_stopping = callbacks.EarlyStopping(patience=0, verbose=True)
    # traindata_tmp,traindata_tmp_y=kmeans1(captrain0x2,captrain0y,captrain1x2,captrain1y,kd,km,A)
    # captrain0x=np.expand_dims(captrain0x,axis=-1)
    # captrain1x = np.expand_dims(captrain1x, axis=-1)
    # traindata_tmp = np.vstack((captrain0x, captrain1x))
    # traindata_tmp_y=np.vstack((captrain0y[:,0:2],captrain1y[:,0:2]))
    # for a in range(1):
    #     if a==1:
    #         np.random.shuffle(captrain0x2)
    #     for i in range(captrain0x2.shape[0]//captrain1x2.shape[0]):
    #         print(i)
    #         # y_train_tmp = traindata_tmp[:, (traindata_tmp.shape[1] - 3):(traindata_tmp.shape[1] - 1)]
    #         # x_train_tmp = traindata_tmp[:, 1:(traindata_tmp.shape[1] - 3)]
    #         # x_train_tmp = normalize(x_train_tmp)
    #         # x_train_tmp = np.expand_dims(x_train_tmp, axis=-1)
    #         # traindata_tmp, traindata_tmp_y = kmeans1(captrain0x2, captrain0y, captrain1x2, captrain1y, kd, km, A,F)
    #         # np.random.shuffle(captrain0x2)
    #         # traindata_tmp_y=[]
    #         # traindata_tmp=[]
    #         # for a in range(captrain1x2.shape[0]):
    #         #     tmp = np.zeros((16, 39))
    #         #     numf = 0
    #         #     for c in range(16):
    #         #         if c % 2 == 0:
    #         #             tmp[c] = captrain0x2[a, 292:]
    #         #         else:
    #         #             tmp[c] = captrain0x2[a, numf * 39:(numf + 1) * 39]
    #         #             numf = numf + 1
    #         #     traindata_tmp.append(tmp)
    #         #     traindata_tmp_y.append([0,1])
    #         # for a in range(2*captrain1x2.shape[0]):
    #         #     tmp = np.zeros((16, 39))
    #         #     numf = 0
    #         #     for c in range(16):
    #         #         if c % 2 == 0:
    #         #             tmp[c] = captrain0x2[a, 292:]
    #         #         else:
    #         #             tmp[c] = captrain0x2[a, numf * 39:(numf + 1) * 39]
    #         #             numf = numf + 1
    #         #     traindata_tmp.append(tmp)
    #         #     traindata_tmp_y.append([1, 0])
    #         # traindata_tmp_y = np.array(traindata_tmp_y)
    #         # traindata_tmp = np.array(traindata_tmp)
    #         traindata_tmp=[]
    #         traindata_tmp_y=[]
    #         for a in range(2*captrain1x2.shape[0]):
    #             if a<captrain1x2.shape[0]:
    #                 tmp = np.zeros((16, 39))
    #                 numf = 0
    #                 for c in range(16):
    #                     if c % 2 == 0:
    #                         tmp[c] = captrain0x2[i*captrain1x2.shape[0]+a, 292:]
    #                     else:
    #                         tmp[c] = captrain0x2[i*captrain1x2.shape[0]+a, numf * 39:(numf + 1) * 39]
    #                         numf = numf + 1
    #                 traindata_tmp.append(tmp)
    #                 traindata_tmp_y.append([1, 0])
    #             else:
    #                 tmp = np.zeros((16, 39))
    #                 numf = 0
    #                 for c in range(16):
    #                     if c % 2 == 0:
    #                         tmp[c] = captrain1x2[a-captrain1x2.shape[0], 292:]
    #                     else:
    #                         tmp[c] = captrain1x2[a-captrain1x2.shape[0], numf * 39:(numf + 1) * 39]
    #                         numf = numf + 1
    #                 traindata_tmp.append(tmp)
    #                 traindata_tmp_y.append([0, 1])
    #         traindata_tmp=np.array(traindata_tmp)
    #         traindata_tmp_y=np.array(traindata_tmp_y)
    #         traindata_tmp=np.expand_dims(traindata_tmp,axis=-1)
    #         # from sklearn.model_selection import train_test_split
    #         # traindata_tmp, x_test,traindata_tmp_y, y_test = train_test_split(traindata_tmp, traindata_tmp_y, test_size=0.2)
    #         model.fit([traindata_tmp, traindata_tmp_y], [traindata_tmp_y, traindata_tmp], batch_size=args.batch_size,
    #                   epochs=args.epochs, callbacks=[log, tb, checkpoint, lr_decay,early_stopping])
    # traindata_tmp=np.expand_dims(traindata_tmp,axis=-1)
    model.fit([traindata_tmp, traindata_tmp_y[:,0:2]], [traindata_tmp_y[:,0:2],traindata_tmp],batch_size=args.batch_size,
                                epochs=args.epochs, callbacks=[log, tb, checkpoint, lr_decay])

    # get_1_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                                 [model.layers[5].output])
    #
    # layer_output1 = get_1_layer_output([traindata_tmp[0:traindata_tmp.shape[0]//3]])[0]
    # layer_output2 = get_1_layer_output([traindata_tmp[traindata_tmp.shape[0] // 3:2*traindata_tmp.shape[0] // 3]])[0]
    # layer_output3 = get_1_layer_output([traindata_tmp[2*traindata_tmp.shape[0] // 3:traindata_tmp.shape[0]]])[0]
    # vec1 = layer_output1[:, 1:2, :]
    # vec2 = layer_output2[:, 1:2, :]
    # vec3 = layer_output3[:, 1:2, :]
    # vec = np.vstack((np.vstack((vec1.reshape(vec1.shape[0], 8),vec2.reshape(vec2.shape[0],8))),vec3.reshape(vec3.shape[0],8)))

    # for a in range(1):
    #     np.random.shuffle(data0)
    #     # traindata_tmp = np.vstack((data1, data0[0:data1.shape[0]]))
    #     for i in range(n):
    #         if i==n-1:
    #             traindata_tmp=np.vstack((data1,data0[i*data1.shape[0]:data0.shape[0]]))
    #         if i!=n-1:
    #             traindata_tmp = np.vstack((data1, data0[(i * data1.shape[0]):(i+1)*data1.shape[0]]))
    #         x_train_tmp=traindata_tmp[:,1:(traindata_tmp.shape[1]-3)]
    #         x_train_tmp = np.expand_dims(x_train_tmp, axis=-1)
    #         y_train_tmp=traindata_tmp[:,(traindata_tmp.shape[1]-3):(traindata_tmp.shape[1]-1)]
    #         model.fit([x_train_tmp, y_train_tmp], [y_train_tmp, x_train_tmp], batch_size=args.batch_size, epochs=args.epochs,callbacks=[log,tb, checkpoint, lr_decay])
    # for i in range(n):
    #     if i == n - 1:
    #         traindata_tmp = np.vstack((data1, data0[i * data1.shape[0]:data0.shape[0]]))
    #     if i != n - 1:
    #         traindata_tmp = np.vstack((data1, data0[(i * data1.shape[0]):(i + 1) * data1.shape[0]]))
    #     x_train_tmp = traindata_tmp[:, 1:(traindata_tmp.shape[1] - 3)]
    #     x_train_tmp = np.expand_dims(x_train_tmp, axis=-1)
    #     capresult,bad=eval_model.predict(x_train_tmp)
    #     LRtrainx=np.hstack((traindata_tmp[:,(traindata_tmp.shape[1]-1):traindata_tmp.shape[1]],capresult[:,1:2]))
    #     LRtrainy=traindata_tmp[:,traindata_tmp.shape[1]-2:traindata_tmp.shape[1]-1]
    #     if LRtrainx.shape[0]>data1.shape[0]:
    #         regr.fit(LRtrainx,LRtrainy)
    # captestx=[]
    # for i in range(captestx2.shape[0]):
    #     tmp = np.zeros(( 16, 39))
    #     numf = 0
    #     for c in range(16):
    #         if c % 2 == 0:
    #             tmp[c] = captestx2[i, 292:]
    #         else:
    #             tmp[c] = captestx2[i, numf * 39:(numf + 1) * 39]
    #             numf = numf + 1
    #     captestx.append(tmp)
    # for a in range(captrain1x2.shape[0]):
    #     tmp = np.zeros((16, 39))
    #     numf = 0
    #     for c in range(16):
    #         if c % 2 == 0:
    #             tmp[c] = captrain0x2[a, 292:]
    #         else:
    #             tmp[c] = captrain0x2[a, numf * 39:(numf + 1) * 39]
    #             numf = numf + 1
    #     captestx.append(tmp)
    # captestx=np.array(captestx)
    # captestx=np.expand_dims(captestx2,axis=-1)
    test=np.expand_dims(test,axis=-1)
    capresult,bad = eval_model.predict(test)
    trainresult,bad=eval_model.predict(traindata_tmp)
    # trainx=np.hstack((traindata_tmp_y[:,2:4],trainx[:,1:2]))
    # trainy=traindata_tmp_y[:,1:2]
    # testx=np.hstack((test_y[:,2:4],capresult[:,1:2]))

    # capresult=capresult[0:captestx.shape[0]]
    # get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                               [model.layers[5].output])
    # layer_out = get_layer_output([test])[0]
    # test_v = layer_out[:, 1:2, :]
    # test_v = test_v.reshape(test_v.shape[0], 8)
    # newtrain=np.hstack((np.vstack((captrain0x2,captrain1x2)),vec))
    # newtrainy=np.vstack((captrain0y,captrain1y))
    # newtest=np.hstack((captestx2,test_v))
    # import RF
    # pre=RF.RF(newtrain,newtrainy,newtest)
    # captrainresult, bad = eval_model.predict(traindata_tmp)
    # import gcForest
    # # gctrain,gcpre = gcForest.gcforestmain(np.hstack((np.hstack((np.hstack((vec[0:captrain0x2.shape[0]],captrain0x2)),captrain0y[:,2:3])),captrainresult[0:captrain0y.shape[0],1:2])), captrain0y,
    # #                                       np.hstack((np.hstack((np.hstack((vec[captrain0x2.shape[0]:],captrain1x2)),
    # #                                                             captrain1y[:,2:3])),captrainresult[captrain0y.shape[0]:,1:2])), captrain1y,
    # #                                       np.hstack((np.hstack((np.hstack((test_v,captestx2)),captesty[:,2:3])),capresult[:,1:2])), captesty, bool)
    # gctrain, gcpre = gcForest.gcforestmain(captrain0x2,captrain0y,captrain1x2,captrain1y,captestx2,captesty,bool)

    # gctrain,gcpre = gcForest.gcforestmain(captrain0x2, captrain0y, captrain1x2, captrain1y, captestx2, captesty, bool)
    # LR
    # LRtrainx = np.hstack((np.vstack((captrain0y[:,2:3],captrain1y[:,2:3])),gctrain[:,1:2]))
    # LRtrainy =np.vstack((captrain0y[:, 1:2],captrain1y[:,1:2]))
    # LRtestx =np.hstack((captesty[:,2:3],gcpre[:,1:2]))
    # LRtesty=captesty[:,0:2]
    # trainx=np.hstack((traindata_tmp_y[:,2:4],trainresult[:,1:2]))
    # trainy=traindata_tmp_y[:,1:2]
    # testx=np.hstack((test_y[:,2:4],capresult[:,1:2]))
    # regr.fit(trainx, trainy)
    # pre = regr.predict_proba(testx)
    # nd = 39
    # auc = 0
    # aup = 0
    # precition = 0
    # recall = 0
    # fmeasure = 0
    # test = test_y
    # # triresult = captesty[:, 2:4]
    # # if bool == 0:
    # for i in range(test.shape[0] // nd):
    #     # result1=sort_index(capresult[i * nd:(i + 1) * nd, 0:1])+sort_index(triresult[i * nd:(i + 1) * nd, 0:1])
    #     auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #         test[i * nd:(i + 1) * nd, 1:2],
    #         capresult[i * nd:(i + 1) * nd, 1:2])
    #     auc = auc + auc1
    #     aup = aup + aup1
    #     precition = precition + precition1
    #     recall = recall + recall1
    #     fmeasure = fmeasure + fmeasure1
    # if bool == 1:
    #     num = 0
    #     for i in range(colum.shape[0]):
    #         if i == 0:
    #             num = colum[i][1]
    #         else:
    #             num = num + colum[i][1]
    #         auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #             test[num - colum[i][1]:num, 1:2],
    #             capresult[num - colum[i][1]:num, 1:2])
    #         auc = auc + auc1
    #         aup = aup + aup1
    #         precition = precition + precition1
    #         recall = recall + recall1
    #         fmeasure = fmeasure + fmeasure1
    # auc = auc / (test.shape[0] // nd)
    # aup = aup / (test.shape[0] // nd)
    # precition = precition / (test.shape[0] // nd)
    # recall = recall / (test.shape[0] // nd)
    # fmeasure = fmeasure / (test.shape[0] // nd)
    # print('cap:'+str(auc))
    # nd = 39
    # auc = 0
    # aup = 0
    # precition = 0
    # recall = 0
    # fmeasure = 0
    # if bool==0:
    #     for i in range(test.shape[0] // nd):
    #         # result1=sort_index(capresult[i * nd:(i + 1) * nd, 0:1])+sort_index(triresult[i * nd:(i + 1) * nd, 0:1])
    #         auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #             test[i * nd:(i + 1) * nd, 1:2],
    #             pre[i * nd:(i + 1) * nd, 1:2])
    #         auc = auc + auc1
    #         aup = aup + aup1
    #         precition = precition + precition1
    #         recall = recall + recall1
    #         fmeasure = fmeasure + fmeasure1
    # if bool==1:
    #     num=0
    #     for i in range(colum.shape[0]):
    #         if i==0:
    #             num=colum[i][1]
    #         else:
    #             num=num+colum[i][1]
    #         auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #             test[num-colum[i][1]:num, 1:2],
    #             pre[num-colum[i][1]:num, 1:2])
    #         auc = auc + auc1
    #         aup = aup + aup1
    #         precition = precition + precition1
    #         recall = recall + recall1
    #         fmeasure = fmeasure + fmeasure1
    # auc = auc / (test.shape[0] // nd)
    # aup = aup / (test.shape[0] // nd)
    # precition = precition / (test.shape[0] // nd)
    # recall = recall / (test.shape[0] // nd)
    # fmeasure = fmeasure / (test.shape[0] // nd)
    # print(auc)

    # captrainresult, bad = eval_model.predict(traindata_tmp)
    # get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                               [model.layers[5].output])
    # layer_out = get_layer_output([testtmp])[0]
    # test_v = layer_out[:, 1:2, :]
    # test_v = test_v.reshape(test_v.shape[0], 8)
    # cls.fit(np.hstack((vec,np.vstack((captrain0x2,captrain1x2)))),traindata_tmp_y[:,1:2])
    # pre=cls.predict_proba(np.hstack((test_v,captestx2)))
    # capresult1=sort_index(capresult[:, 1:2])
    # capresult=np.zeros((capresult1.shape[0],1))
    # for i in range(capresult1.shape[0]):
    #     capresult[int(capresult1[i][1])][0]=i
    # capresult=capresult/capresult.shape[0]
    # triresult1=sort_index(test[:, (test.shape[1] - 1):test.shape[1]])
    # triresult = np.zeros((triresult1.shape[0], 1))
    # for i in range(triresult1.shape[0]):
    #     triresult[int(triresult1[i][1])][0] = i
    # triresult=triresult/triresult.shape[0]
    # pre=capresult+triresult
    #LR
    # LRtrainx = np.hstack((np.vstack((captrain0y[:,2:3],captrain1y[:,2:3])),captrainresult[:,1:2]))
    # LRtrainy =np.vstack((captrain0y[:, 1:2],captrain1y[:,1:2]))
    # LRtestx =np.hstack((captesty[:,2:3], capresult[:, 1:2]))
    # LRtesty=captesty[:,0:2]
    # regr.fit(LRtrainx, LRtrainy)
    # pre = regr.predict_proba(LRtestx)
    #xgboost
    # trainx=np.hstack((vec,np.vstack((captrain0x2,captrain1x2))))
    # testx=np.hstack((test_v,captestx2))
    # from gcForest import gcForest
    # gcf=gcForest(shape_1X=trainx.shape[1],window=3,n_cascadeRF=2)
    # gcf.fit(trainx,traindata_tmp_y[:,1:2])
    # result=gcf.predict_proba(testx)

    # auc, aup, precition, recall, fmeasure=CNN.CNNmain(trainx,traindata_tmp_y,testx,captesty[:,0:2],bool)
    # pre=softmax(LRtrainx,LRtrainy,LRtestx,LRtesty)
    # result1=capresult[:,1:2]+1e+3*captesty[:,2:3]
    # clf = XGBClassifier(silent=True, objective='binary:logistic', min_child_weight=31)
    # param_test = {
    #     'learning_rate': [0.0256, 0.026, 0.0261, 0.0262, 0.0264],
    #     'max_depth': [2, 5],
    #     'reg_lambda': [17, 20, 30],
    #     'n_estimators': [21, 200, 210],
    # }
    #
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_test, scoring='roc_auc', cv=5)
    # grid_search.fit(LRtrainx, LRtrainy)
    # pre=grid_search.predict_proba(LRtestx)
    # cls = RandomForestClassifier(n_estimators=2000)
    # cls.fit(LRtrainx,LRtrainy)
    # pre=cls.predict_proba(LRtestx)
    # regr.fit(LRtrainx, LRtrainy)
    # LRtestx = np.hstack((np.hstack((captesty[:,2:3], capresult[:, 1:2])),test_v))
    # pre = regr.predict_proba(LRtestx)

    # if bool==0:
    #     a2=test.shape[0]//n
    #     a1=(test.shape[0]-2*a2)//data1.shape[0]
    #     for i in range(a1):
    #         if i==a1-1:
    #             testtmp=np.vstack((data1,test[i*data1.shape[0]:test.shape[0]]))
    #         else:
    #             testtmp=np.vstack((data1,test[i*data1.shape[0]:(i+1)*data1.shape[0]]))
    #         testtmp = testtmp[:, 1:testtmp.shape[1] - 3]
    #         testtmp1=testtmp.copy()
    #         testtmp = np.expand_dims(testtmp, axis=-1)
    #         capresult, bad = eval_model.predict(testtmp)
    #         LRtestx = np.hstack((testtmp1[:, (testtmp1.shape[1] - 1):testtmp1.shape[1]], capresult[:, 1:2]))
    #         pre_result = regr.predict_proba(LRtestx)
    #         if i==0:
    #             pre = pre_result[data1.shape[0]:pre_result.shape[0], 1:2]
    #         else:
    #             pre=np.vstack((pre,pre_result[data1.shape[0]:pre_result.shape[0],1:2]))
    # import capsulenet
    # newtrainx0=np.hstack((np.hstack((vec[0:captrain0y.shape[0]],captrain0y[:,2:3])),captrainresult[0:captrain0y.shape[0],1:2]))
    # newtrainx1=np.hstack((np.hstack((vec[captrain0y.shape[0]:vec.shape[0]],captrain1y[:,2:3])),captrainresult[captrain0y.shape[0]:captrainresult.shape[0],1:2]))
    # newtest=np.hstack((np.hstack((test_v,captesty[:,2:3])),capresult[:,1:2]))
    # result=capsulenet.capmain(np.hstack((newtrainx0,captrain0y[:,0:2])),np.hstack((newtrainx1,captrain1y[:,0:2])),np.hstack((newtest,captesty[:,0:2])),0)
    # nd = 39
    # auc = 0
    # aup = 0
    # precition = 0
    # recall = 0
    # fmeasure = 0
    # test=captesty
    # triresult=captesty[:, 2:4]
    # if bool == 0:
    #     for i in range(test.shape[0] // nd):
    #         # result1=sort_index(capresult[i * nd:(i + 1) * nd, 0:1])+sort_index(triresult[i * nd:(i + 1) * nd, 0:1])
    #         auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #             test[i * nd:(i + 1) * nd, 1:2],
    #             result[i * nd:(i + 1) * nd, 1:2])
    #         auc = auc + auc1
    #         aup = aup + aup1
    #         precition = precition + precition1
    #         recall = recall + recall1
    #         fmeasure = fmeasure + fmeasure1
    # if bool == 1:
    #     num = 0
    #     for i in range(colum.shape[0]):
    #
    #         if i == 0:
    #             num = colum[i][1]
    #         else:
    #             num =num+ colum[i][1]
    #         auc1, aup1, precition1, recall1, fmeasure1 = newmetric(
    #             test[num - colum[i][1]:num, 1:2],
    #             capresult[num - colum[i][1]:num, 1:2])
    #         auc = auc + auc1
    #         aup = aup + aup1
    #         precition = precition + precition1
    #         recall = recall + recall1
    #         fmeasure = fmeasure + fmeasure1
    # auc = auc / (test.shape[0] // nd)
    # aup = aup / (test.shape[0] // nd)
    # precition = precition / (test.shape[0] // nd)
    # recall = recall / (test.shape[0] // nd)
    # fmeasure = fmeasure / (test.shape[0] // nd)
    # print(auc)
    # import uniform
    #
    # auc, aup, precition, recall, fmeasure=uniform.main(trainx,trainy,testx,testy,bool)
    return trainresult,capresult


def normalize(arr):
    result = np.zeros((arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i][j] = float(arr[i][j] - np.min(arr[i, :])) / (np.max(arr[i, :]) - np.min(arr[i, :]))
    return result




def capmain(captrain0x2,captrain0y,captrain1x2,captrain1y,captestx2,captesty):
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Capsule Network.")
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.04, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float, help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=4, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir',
                        default='result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(args.routings)
    # from MDA import MDAmain
    # captrain,captest1=MDAmain(np.vstack((captrain1[:,1:captrain1.shape[1]-3],captrain0[:,1:captrain0.shape[1]-3])),captest1[:,1:captest1.shape[1]-3])
    #
    # captrain1=np.hstack((captrain1[:,0:1],np.hstack((captrain[0:captrain1.shape[0]],captrain1[:,captrain1.shape[1]-3:captrain1.shape[1]]))))
    # captrain0=np.hstack((captrain0[:,0:1],np.hstack((captrain[captrain1.shape[0]:captrain.shape[0]],captrain0[:,captrain0.shape[1]-3:captrain0.shape[1]]))))
    # captest1=np.hstack((captest1[:,0:1],np.hstack((captest1,captest1[:,captest1.shape[1]-3:captest1.shape[1]]))))
    # traindata_tmp_y=[]
    # traindata_tmp=[]
    # for a in range(captrain1x2.shape[0]):
    #     tmp = np.zeros((16, 39))
    #     numf = 0
    #     for c in range(16):
    #         if c % 2 == 0:
    #             tmp[c] = captrain0x2[a, 292:]
    #         else:
    #             tmp[c] = captrain0x2[a, numf * 39:(numf + 1) * 39]
    #             numf = numf + 1
    #     traindata_tmp.append(tmp)
    #     traindata_tmp_y.append([0,1])
    # for a in range(captrain0x2.shape[0]):
    #     tmp = np.zeros((16, 39))
    #     numf = 0
    #     for c in range(16):
    #         if c % 2 == 0:
    #             tmp[c] = captrain0x2[a, 292:]
    #         else:
    #             tmp[c] = captrain0x2[a, numf * 39:(numf + 1) * 39]
    #             numf = numf + 1
    #     traindata_tmp.append(tmp)
    #     traindata_tmp_y.append([1, 0])
    # traindata_tmp_y = np.array(traindata_tmp_y)
    # traindata_tmp = np.array(traindata_tmp)
    # test=[]
    # test_y=[]
    # numf=0
    # for a in range(captestx2.shape[0]):
    #     tmp = np.zeros((16, 39))
    #     numf = 0
    #     for c in range(16):
    #         if c % 2 == 0:
    #             tmp[c] = captestx2[a, 292:]
    #         else:
    #             tmp[c] = captestx2[a, numf * 39:(numf + 1) * 39]
    #             numf = numf + 1
    #     test.append(tmp)
    #     test_y.append(captesty[a,0:2])
    # test=np.array(test)
    # test_y=np.array(test_y)
    traindata_tmp=np.vstack((captrain0x2,captrain1x2))
    traindata_tmp_y=np.vstack((captrain0y,captrain1y))
    test=captestx2
    test_y=captesty
    traindata_tmp=np.expand_dims(traindata_tmp,axis=-1)
    model,eval_model= CapsNet(input_shape=traindata_tmp.shape[1:], n_class=2, routings=args.routings)
    model.summary()
    # captrain0y=np.hstack((np.hstack((captrain0y[:,1:2],captrain0y[:,0:1])),captrain0y[:,2:4]))
    # captrain1y=np.hstack((np.hstack((captrain1y[:,1:2],captrain1y[:,0:1])),captrain1y[:,2:4]))

    trainre,result = train(model=model,eval_model=eval_model,
                                                  data=(traindata_tmp,traindata_tmp_y,test,test_y), args=args)
    return trainre,result

