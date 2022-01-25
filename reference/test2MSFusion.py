'''
This script test MSS/MSI on 1 cases:
    low on low and middle on middle FUSION
'''
import scipy.stats
import torch
import torchvision
import PIL
import json
import os
import numpy
import time
import scipy
import random
import staintools
import threading
import scipy
from sklearn import metrics
torch.set_printoptions(profile='full')
hyper_parameters = open('dataset/hyper_parameters_json', 'r')
hyper_parameters = json.load(hyper_parameters)
MAKE_NEW_LOG = 1
FILE_NAME = __file__+'_'+hyper_parameters['task']


def Write_log(_string):
    global MAKE_NEW_LOG
    if MAKE_NEW_LOG:
        MAKE_NEW_LOG = 0
        logFile = open('logs/' + FILE_NAME + '.log', 'w')
        logFile.writelines('Mission: ' + FILE_NAME + '\n' +
                           time.asctime(time.localtime())+' Starts!\n')
        logFile.close()
    logFile = open('logs/' + FILE_NAME + '.log', 'a')
    logFile.writelines(time.asctime(time.localtime()) + ' ' + _string + '\n')
    logFile.close()


file0 = hyper_parameters['file0']
file1 = hyper_parameters['file1']
fileA0 = hyper_parameters['fileA0']
fileA1 = hyper_parameters['fileA1']
for i in range(len(file0)):
    file0[i] = 'meta/' + file0[i]
for i in range(len(file1)):
    file1[i] = 'meta/' + file1[i]
for i in range(len(fileA0)):
    fileA0[i] = 'meta/' + fileA0[i]
for i in range(len(fileA1)):
    fileA1[i] = 'meta/' + fileA1[i]
if hyper_parameters["initialized"] == 0:
    Write_log(
        'Warning: hyper-parameters not initialized')
else:
    Write_log(
        json.dumps(hyper_parameters))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    Write_log(
        'using GPU '+str(torch.cuda.current_device()))
else:
    Write_log(
        '  '+'warning: using CPU')
data0 = []
data1 = []
for i in file0:  # mss=1 msi=0
    f = open(i, 'r')
    for line in f.readlines():
        data0.append(line[:-1])
    f.close()
for i in file1:
    f = open(i, 'r')
    for line in f.readlines():
        data1.append(line[:-1])
    f.close()
testFilesMSI = data0
testFilesMSS = data1
Write_log('MSI: {}, MSS: {}'.format(len(data0), len(data1)))
L = []
F = []
P = {}
result = []
for i in range(len(testFilesMSS)):
    name = testFilesMSS[i].split('_&_')[0]
    F.append(testFilesMSS[i])
    if name in P:
        pass
    else:
        P[name] = 1
    testFilesMSS[i] = (testFilesMSS[i], 1)
    L.append(1)
for i in range(len(testFilesMSI)):
    name = testFilesMSI[i].split('_&_')[0]
    F.append(testFilesMSI[i])
    if name in P:
        pass
    else:
        P[name] = 0
    testFilesMSI[i] = (testFilesMSI[i], 0)
    L.append(0)

data = testFilesMSS+testFilesMSI


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, transforms):
        self.data_set = data_set
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        if index >= len(self.data_set):
            log.writelines(time.asctime(time.localtime()) +
                           '  '+"Error: __getitem__ out of range" + '\n')
        else:
            image_file, label = self.data_set[index]
            label_tensor = torch.Tensor([1])
            label_tensor[0] = label
            image = PIL.Image.open(image_file)
            image_tensor = self.transforms(image)
            image_tensor = image_tensor.type(torch.FloatTensor)
            return image_tensor, label, index


dataset = NewDataset(data, torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()]))
dataloader = torch.utils.data.dataloader.DataLoader(
    dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'])
torch.cuda.set_device(0)
net = torch.load("models/" + 'train2MS.py_Low' + '.model')
Write_log('using model: '+"models/" +
          FILE_NAME.replace('test', 'train') + '.model')
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, label, index in dataloader:
        data = data.to(device)
        predicted = net(data)
        predicted = torch.nn.functional.softmax(predicted, dim=1)
        predicted = torch.max(predicted, 1)[1]
        for i in range(len(index)):
            result.append(int(predicted[i]))


def calcAUC_byRocArea(labels, probs):
    # initialize
    P = 0
    N = 0
    for i in labels:
        if (i == 1):
            P += 1
        else:
            N += 1
    TP = 0
    FP = 0
    TPR_last = 0
    FPR_last = 0
    AUC = 0
    pair = zip(probs, labels)
    pair = sorted(pair, key=lambda x: x[0], reverse=True)
    i = 0
    while i < len(pair):
        if (pair[i][1] == 1):
            TP += 1
        else:
            FP += 1
        # maybe have the same probs
        while (i + 1 < len(pair) and pair[i][0] == pair[i+1][0]):
            i += 1
            if (pair[i][1] == 1):
                TP += 1
            else:
                FP += 1
        TPR = TP / P
        FPR = FP / N
        AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)
        TPR_last = TPR
        FPR_last = FPR
        i += 1
    return AUC


def calcAUC_byProb(labels, probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []
    for _, i in enumerate(labels):
        if (i == 1):
            P += 1
            pos_prob.append(probs[_])
        else:
            N += 1
            neg_prob.append(probs[_])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if (pos > neg):
                number += 1
            elif (pos == neg):
                number += 0.5
    return number / (N * P)


Write_log('result length: '+str(len(result)))
Write_log('result L: '+str(len(L)))
Write_log('result F: '+str(len(F)))
Write_log('tile AUC1 is: '+str(calcAUC_byRocArea(L, result)))
Write_log('tile AUC2 is: '+str(calcAUC_byProb(L, result)))
Pfinal = []
PLfinal = []
dict_data = {}
for i in range(len(F)):
    name = F[i].split('_&_')[0]
    if name in dict_data.keys():
        dict_data[name][1].append((F[i], L[i], result[i]))
    else:
        dict_data[name] = []
        dict_data[name].append(P[name])
        dict_data[name].append([])
        dict_data[name][1].append((F[i], L[i], result[i]))
Write_log('dict length ' + str(len(dict_data)))
d = []
for k in dict_data:
    real, li = dict_data[k]
    if len(li) <= 10:
        d.append(k)
for i in d:
    print('delete')
    del dict_data[i]
Write_log('edited dict length ' + str(len(dict_data)))
for k in dict_data:
    mss = 0
    msi = 0
    real, li = dict_data[k]
    for i in li:
        f, l, r = i
        if r == 1:
            mss += 1
        elif r == 0:
            msi += 1
        else:
            print('error')
            exit()
    Pfinal.append(mss/(mss+msi))
    PLfinal.append(real)
Write_log('method1: patient AUC is: ' +
          str(calcAUC_byRocArea(PLfinal, Pfinal)))
Write_log('method2: patient AUC is: ' + str(calcAUC_byProb(PLfinal, Pfinal)))
Write_log('method3: patient AUC is ' +
          str(metrics.roc_auc_score(PLfinal, Pfinal)))
auc = []
for i in range(500):
    pt = []
    plt = []
    for j in range(len(PLfinal)):
        x = random.randint(0, len(PLfinal)-1)
        pt.append(Pfinal[x])
        plt.append(PLfinal[x])
    auc.append(calcAUC_byProb(plt, pt))
auc1 = numpy.array(auc)
mean, std = auc1.mean(), auc1.std(ddof=1)
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
# asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss#asssssss
data0 = []
data1 = []
for i in fileA0:  # mss=1 msi=0
    f = open(i, 'r')
    for line in f.readlines():
        data0.append(line[:-1])
    f.close()
for i in fileA1:
    f = open(i, 'r')
    for line in f.readlines():
        data1.append(line[:-1])
    f.close()
testFilesMSI = data0
testFilesMSS = data1
Write_log('MSI: {}, MSS: {}'.format(len(data0), len(data1)))
for i in range(len(testFilesMSS)):
    name = testFilesMSS[i].split('_&_')[0]
    F.append(testFilesMSS[i])
    if name in P:
        pass
    else:
        P[name] = 1
    testFilesMSS[i] = (testFilesMSS[i], 1)
    L.append(1)
for i in range(len(testFilesMSI)):
    name = testFilesMSI[i].split('_&_')[0]
    F.append(testFilesMSI[i])
    if name in P:
        pass
    else:
        P[name] = 0
    testFilesMSI[i] = (testFilesMSI[i], 0)
    L.append(0)

data = testFilesMSS+testFilesMSI
dataset = NewDataset(data, torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()]))
dataloader = torch.utils.data.dataloader.DataLoader(
    dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'])
torch.cuda.set_device(1)
net = torch.load("models/" + 'train2MS.py_Middle' + '.model')
Write_log('using model: '+"models/" + 'train2MS.py_Middle' + '.model')
net.eval()
result1 = []
with torch.no_grad():
    correct = 0
    total = 0
    for data, label, index in dataloader:
        data = data.to(device)
        predicted = net(data)
        predicted = torch.nn.functional.softmax(predicted, dim=1)
        predicted = torch.max(predicted, 1)[1]
        for i in range(len(index)):
            result1.append(int(predicted[i]))
result = result+result1

Write_log('result length: '+str(len(result)))
Write_log('result L: '+str(len(L)))
Write_log('result F: '+str(len(F)))
Write_log('tile AUC1 is: '+str(calcAUC_byRocArea(L, result)))
Write_log('tile AUC2 is: '+str(calcAUC_byProb(L, result)))
for i in range(len(F)):
    name = F[i].split('_&_')[0]
    if name in dict_data.keys():
        dict_data[name][1].append((F[i], L[i], result[i]))
    else:
        dict_data[name] = []
        dict_data[name].append(P[name])
        dict_data[name].append([])
        dict_data[name][1].append((F[i], L[i], result[i]))
Write_log('dict length ' + str(len(dict_data)))
for k in dict_data:
    real, li = dict_data[k]
    if len(li) <= 10:
        d.append(k)
for i in d:
    print('delete')
    del dict_data[i]
Write_log('edited dict length ' + str(len(dict_data)))
for k in dict_data:
    mss = 0
    msi = 0
    real, li = dict_data[k]
    for i in li:
        f, l, r = i
        if r == 1:
            mss += 1
        elif r == 0:
            msi += 1
        else:
            print('error')
            exit()
    Pfinal.append(mss/(mss+msi))
    PLfinal.append(real)
Write_log('method1: patient AUC is: ' +
          str(calcAUC_byRocArea(PLfinal, Pfinal)))
Write_log('method2: patient AUC is: ' + str(calcAUC_byProb(PLfinal, Pfinal)))
Write_log('method3: patient AUC is ' +
          str(metrics.roc_auc_score(PLfinal, Pfinal)))
for i in range(500):
    pt = []
    plt = []
    for j in range(len(PLfinal)):
        x = random.randint(0, len(PLfinal)-1)
        pt.append(Pfinal[x])
        plt.append(PLfinal[x])
    auc.append(calcAUC_byProb(plt, pt))
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
