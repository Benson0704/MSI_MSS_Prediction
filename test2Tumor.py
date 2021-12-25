'''
This script test other-tumor
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
for i in range(len(file0)):
    file0[i] = 'meta/' + file0[i]
for i in range(len(file1)):
    file1[i] = 'meta/' + file1[i]
if hyper_parameters["initialized"] == 0:
    Write_log(
        'Warning: hyper-parameters not initialized')
else:
    Write_log(
        json.dumps(hyper_parameters))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(hyper_parameters['GPU'])
if torch.cuda.is_available():
    Write_log(
        'using GPU '+str(torch.cuda.current_device()))
else:
    Write_log(
        '  '+'warning: using CPU')
data0 = []
data1 = []
for i in file0:
    f = open(i, 'r')
    for line in f.readlines():
        data0.append(line[:-1])
    f.close()
for i in file1:
    f = open(i, 'r')
    for line in f.readlines():
        data1.append(line[:-1])
    f.close()
testFilesOther = data0
testFilesTumor = data1
Write_log('Other: {}, Tumor: {}'.format(
    len(data0), len(data1)))
L = []
for i in range(len(testFilesOther)):
    testFilesOther[i] = (testFilesOther[i], 0)
    L.append(0)
for i in range(len(testFilesTumor)):
    testFilesTumor[i] = (testFilesTumor[i], 1)
    L.append(1)
result = []
probability = []
data = testFilesOther+testFilesTumor


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
net = torch.load("models/" + FILE_NAME.replace('test', 'train') + '.model')
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
        for i in range(len(index)):
            b = []
            for j in range(2):
                b.append(float(predicted[i][j]))
            probability.append(b)
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
Write_log('L length: ' + str(len(L)))
Write_log('probability length: ' + str(len(probability)))
real0to0 = 0
real0to1 = 0
real1to0 = 0
real1to1 = 0
for i in range(len(result)):
    if result[i] == 0:
        if L[i] == 0:
            real0to0 += 1
        if L[i] == 1:
            real0to1 += 1
    if result[i] == 1:
        if L[i] == 0:
            real1to0 += 1
        if L[i] == 1:
            real1to1 += 1
Write_log('real0to0: {}, real0to1: {}'.format(
    real0to0, real0to1))
Write_log('real1to0: {}, real1to1: {}'.format(
    real1to0, real1to1))
P = real0to0 / (real0to0 + real0to1)
R = real0to0 / (real0to0 + real1to0)
Write_log('Other pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
P = real1to1 / (real1to0 + real1to1)
R = real1to1 / (real0to1 + real1to1)
Write_log('Tumor pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
PLfinal = []
Pfinal = []
for i in range(len(result)):
    if L[i] > 0:
        PLfinal.append(1)
    else:
        PLfinal.append(0)
    Pfinal.append(probability[i][1])
Write_log('method1: tile AUC is: ' + str(calcAUC_byProb(PLfinal, Pfinal)))
Write_log('method2: tile AUC is ' +
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
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
