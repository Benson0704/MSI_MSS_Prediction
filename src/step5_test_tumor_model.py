'''
Description:
    This script calculates the F1, precistion, recall, AUC, bootstrap 500 mean and confidential interval of tumor classifier
Author:
    Benson0704@outlook.com
Version:
    origin
WARNING:
    If model re-training is needed, dataset must be added manually as comments below
'''
import scipy.stats
import torch
import torchvision
import PIL
import json
import numpy
import scipy
import random
from sklearn import metrics
HYPER_PARAMETERS = open('../hyper_parameters.json', 'r')
HYPER_PARAMETERS = json.load(HYPER_PARAMETERS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(HYPER_PARAMETERS['GPU'])
'''
Now you can add your dataset as training data
    data0 refers to those labeled as 'other'
    data1 refers to those labeled as 'tumor'
OR
    data0 refers to those labeled as 'other'
    data1 refers to those labeled as 'PDA'
    data2 refers to those labeled as 'WDA'
those two python list should append the tile's path
    like: data0=['1.png','2.png']
'''
data0 = []
data1 = []
data2 = []

random.shuffle(data0)
random.shuffle(data1)
random.shuffle(data2)

testFilesOther = data0
testFilesLow = data1
testFilesMiddle = data2
L = []
for i in range(len(testFilesOther)):
    testFilesOther[i] = (testFilesOther[i], 0)
    L.append(0)
for i in range(len(testFilesLow)):
    testFilesLow[i] = (testFilesLow[i], 1)
    L.append(1)
for i in range(len(testFilesMiddle)):
    testFilesMiddle[i] = (testFilesMiddle[i], 2)
    L.append(2)
result = []
probability = []
data = testFilesOther+testFilesLow+testFilesMiddle


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, transforms):
        self.data_set = data_set
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
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
    dataset, batch_size=HYPER_PARAMETERS['BS'], num_workers=HYPER_PARAMETERS['NW'])
net = torch.load("models/"+'tumor' + '.model')
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
            for j in range(3):
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


real0to0 = 0
real0to1 = 0
real0to2 = 0
real1to0 = 0
real1to1 = 0
real1to2 = 0
real2to0 = 0
real2to1 = 0
real2to2 = 0
for i in range(len(result)):
    if result[i] == 0:
        if L[i] == 0:
            real0to0 += 1
        if L[i] == 1:
            real1to0 += 1
        if L[i] == 2:
            real2to0 += 1
    if result[i] == 1:
        if L[i] == 0:
            real0to1 += 1
        if L[i] == 1:
            real1to1 += 1
        if L[i] == 2:
            real2to1 += 1
    if result[i] == 2:
        if L[i] == 0:
            real0to2 += 1
        if L[i] == 1:
            real1to2 += 1
        if L[i] == 2:
            real2to2 += 1
print('real0to0: {}, real0to1: {}, real0to2: {}'.format(
    real0to0, real0to1, real0to2))
print('real1to0: {}, real1to1: {}, real1to2: {}'.format(
    real1to0, real1to1, real1to2))
print('real2to0: {}, real2to1: {}, real2to2: {}'.format(
    real2to0, real2to1, real2to2))
P = real0to0 / (real0to0 + real1to0 + real2to0)
R = real0to0 / (real0to0 + real0to1 + real0to2)
print('Other pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
P = real1to1 / (real0to1 + real1to1 + real2to1)
R = real1to1 / (real1to0 + real1to1 + real1to2)
print('Low pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
P = real2to2 / (real0to2 + real1to2 + real2to2)
R = real2to2 / (real2to0 + real2to1 + real2to2)
print('Middle pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2*P*R / (P + R)))
P = (real1to1 + real1to2 + real2to2 + real2to1) / \
    (real1to1 + real1to2 + real2to2 + real2to1 + real0to1+real0to2)
R = (real1to1+real1to2+real2to2+real2to1) / \
    (real1to1 + real1to2 + real2to2 + real2to1 + real1to0 + real2to0)

print('Tumor(Low+Middle) pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2*P*R / (P + R)))
PLfinal = []
Pfinal = []
for i in range(len(result)):
    if L[i] > 0:
        PLfinal.append(1)
    else:
        PLfinal.append(0)
    Pfinal.append(probability[i][1]+probability[i][2])
print('method1: tile AUC is: ' + str(calcAUC_byProb(PLfinal, Pfinal)))
print('method2: tile AUC is ' +
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
print('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
print('conf_intveral ' + str(conf_intveral))
