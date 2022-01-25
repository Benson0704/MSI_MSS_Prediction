'''
Description:
    This script calculates the F1, precistion, recall, AUC, bootstrap 500 mean and confidential interval of msi classifier and paints ROC curve
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
import matplotlib.pyplot
HYPER_PARAMETERS = open('../hyper_parameters.json', 'r')
HYPER_PARAMETERS = json.load(HYPER_PARAMETERS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(HYPER_PARAMETERS['GPU'])


def paint_roc(label_list, predicted_list, output_str='test.png'):
    fpr, tpr, thersholds = metrics.roc_curve(
        label_list, predicted_list, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('auc in roc painting is: {}'.format(roc_auc))
    matplotlib.pyplot.plot(
        fpr, tpr, '-', label='ROC (area = {0:.4f})'.format(roc_auc), lw=1)

    matplotlib.pyplot.xlim([-0.05, 1.05])
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.title('ROC Curve')
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.savefig(output_str.replace('.png', '.eps'), format='eps')
    matplotlib.pyplot.close()


'''
Now you can add your dataset as training data
data0 refers to those labeled as 'MSI'
data1 refers to those labeled as 'MSS'
those two python list should append the tile's path
additionally, MSI-MSS classifying is on patient level, so you need add patient name and '_&_' as prefix
    like: data0=['tom_&_1.png','patient_&_2.png']
'''
data0 = []
data1 = []
testFilesMSI = data0
testFilesMSS = data1
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
net = torch.load("models/" + 'msi' + '.model')
net.eval()
probability = []
label_list = []

with torch.no_grad():
    total = 0
    for image, label, index in dataloader:
        image = image.to(device)
        predicted = net(image)
        predicted = torch.nn.functional.softmax(predicted, dim=1)
        for i in range(len(index)):
            probability.append(float(predicted[i][1]))
            label_list.append(float(label[i]))
            total += 1
        predicted = torch.max(predicted, 1)[1]
        for i in range(len(index)):
            result.append(int(predicted[i]))
        if total % 2560 == 0:
            print('{}/{} completed!'.format(total, len(data)))

print('now start using probability and label for tile')
print('tile AUC using probability and label is ' +
      str(metrics.roc_auc_score(label_list, probability)))
paint_roc(label_list, probability, 'tile_probability.png')
auc = []
for i in range(500):
    pt = []
    plt = []
    for j in range(len(label_list)):
        x = random.randint(0, len(label_list)-1)
        pt.append(probability[x])
        plt.append(label_list[x])
    auc.append(metrics.roc_auc_score(plt, pt))
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
print('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
print('conf_intveral ' + str(conf_intveral))
print('now start using result and label for tile')
for i, pr in enumerate(probability):
    if pr >= 0.5:
        probability[i] = 1
    else:
        probability[i] = 0
print('tile AUC using result and label is ' +
      str(metrics.roc_auc_score(label_list, probability)))
paint_roc(label_list, probability, 'tile_result.png')
auc = []
for i in range(500):
    pt = []
    plt = []
    for j in range(len(label_list)):
        x = random.randint(0, len(label_list)-1)
        pt.append(probability[x])
        plt.append(label_list[x])
    auc.append(metrics.roc_auc_score(plt, pt))
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
print('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
print('conf_intveral ' + str(conf_intveral))
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
d = []
for k in dict_data:
    real, li = dict_data[k]
    if len(li) <= 10:
        d.append(k)
for i in d:
    del dict_data[i]
wrong = []
MSS = 0
MSI = 0
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

print('now start using probability and label for patient')
print(PLfinal.__str__())
print(Pfinal.__str__())
print('patient AUC using probability and label is ' +
      str(metrics.roc_auc_score(PLfinal, Pfinal)))
paint_roc(PLfinal, Pfinal, 'patient_probability.png')
auc = []
for i in range(500):
    pt = []
    plt = []
    for j in range(len(PLfinal)):
        x = random.randint(0, len(PLfinal)-1)
        pt.append(Pfinal[x])
        plt.append(PLfinal[x])
    auc.append(metrics.roc_auc_score(plt, pt))
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
print('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
print('conf_intveral ' + str(conf_intveral))
print('now start using result and label for patient')

for i, pr in enumerate(Pfinal):
    if pr >= 0.5:
        Pfinal[i] = 1
    else:
        Pfinal[i] = 0
print(PLfinal.__str__())
print(Pfinal.__str__())
print('patient AUC using result and label is ' +
      str(metrics.roc_auc_score(PLfinal, Pfinal)))
paint_roc(PLfinal, Pfinal, 'patient_result.png')
auc = []
for i in range(500):
    pt = []
    plt = []
    for j in range(len(PLfinal)):
        x = random.randint(0, len(PLfinal)-1)
        pt.append(Pfinal[x])
        plt.append(PLfinal[x])
    auc.append(metrics.roc_auc_score(plt, pt))
auc = numpy.array(auc)
mean, std = auc.mean(), auc.std(ddof=1)
print('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
print('conf_intveral ' + str(conf_intveral))

real0to0 = 0
real0to1 = 0
real1to0 = 0
real1to1 = 0
for i in range(len(result)):
    if result[i] == 0:
        if L[i] == 0:
            real0to0 += 1
        if L[i] == 1:
            real1to0 += 1
    if result[i] == 1:
        if L[i] == 0:
            real0to1 += 1
        if L[i] == 1:
            real1to1 += 1
print('real0to0: {}, real0to1: {}'.format(
    real0to0, real0to1))
print('real1to0: {}, real1to1: {}'.format(
    real1to0, real1to1))
P = real0to0 / (real0to0 + real0to1)
R = real0to0 / (real0to0 + real1to0)
print('MSI pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
P = real1to1 / (real1to0 + real1to1)
R = real1to1 / (real0to1 + real1to1)
print('MSS pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
