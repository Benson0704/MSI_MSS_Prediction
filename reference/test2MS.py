'''
This script test MSS/MSI on 3 cases:
    1.low+middle
    2.low
    3.middle
'''
import scipy.stats
import torch
import torchvision
import PIL
import json
import numpy
import time
import scipy
import random
import scipy
from sklearn import metrics
import matplotlib.pyplot
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


def paint_roc(label_list, predicted_list, output_str='test.png'):
    fpr, tpr, thersholds = metrics.roc_curve(
        label_list, predicted_list, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    Write_log('auc in roc painting is: {}'.format(roc_auc))
    matplotlib.pyplot.plot(
        fpr, tpr, '-', label='ROC (area = {0:.4f})'.format(roc_auc), lw=1)

    matplotlib.pyplot.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    matplotlib.pyplot.title('ROC Curve')
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.savefig(output_str.replace('.png', '.eps'), format='eps')
    matplotlib.pyplot.close()


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
            Write_log(time.asctime(time.localtime()) +
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
            Write_log('{}/{} completed!'.format(total, len(data)))
print(len(data))
print(len(probability))
print(len(label_list))
Write_log('now start using probability and label for tile')
Write_log('tile AUC using probability and label is ' +
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
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
Write_log('now start using result and label for tile')
for i, pr in enumerate(probability):
    if pr >= 0.5:
        probability[i] = 1
    else:
        probability[i] = 0
Write_log('tile AUC using result and label is ' +
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
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
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

Write_log('now start using probability and label for patient')
Write_log(PLfinal.__str__())
Write_log(Pfinal.__str__())
Write_log('patient AUC using probability and label is ' +
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
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))
Write_log('now start using result and label for patient')

for i, pr in enumerate(Pfinal):
    if pr >= 0.5:
        Pfinal[i] = 1
    else:
        Pfinal[i] = 0
Write_log(PLfinal.__str__())
Write_log(Pfinal.__str__())
Write_log('patient AUC using result and label is ' +
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
Write_log('(bootstrap 500) mean is ' + str(mean))
conf_intveral = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
Write_log('conf_intveral ' + str(conf_intveral))

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
Write_log('real0to0: {}, real0to1: {}'.format(
    real0to0, real0to1))
Write_log('real1to0: {}, real1to1: {}'.format(
    real1to0, real1to1))
P = real0to0 / (real0to0 + real0to1)
R = real0to0 / (real0to0 + real1to0)
Write_log('MSI pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
P = real1to1 / (real1to0 + real1to1)
R = real1to1 / (real0to1 + real1to1)
Write_log('MSS pricision: {}, recall: {}, F1: {}'.format(
    P, R, 2 * P * R / (P + R)))
