'''
at new directory
'''

import PIL
import os
import numpy
import time
import staintools
import traceback
MAKE_NEW_LOG = 1
FILE_NAME = __file__


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


dataset = []


def Get_file_paths(_targetDirectory):
    path = _targetDirectory
    for i in os.listdir(path):
        if os.path.isfile(path + '/' + i):
            if i[-4:] != '.zip':
                dataset.append(path + '/' + i)
        else:
            Get_file_paths(path+'/'+i)


Get_file_paths('meta')
d = []
for i in dataset:
    if i[5] == 's':
        d.append(i.replace('meta/', ''))
print(d)
exit()
Get_file_paths('newdata')
Write_log('Loading images done!')
refImage = PIL.Image.open('dataset/Ref.png')
refImage = numpy.array(refImage)
refImage.astype('uint8')
normalizer = staintools.StainNormalizer('macenko')
normalizer.fit(refImage)
Write_log('Initailize normalizer done!')
filelist = []


def testHandle():
    for it, i in enumerate(dataset):
        try:
            if it % 100 == 0:
                Write_log('testHandle: {}/{}'.format(it, len(dataset))
                          + ' finished!')
        except:
            print('/0 error in '+FILE_NAME)
            exit()

        if os.path.exists('new/'+i.split('/')
                          [-2] + '/' + i.split('/')
                          [-2] + '_&_' + i.split('/')[-1]):
            continue
        try:
            if i.split('/')[-2] not in filelist:
                filelist.append(i.split('/')[-2])
                os.mkdir('new/{}'.format(filelist[-1]))
            image = PIL.Image.open(i)
            image = image.resize((224, 224), PIL.Image.ANTIALIAS)
            image = numpy.array(image)
            image.astype('uint8')
            image = normalizer.transform(image)
            image = PIL.Image.fromarray(image)
            image.save('new/'+i.split('/')
                       [-2] + '/'+i.split('/')
                       [-2] + '_&_' + i.split('/')[-1])
        except:
            print(i + ' error in ' + FILE_NAME)
            print(traceback.format_exc())
            exit()


print(len(dataset))
testHandle()

Write_log('Over!')

dataset = []


Get_file_paths('new')
for i in filelist:
    f = open('meta/{}.meta'.format(i), 'w')
    f.close()
for i in dataset:
    filename = ''
    for j in filelist:
        if j == i.split('/')[-2]:
            filename = j
    assert (filename != '')
    f = open('meta/{}.meta'.format(filename), 'a')
    f.writelines(i+'\n')
    f.close()
