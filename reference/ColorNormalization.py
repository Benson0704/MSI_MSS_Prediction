# small and big needed
'''
This script divide files to dataset, resize them to 512*512 and do the color normalization
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

MAKE_NEW_LOG = 1
FILE_NAME = __file__
TEST_NUM = 10
TRAIN_NUM = 40


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


testFiles = []
trainFiles = []


def Get_file_paths(_targetDirectory):
    path = _targetDirectory
    for i in os.listdir(path):
        if os.path.isfile(path + '/' + i):
            if 'ImageCrops Test' in path:
                testFiles.append(path + '/' + i)
            elif 'ImageCrops Train' in path:
                trainFiles.append(path + '/' + i)
            else:
                raise ValueError('no such files! Get_file_paths')
        else:
            Get_file_paths(path+'/'+i)


Get_file_paths('ImageCrops Test')
Get_file_paths('ImageCrops Train')
for a in testFiles:
    if 'Other' not in a:
        if '╡═╖╓╗п╧┘░й' not in a:  # low
            if '╓╨╖╓╗п╧┘░й' in a:  # middle
                print(a)
                exit()
exit()
Write_log('Loading images done!')
refImage = PIL.Image.open('dataset/Ref.png')
refImage = numpy.array(refImage)
refImage.astype('uint8')
normalizer = staintools.StainNormalizer('macenko')
normalizer.fit(refImage)
Write_log('Initailize normalizer done!')


def testHandle(small, big):
    for it, i in enumerate(testFiles[small:big]):
        try:
            if it % 100 == 0:
                Write_log('testHandle_{}: {}/{}'.format(small//(len(testFiles)//TEST_NUM), it, big-small)
                          + ' finished!')
        except:
            print('/0 error in '+FILE_NAME)

        if os.path.exists('dataset/test/MSS/' + i.split('/')
                          [3] + '_&_' + i.split('/')[-1]):
            continue
        if os.path.exists('dataset/test/MSI/' + i.split('/')
                          [3] + '_&_' + i.split('/')[-1]):
            continue
        try:
            image = PIL.Image.open(i)
            image = image.resize((224, 224), PIL.Image.ANTIALIAS)
            image = numpy.array(image)
            image.astype('uint8')
            image = normalizer.transform(image)
            image = PIL.Image.fromarray(image)
            if 'MSS' in i:
                image.save('dataset/test/MSS/' + i.split('/')
                           [3] + '_&_' + i.split('/')[-1])
            elif 'MSI' in i:
                image.save('dataset/test/MSI/' + i.split('/')
                           [3] + '_&_' + i.split('/')[-1])
            else:
                raise ValueError('no such files! testFiles')
        except:
            print(i + ' error in ' + FILE_NAME)


def trainHandle(small, big):
    for it, i in enumerate(trainFiles[small:big]):
        try:
            if it % 100 == 0:
                Write_log('trainHandle_{}: {}/{}'.format(small//(len(trainFiles)//TRAIN_NUM), it, big-small)
                          + ' finished!')
        except:
            print('/0 error in '+FILE_NAME)
        if os.path.exists('dataset/train/MSS/' + i.split('/')
                          [3] + '_&_' + i.split('/')[-1]):
            continue
        if os.path.exists('dataset/train/MSI/' + i.split('/')
                          [3] + '_&_' + i.split('/')[-1]):
            continue
        try:
            image = PIL.Image.open(i)
            image = numpy.array(image)
            image.astype('uint8')
            image = normalizer.transform(image)
            image = PIL.Image.fromarray(image)
            image = image.resize((224, 224), PIL.Image.ANTIALIAS)
            if 'MSS' in i:
                image.save('dataset/train/MSS/' + i.split('/')
                           [3] + '_&_' + i.split('/')[-1])
            elif 'MSI' in i:
                image.save('dataset/train/MSI/' + i.split('/')
                           [3] + '_&_' + i.split('/')[-1])
            else:
                raise ValueError('no such files! trainFiles')
        except:
            print(i + ' error in ' + FILE_NAME)


trainHandle(trainSmall, trainBig)
testHandle(testSmall, testBig)

Write_log('Over!')
