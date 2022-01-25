'''
'╡═╖╓╗п╧┘░й' : low
'╓╨╖╓╗п╧┘░й' : middle
this file will create 12 metas for these 14 needs:
    part A: all files
    1.test meta other
    2.test meta low
    3.test meta middle
    4.train meta other
    5.train meta low
    6.train meta other

    part B: exclude 'Other' file
    1.test meta msi low
    2.train meta msi low
    3.test meta mss low
    4.train meta mss low
    5.test meta msi middle
    6.train meta msi middle
    7.test meta mss middle
    8.train meta mss middle
'''
import json
import os
import numpy
import time
MAKE_NEW_LOG = 1
FILE_NAME = __file__
a = 0
f = open('meta/testOtherMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/testOtherMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/testLowMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/testLowMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/testMiddleMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/testMiddleMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainOtherMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainOtherMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainLowMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainLowMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainMiddleMSS.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
f = open('meta/trainMiddleMSI.meta', 'r')
for line in f.readlines():
    a += 1
f.close()
print(a)
exit()


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
testOtherMSS = []
testLowMSS = []
testMiddleMSS = []
trainOtherMSS = []
trainLowMSS = []
trainMiddleMSS = []
testOtherMSI = []
testLowMSI = []
testMiddleMSI = []
trainOtherMSI = []
trainLowMSI = []
trainMiddleMSI = []


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
for i in testFiles:
    if 'Other' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testOtherMSS.append(i)
    if 'Other' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testOtherMSI.append(i)
    if '╡═╖╓╗п╧┘░й' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testLowMSS.append(i)
    if '╡═╖╓╗п╧┘░й' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testLowMSI.append(i)
    if '╓╨╖╓╗п╧┘░й' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testMiddleMSS.append(i)
    if '╓╨╖╓╗п╧┘░й' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        testMiddleMSI.append(i)
print(len(testOtherMSS), len(testOtherMSI), len(
    testLowMSS), len(testLowMSI), len(testMiddleMSS), len(testMiddleMSI), len(testFiles))
for i in trainFiles:
    if 'Other' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainOtherMSS.append(i)
    if 'Other' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainOtherMSI.append(i)
    if '╡═╖╓╗п╧┘░й' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainLowMSS.append(i)
    if '╡═╖╓╗п╧┘░й' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainLowMSI.append(i)
    if '╓╨╖╓╗п╧┘░й' in i and 'MSS' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainMiddleMSS.append(i)
    if '╓╨╖╓╗п╧┘░й' in i and 'MSI' in i:
        i = i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1]
        trainMiddleMSI.append(i)
print(len(trainOtherMSS), len(trainOtherMSI), len(
    trainLowMSS), len(trainLowMSI), len(trainMiddleMSS), len(trainMiddleMSI), len(trainFiles))
Write_log('Loading images done!')
testFilesMSI = []
testFilesMSS = []
trainFilesMSI = []
trainFilesMSS = []

f = open('meta/oldmetas/testFilesMSI.meta', 'r')
for line in f.readlines():
    testFilesMSI.append(line[:-1])
f.close()
f = open('meta/oldmetas/testFilesMSS.meta', 'r')
for line in f.readlines():
    testFilesMSS.append(line[:-1])
f.close()
f = open('meta/oldmetas/trainFilesMSI.meta', 'r')
for line in f.readlines():
    trainFilesMSI.append(line[:-1])
f.close()
f = open('meta/oldmetas/trainFilesMSS.meta', 'r')
for line in f.readlines():
    trainFilesMSS.append(line[:-1])

testOtherMSSnew = []
testLowMSSnew = []
testMiddleMSSnew = []
trainOtherMSSnew = []
trainLowMSSnew = []
trainMiddleMSSnew = []
testOtherMSInew = []
testLowMSInew = []
testMiddleMSInew = []
trainOtherMSInew = []
trainLowMSInew = []
trainMiddleMSInew = []


def check():
    if iter % 1000 == 0:
        Write_log(str(iter)+' done')


iter = 0
for i in testFilesMSI:
    iter += 1
    check()
    tmp = i.split('/')
    part0 = tmp[-1].split('_&_')[0]
    part1 = tmp[-1].split('_&_')[1]
    if part0 + '/Other/' + part1 in testOtherMSI:
        testOtherMSInew.append(i)
    if part0 + '/╡═╖╓╗п╧┘░й/' + part1 in testLowMSI:
        testLowMSInew.append(i)
    if part0 + '/╓╨╖╓╗п╧┘░й/' + part1 in testMiddleMSI:
        testMiddleMSInew.append(i)
for i in trainFilesMSI:
    iter += 1
    check()
    tmp = i.split('/')
    part0 = tmp[-1].split('_&_')[0]
    part1 = tmp[-1].split('_&_')[1]
    if part0 + '/Other/' + part1 in trainOtherMSI:
        trainOtherMSInew.append(i)
    if part0 + '/╡═╖╓╗п╧┘░й/' + part1 in trainLowMSI:
        trainLowMSInew.append(i)
    if part0 + '/╓╨╖╓╗п╧┘░й/' + part1 in trainMiddleMSI:
        trainMiddleMSInew.append(i)
for i in testFilesMSS:
    iter += 1
    check()
    tmp = i.split('/')
    part0 = tmp[-1].split('_&_')[0]
    part1 = tmp[-1].split('_&_')[1]
    if part0 + '/Other/' + part1 in testOtherMSS:
        testOtherMSSnew.append(i)
    if part0 + '/╡═╖╓╗п╧┘░й/' + part1 in testLowMSS:
        testLowMSSnew.append(i)
    if part0 + '/╓╨╖╓╗п╧┘░й/' + part1 in testMiddleMSS:
        testMiddleMSSnew.append(i)
for i in trainFilesMSS:
    iter += 1
    check()
    tmp = i.split('/')
    part0 = tmp[-1].split('_&_')[0]
    part1 = tmp[-1].split('_&_')[1]
    if part0 + '/Other/' + part1 in trainOtherMSS:
        trainOtherMSSnew.append(i)
    if part0 + '/╡═╖╓╗п╧┘░й/' + part1 in trainLowMSS:
        trainLowMSSnew.append(i)
    if part0 + '/╓╨╖╓╗п╧┘░й/' + part1 in trainMiddleMSS:
        trainMiddleMSSnew.append(i)
print(len(testOtherMSSnew), len(testOtherMSInew), len(
    testLowMSSnew), len(testLowMSInew), len(testMiddleMSSnew), len(testMiddleMSInew), len(testFilesMSS+testFilesMSI))
print(len(trainOtherMSSnew), len(trainOtherMSInew), len(
    trainLowMSSnew), len(trainLowMSInew), len(trainMiddleMSSnew), len(trainMiddleMSInew), len(trainFilesMSS+trainFilesMSI))
f = open('meta/testOtherMSS.meta', 'w')
for i in testOtherMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/testOtherMSI.meta', 'w')
for i in testOtherMSInew:
    f.writelines(i+'\n')
f.close()
f = open('meta/testLowMSS.meta', 'w')
for i in testLowMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/testLowMSI.meta', 'w')
for i in testLowMSInew:
    f.writelines(i+'\n')
f.close()
f = open('meta/testMiddleMSS.meta', 'w')
for i in testMiddleMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/testMiddleMSI.meta', 'w')
for i in testMiddleMSInew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainOtherMSS.meta', 'w')
for i in trainOtherMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainOtherMSI.meta', 'w')
for i in trainOtherMSInew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainLowMSS.meta', 'w')
for i in trainLowMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainLowMSI.meta', 'w')
for i in trainLowMSInew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainMiddleMSS.meta', 'w')
for i in trainMiddleMSSnew:
    f.writelines(i+'\n')
f.close()
f = open('meta/trainMiddleMSI.meta', 'w')
for i in trainMiddleMSInew:
    f.writelines(i+'\n')
f.close()
