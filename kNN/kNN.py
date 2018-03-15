

# Machine Learning in Action
# 程序清单 ch02
# 2018.3.14
# https://github.com/longxi7997
# kNN


from numpy import *
import  operator

import matplotlib
import matplotlib.pyplot as plt



def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile( inX , (dataSetSize , 1) )  - dataSet
    sqDifMat = diffMat**2


    sqDistance = sqDifMat.sum(axis=1)
    distances = sqDistance**0.5

    # 返回排序的索引值
    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted( classCount.items() , key=operator.itemgetter(1) ,reverse=True )
    return  sortedClassCount[0][0]


# 读取文件，切分，得到矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros( (numberOfLines,3) )

    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]

        classLabelVector.append( int(listFromLine[-1]) )
        index+=1

    return returnMat,classLabelVector

# 归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros( shape(dataSet) )
    m = dataSet.shape[0]

    normDataSet = dataSet - tile( minVals , (m,1) )
    normDataSet = normDataSet/tile( ranges , (m,1) )

    return normDataSet , ranges , minVals


# 正确率，测试集
def datingClassTest():

    hoRatio = 0.10

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()

    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    erroCount = 0.0
    for i in range(numTestVecs):

        classifierResult = classify0( normMat[i,:] , normMat[numTestVecs:m, :], datingLabels[numTestVecs:m] , 3 )

        print( "predict : %d , real answer : %d" % (classifierResult , datingLabels[i]) )

        if( classifierResult != datingLabels[i] ) : erroCount +=1.0


    print( "total error number : %d ,  error rate : %f , total : %d" % ( erroCount , erroCount/float(numTestVecs) , numTestVecs) )




if __name__ == "__main__":

    # group , labels = createDataSet()

    # classify0( [0,0] , group , labels , 3)

    datingClassTest()