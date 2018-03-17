

# Machine Learning in Action
# 程序清单 ch02
# 2018.3.17
# https://github.com/longxi7997
# logistic regression


from numpy import *

def loadDataSet():

    dataMat = []
    labelMat = []

    fr = open('./testSet.txt')

    for line in fr.readlines():

        lineArr = line.strip().split()
        # 加入常数项
        dataMat.append( [1.0 , float(lineArr[0]) , float(lineArr[1]) ] )
        labelMat.append( int(lineArr[2]) )

    return  dataMat , labelMat

def sigmod(inX):
    return  1.0/(1+ exp(-inX) )


# 梯度上升
def gradAscent( datamatIn , classLabels ):

    # mat

    # numpy 矩阵
    dataMatrix = mat( datamatIn )
    labelMat = mat( classLabels).transpose()

    m , n = shape( dataMatrix )
   # 步长
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))

    for k in range( maxCycles):
        h = sigmod( dataMatrix*weights )

        if k==0: print ( dataMatrix*weights)
        error = labelMat - h
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

# 随机梯度上升
def stocGradAscent0( dataMatrix , classLabels ):

    # array

    m , n = shape( dataMatrix )
   # 步长
    alpha = 0.01
    weights = ones(n)

    for i in range( m ):

        h = sigmod( sum(dataMatrix[i]*weights ) )
        # h = sigmod( dataMatrix[i] * weights)

        # print ( dataMatrix[i].transpose(), weights  , dot(dataMatrix[i],weights ) )
        error = (float)( classLabels[i] - h )
        weights = weights + alpha*error*dataMatrix[i]
    return weights


# 随机梯度上升
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # array

    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range( numIter ):
        dataIndex = list( range(m) )
        for i in range(m):
            # 动态步长
            alpha = 4.0/(1.0+j+i) + 0.01

            randIndex = int( random.uniform(0,len( dataIndex) ) )
            h = sigmod(sum(dataMatrix[ randIndex ] * weights))
            error = (float)(classLabels[randIndex] - h)
            weights = weights + alpha * error * dataMatrix[randIndex]
            del( dataIndex[randIndex] )
    return weights


def plotBestFit( weights ):
    import matplotlib.pyplot as plt
    dataMat , labelMat = loadDataSet()

    dataArr = array( dataMat )
    n = shape( dataArr)[0]
    xcord1 = [] ; ycord1 = []
    xcord2 = [] ; ycord2 = []

    for i in range(n):
        if int( labelMat[i]) == 1:
            xcord1.append( dataArr[i,1] ); ycord1.append(dataArr[i,2])
        else:
            xcord2.append( dataArr[i,1] ); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter( xcord1 ,ycord1 , s=30 , c='red' , marker='s')
    ax.scatter( xcord2, ycord2 , s=30 , c='green')

    x=arange( -3.0 , 3.0 , 0.1)
    y=(-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel("X1");plt.ylabel("X2")
    plt.show()


# 预测、分类
def classifyVector( inX , weights ):

    prob = sigmod( sum( inX*weights ) )
    if prob>0.5: return 1.0
    else: return 0.0

# 氙气病
def colicTest():
    ftTrain = open('./horseColicTraining.txt')
    ftTest = open('./horseColicTraining.txt')

    trainingSet = []
    trainingLabels= []

    for line in ftTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(21):
            lineArr.append( float(currLine[i]) )
        trainingSet.append( lineArr )
        trainingLabels.append( float(currLine[21]) )

    trainWeights = stocGradAscent1( array(trainingSet) , trainingLabels, 500 )

    errorCount = 0
    numTestVec =0.0

    for line in ftTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append( float(currLine[i]) )

        if int( classifyVector(array(lineArr) , trainWeights ) ) != int( float(currLine[21]) ) :
            errorCount +=1


    errorRate = (float(errorCount)/numTestVec )

    print ( "error Rate : " , errorRate )

    return  errorRate


# 多次测试求均值 ， 因为是随机取样，随机梯度下降
def multiTest():
    numTests = 10
    errorSum = 0.0

    for k in range( numTests ):
        errorSum += colicTest()

    print (  "%d , %f" %( numTests ,  errorSum/float(numTests) ) )




if __name__ == "__main__":

    dataArr , labelMat = loadDataSet()

    # weights = gradAscent( dataArr , labelMat )
    # plotBestFit( weights.getA() )


    # weights = stocGradAscent0( array(dataArr), labelMat)
    # plotBestFit( weights )

    # weights = stocGradAscent1(array(dataArr), labelMat)
    # plotBestFit(weights)
    #
    # print (weights)


    multiTest()














