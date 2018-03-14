# Machine Learning in Action
# 程序清单 3.1
# 2018.3.11
# https://github.com/longxi7997
# 创建决策树


from math import log
import operator

import treePlotter



# 计算 熵 ,  只注意标签的散度
def calcShannonEnt (dataset):

    numEntries = len(dataset);
    labelCounts = {}
    for featVec in dataset:

        # 最后一列
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:
        prob = float( labelCounts[key] )/numEntries
        shannonEnt -= prob*log(prob,2)

    return  shannonEnt


# 根据给定的属性、属性值 划分数据集
def splitDataSet(dataSet , axis , value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 删除该属性
            reducedFeatVec = featVec[:axis]
            # 注意 extend 和 append 的用法区别
            reducedFeatVec.extend( featVec[axis+1:])
            retDataSet.append( reducedFeatVec )
    return retDataSet


# 选择最好的属性划分， 即 熵增益最大的属性
def chooseBestFeatureToSplit(dataSet):

    numFeatures = len(dataSet[0])-1

    baseEntropy = calcShannonEnt( dataSet )
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range( numFeatures ):

        # 提取每一列属性的全部值
        featList = [example[i] for example in dataSet]
        # 去重
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i


    return  bestFeature


# 当划分之后标签仍不一样时， 返回标签最多的作为最终的判断结果
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1

    # 将字典 转换为 可迭代对象 , 相比于 iterms ， iteritmes 效率更高
    sortedClassCount = sorted( classCount.iteritems() , key=operator.itemgetter(1) , reverse=True )

    return sortedClassCount[0][0]


# 创建决策树 ， 其中 labels 参数可有可无 ， 是 标签的一个映射
def createTree( dataSet , labels ):
    classList = [ example[-1] for example in dataSet]

    # 标签一样的时候， 停止划分
    if classList.count( classList[0]) ==len(classList):
        return classList[0]

    # 没有属性， 只有标签的时候
    if len(dataSet[0]) == 1:
        return majorityCnt( classList )

    bestFeat = chooseBestFeatureToSplit( dataSet )

    bestFeatLabel = labels[bestFeat]
    myTree = { bestFeatLabel:{} }
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    # 循环创建决策树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree( splitDataSet(dataSet , bestFeat , value) , subLabels  )

    return  myTree

# 使用决策树进行分类 , featLabel 指定样本属性的顺序
def classify(inputTree,featLabels,testVec):
    firstStr = list( inputTree.keys() )[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 如果是字典，继续
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# 产生样例数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0, 1, 'no']]

    labels = ['no surfacing' , 'flippers' ]
    return dataSet , labels


# 序列化决策树，存储下来
# 存储树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb' )
    pickle.dump(inputTree, fw)
    fw.close()
# 解析树
def grabTree(filename):
    import pickle
    fr = open(filename , 'rb')
    return pickle.load(fr)


if __name__ == '__main__':

    # myDat, labels = createDataSet()
    #
    # myTree = treePlotter.retrieveTree(0)
    #
    # cla = classify( myTree , labels , [1,1] )
    #
    # print(cla)
    # # print( myTree[ myTree.key()] )


    # 以隐形眼镜数据为例，进行操作
    fr = open( 'lenses.txt' )
    lenses = [ inst.strip().split('\t') for inst in fr.readlines() ]
    lensesLabels = ['age' , 'prescript' , 'astigmatic' , 'tearRate']

    lensesTree = createTree( lenses , lensesLabels )

    storeTree( lensesTree , "classifierStorage.txt")

    mytree = grabTree("classifierStorage.txt")

    # print( lensesTree )
    print( mytree)

    treePlotter.createPlot( lensesTree )


