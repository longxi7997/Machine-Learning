# Machine Learning in Action
# 程序清单 3.1
# 2018.3.11
# https://github.com/longxi7997
# 绘制决策树


#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


decisionNode = dict(boxstyle="sawtooth" , fc="0.8")
leafNode = dict(boxstyle="round4" , fc="0.8")
arrow_args = dict( arrowstyle="<-")


# 绘制节点
def plotNode(nodeTxt , centerPt , parentPt , nodeType ):
    createPlot.ax1.annotate(nodeTxt , xy=parentPt , xycoords='axes fraction', xytext=centerPt , textcoords='axes fraction',\
                            va="center" , ha="center" , bbox=nodeType , arrowprops=arrow_args)
# 线上的文字
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    # createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    createPlot.ax1.text(xMid, yMid, txtString, rotation=0)

#
# # 创建画布
# def createPlot():
#     fig = plt.figure(1,facecolor="white")
#     fig.clf()
#
#     # 画布
#     createPlot.axl = plt.subplot(111,frameon=False)
#     plotNode("决策节点", (0.5,0.1) , (0.1,0.5) , decisionNode )
#     plotNode("叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

# 获取叶子节点个数 , 计算 X的坐标
def getNumLeafs( myTree ):
    numLeafs = 0
    firstStr = list( myTree.keys() )[0]

    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1

    return  numLeafs

# 获取树的高度
def getTreeDepth( myTree ):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth =1

        if thisDepth > maxDepth : maxDepth=thisDepth

    return  maxDepth

# 绘制树的局部结构
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list( myTree.keys() )[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    plotMidText(cntrPt, parentPt, nodeTxt)

    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 创建画布 ， 绘制树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


# 产生样例树
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


if __name__ == "__main__":
    # createPlot()

    myTree = retrieveTree(1)

    createPlot( myTree )