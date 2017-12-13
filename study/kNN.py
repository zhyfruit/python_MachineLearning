#coding=utf-8
from numpy import *
import operator
from os import listdir
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#shape函数可以查看矩阵的行、列数,shape[1]是一维长度，shape[0]是二维长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#sum而当加入axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#argsort()默认从小到大排列，并且返回的是从0开始的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3)) #array([[0,0,0],[0,0,0],···])
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()#去除首位的空格
        listFromLine = line.split('\t')#通过'\t'进行切片
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) #每列的最小值，参数0可以从列中选取最小值
    maxVals = dataSet.MAX(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape(0)
    normDataSet = dataSet - tile(minVals,(m,1)
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals



