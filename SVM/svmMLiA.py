

# Machine Learning in Action
# 程序清单 ch04
# 2018.3.24
# https://github.com/longxi7997
# SVM

#  reference ； https://github.com/apachecn/MachineLearning/blob/master/docs/6.%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.md


import random
from numpy import *


# 载入数据 , 类别为 -1 ， 1 而不是 0 ，1
def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat


def selectJrand( i ,m ):
	j=i
	while (j==i):
		j = int (random.uniform(0,m))

	return j

def clipAlpha( aj, H, L):
	if aj >H:
		aj = H
	if L>aj:
		aj=L
	return aj


# 简化的 SMO 算法 ; toler 容错率  ，C惩罚因子
def smoSimple( dataMatIn , classLabels , C , toler , maxIter ):
	"""smoSimple

	Args:
		dataMatIn	特征集合
		classLabels  类别标签
		C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
			控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
			可以通过调节该参数达到不同的结果。
		toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
		maxIter 退出前最大的循环次数
	Returns:
		b	   模型的常量值
		alphas  拉格朗日乘子
	"""
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()

	b = 0; m,n = shape(dataMatrix)
	
	alphas = mat(zeros((m,1)))

	# 没有任何alpha改变的情况下遍历数据的次数
	iter = 0
	while (iter < maxIter):
		# 监测 变量是否被优化
		alphaPairsChanged = 0
		for i in range(m):

			# 预测类别   y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n] !!!!!!
			fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
			# 预测结果和真值的误差
			Ei = fXi - float(labelMat[i])

			# 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
			# 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
			# 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
			'''
			# 检验训练样本(xi, yi)是否满足KKT条件
			yi*f(i) >= 1 and alpha = 0 (outside the boundary)
			yi*f(i) == 1 and 0<alpha< C (on the boundary)
			yi*f(i) <= 1 and alpha = C (between the boundary)
			'''


			# 是否继续优化 ； 并且确保满足 KKT 条件 。 如果 alpha ==0 or alpha == C ，则 akpha 已经在边界上了，不需要再优化了。
			if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
				 # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
				j = selectJrand(i,m)
				# 预测j的结果
				fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();

				# L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
				# labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				# 如果相同，就没法优化了
				if L==H: 
					print ("L==H"); 
					continue

				# eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
				# 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
				eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
				if eta >= 0: 
					print ("eta>=0"); 
					continue

				# 计算出一个新的alphas[j]值
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				# 并使用辅助函数，以及L和H对其进行调整
				alphas[j] = clipAlpha(alphas[j],H,L)
				# 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
				if (abs(alphas[j] - alphaJold) < 0.00001): 
					print ("j not moving enough"); 
					continue

				# 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
																		#the update is in the oppostie direction

				# 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
				# w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
				# 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
				# 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
				if (0 < alphas[i]) and (C > alphas[i]): b = b1
				elif (0 < alphas[j]) and (C > alphas[j]): b = b2
				else: b = (b1 + b2)/2.0
				alphaPairsChanged += 1
				print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )

		# 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
		# 知道更新完毕后，iter次循环无变化，才推出循环。
		if (alphaPairsChanged == 0): iter += 1
		else: iter = 0
		print ("iteration number: %d" % iter)
	return b,alphas



if __name__ == '__main__':
	
	dataArr , labelArr = loadDataSet( "./testSet.txt" )


	b ,alphas = smoSimple( dataArr , labelArr , 0.6 , 0.001 , 40)	

	print(b)