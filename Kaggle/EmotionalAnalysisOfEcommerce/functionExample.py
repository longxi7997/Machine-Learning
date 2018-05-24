


# 去除连续的重复字串
import re
import jieba 
import pandas as pd
import numpy as np


def findMinmumSubstring( str ):

	strLen = len( str )

	punc = re.findall(r"[\W]+" , str)
	# print( punc )

	str = re.findall(r"[\w']+", str)
	# print( pd.unique(str) )
	uniStr = pd.unique( str )


	# 拼接字符串 和 标点符号
	resStr = ""
	for i in uniStr:
		# print (i)
		if ( str.index(i) < len( punc ) ):
			resStr += i + punc[ str.index( i ) ]
		else:
			resStr += i + " "
	# print( resStr )
	return resStr

if __name__ == '__main__':
	
	str = """买的放心，京东商城信a得过信得过信得过，买的放心买的放心，买的放心，买的放心，用的省心、安心、放心！"""
	str2 = "安装材料感觉像正品"

	str = pd.DataFrame( [ str , str2] )

	str = str.apply(  lambda row: findMinmumSubstring( row[0] ) , axis=1 )
	print( str )


	
