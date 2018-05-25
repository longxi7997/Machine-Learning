# 任务：
1、对京东电商的评论进行爬取 <br>
2、分析某一个品牌热水器的用户情感的倾向<br>
3、从评论文本中挖掘出该品牌热水器的优点与不足<br>
4、提炼不同品牌热水器的卖点<br>

----------

## day1:
 1. 看完了 《python数据分析与挖掘实战》 第十五章内容。 <br>
 2. 借用工具（如，八爪鱼等）爬取网页，不要再自己写了。自己得不偿失，现在很多软件已经做得很好了。<br> 
 3. 文档去重需要进行相似性比较，方法有：编辑距离（涉及到动态规划，要会写），SimHash，MinHash等方法，要了解思想。
 4. 机械压缩完成了部分代码，但是没来得及解决连续去重的问题（其实写了一部分，再注释测试其他代码的时候，不小心删了），明天或者后天有时间解决
 5. 短文本删除的实现还比较简单（幼稚），有时间看有没有其他的方法，而不是直接以文本长度来解决（估计没有）？
 6. 有时间深入了解模型构建的部分，原来只知道理论，没有实践和操作。
 7. 编程基础有，但是python用的太少了，各种API不熟，太耽误时间。



----------


# 问题总结 #



## jupyter notebook 缩进问题
四个空格改成 tab, 将下列代码加入到 notebook 中 <br> 
    
	%%javascript
    IPython.tab_as_tab_everywhere = function(use_tabs) {
    if (use_tabs === undefined) {
    use_tabs = true; 
    }
    
    // apply setting to all current CodeMirror instances
    IPython.notebook.get_cells().map(
    function(c) {  return c.code_mirror.options.indentWithTabs=use_tabs;  }
    );
    // make sure new CodeMirror instances created in the future also use this setting
    CodeMirror.defaults.indentWithTabs=use_tabs;
	
    };
    
    IPython.tab_as_tab_everywhere()


## 文本相似度计算
### 1. 编辑距离（ 莱文斯坦距离 Levenshtein ）
编辑距离是一个经典的动态规划题目，实现方式如下：
[https://leetcode.com/problems/edit-distance/description/](https://leetcode.com/problems/edit-distance/description/ "LeetCode 题目地址")
	
    class Solution 
    {
    
	    public:
	    int minDistance(string word1, string word2) 
	    {
	    
		    int len1 = word1.length();
		    int len2 = word2.length();
		    
		    int **matDP = new int* [ len1+1 ];
		    for ( int i=0; i<len1+1 ; i++ )
		    matDP[i] = new int[ len2+1 ];
		    
		    // 初始化
		    for ( int i=0 ; i< len1+1 ; i++ )
		    	matDP[i][0] = i;
		    for ( int i=0 ; i< len2+1 ; i++ )
		    	matDP[0][i] = i;
		    
		    
		    // DP 计算整个矩阵
		    for( int i =0 ; i <len1 ; i++ )
		    {
		    	for( int j=0 ; j<len2 ; j++ )
			    {
				    if( word1[i]== word2[j]  )
				    	matDP[i+1][j+1] = matDP[i][j];
				    else
				   		matDP[i+1][j+1] =  min( matDP[i][j] , min(matDP[i][j+1] , matDP[i+1][j] ) ) + 1;
			    }
		    }
		    
		    
		    return matDP[len1][len2];
	    }
    };
    
### 2. Jaro距离
[http://www.cnblogs.com/xlturing/p/6136690.html](http://www.cnblogs.com/xlturing/p/6136690.html "引用参考")


### 3. SimHash
不同于传统的hash，SimHash是一种LSH(局部敏感哈希)，即语义相似的文本，hash值也相近。可以采用抽屉原理粗过滤，减少候选匹配集进行加速。

1. **分词:** 把需要判断文本分词形成这个文章的特征单词。最后形成去掉噪音词的单词序列并为每个词加上权重，我们假权重分为5个级别（1~5）。比如：“ 美国“51区”雇员称内部有9架飞碟，曾看见灰色外星人 ” ==> 分词后为 “ 美国（4） 51区（5） 雇员（3） 称（1） 内部（2） 有（1） 9架（3） 飞碟（5） 曾（1） 看见（3） 灰色（4） 外星人（5）”，括号里是代表单词在整个句子里重要程度，数字越大越重要。
2. **hash:** 通过hash算法把每个词变成hash值，比如“美国”通过hash算法计算为 100101,“51区”通过hash算法计算为 101011。这样我们的字符串就变成了一串串数字，还记得文章开头说过的吗，要把文章变为数字计算才能提高相似度计算性能，现在是降维过程进行时
3. **加权:**通过 2步骤的hash生成结果，需要按照单词的权重形成加权数字串，比如“美国”的hash值为“100101”，通过加权计算为“4 -4 -4 4 -4 4”；“51区”的hash值为“101011”，通过加权计算为 “ 5 -5 5 -5 5 5”。
4. **合并:**把上面各个单词算出来的序列值累加，变成只有一个序列串。比如 “美国”的 “4 -4 -4 4 -4 4”，“51区”的 “ 5 -5 5 -5 5 5”， 把每一位进行累加， “4+5 -4+-5 -4+5 4+-5 -4+5 4+5” ==》 “9 -9 1 -1 1 9”。这里作为示例只算了两个单词的，真实计算需要把所有单词的序列串累加。
5. **降维（转成二进制，二值化）:**把4步算出来的 “9 -9 1 -1 1 9” 变成 0 1 串，形成我们最终的simhash签名。 如果每一位大于0 记为 1，小于0 记为 0。最后算出结果为：“1 0 1 0 1 1”。<br><br>
 ![](https://i.imgur.com/yN53irb.png)<br>

6. **相似度度量：**汉明距离，一般取阈值3。
7. **存储：**
	1. 将一个64位的simhash签名拆分成4个16位的二进制码。（图上红色的16位）
	1.  分别拿着4个16位二进制码查找当前对应位置上是否有元素。（放大后的16位）
	1.  对应位置没有元素，直接追加到链表上；对应位置有则直接追加到链表尾端。（图上的 S1 — SN）
 ![](https://i.imgur.com/8nrbJMX.png)

8. **查找：**
	1. 将需要比较的simhash签名拆分成4个16位的二进制码。
	1. 分别拿着4个16位二进制码每一个去查找simhash集合对应位置上是否有元素。
	1. 如果有元素，则把链表拿出来顺序查找比较，直到simhash小于一定大小的值，整个过程完成。
	1. 在去重时，因为汉明距离小于3则为重复文本，那么如果存在simhash相似的文本，对于四段simhash则至少有一段simhash是相同的，所以在去重时对于待判断文本D，如果D中每一段的simhash都没有相同的，那么D为无重复文本。
	 
9. **原理：**借鉴hashmap算法找出可以hash的key值，因为我们使用的simhash是局部敏感哈希，这个算法的特点是只要相似的字符串只有个别的位数是有差别变化。那这样我们可以推断两个相似的文本，至少有16位的simhash是一样的。具体选择16位、8位、4位，大家根据自己的数据测试选择，虽然比较的位数越小越精准，但是空间会变大。分为4个16位段的存储空间是单独simhash存储空间的4倍。
10. **实现：** 原博主利用Murmur3作为字符串的64位哈希值，用Java和spark分别实现了一个simhash的版本
将源码放在了github上，如下链接：
[https://github.com/xlturing/Simhash4J](https://github.com/xlturing/Simhash4J)

	其中利用了结巴作为文本的分词工具，Murmur3用来产生64位的hashcode。另外根据上述存储方式，进行了simhash分段存储，提高搜索速度，从而进行高效查重

11. **应用（优缺点）：**simhash从最一开始用的最多的**场景便是大规模文本的去重**，对于爬虫从网上爬取的大规模语料数据，我们需要进行预处理，删除重复的文档才能进行后续的文本处理和挖掘，那么利用simhash是一种不错的选择，其计算复杂度和效果都有一个很好的折中。<br>
	但是在实际应用过程中，也发现一些badcase，**完全无关的文本正好对应成了相同的simhash**，精确度并不是很高，而且simhash**更适用于较长的文本**，但是在大规模语料进行去重时，simhash的计算速度优势还是很不错的。


## 词的向量表示法
例如，king-queen 和 man-woman 的关系
引申有:字向量、句向量、文档向量

### 1. 相关知识: n-gram 模型
NLP领域通常有两种主流思想：基于语法规则的方法（以前用的很多）和基于统计的方法（现在的常用方法），两者的具体发展历程可以阅读 吴军老师的《数学之美》。n-gram就是考虑一个词在当前上下文出现的条件概率，然后计算这段话中，每个词的一个联合概率，通过最大似然，得到结果。n代表了上下文的范围（马尔科夫链的长度）。
其中，有一些小的 tricks ：<br>
- 为了头尾的统一处理，会在头尾加上一个标记；<br>
- 为了避免0、1的特殊概率情况，会进行平滑，通常有拉普拉斯平滑；<br>
- 为了防止联合概率连乘导致的类“梯度消失问题”，进行取对数处理. <br>


### 2. One-hot
具体的实现方式，可以给每个词分配一个ID；用Hash值表示一个词语。（ 备注：候选查看 One-hot的压缩，矩阵的稀疏表示 ）

这种方式在 SVM 、 最大熵和CRF中很好用（阅读）
问题：维数过大；语义相似的词之间，距离不一定相同，即词之间没有相关性；


### 3. 文档表示
选取特征词来代表该文档，通常通过计算 tf-idf 来得到每个词的权重，取信息量高的词。然后通过聚类，得到相似主题的文档。 <br>
tricks：注意每个文档长度不同，需要归一化，即调整tf的计算方式（概率）。

### 4. Distributed Representation
相似的词语，距离也相近；维度低。

后续。。。。
http://www.cnblogs.com/xlturing/p/6136690.html
http://www.cnblogs.com/xlturing/default.html?page=1



