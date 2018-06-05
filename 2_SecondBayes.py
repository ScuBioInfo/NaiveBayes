#coding=utf-8
#
import random
import sys
import operator
import copy
from math import log

def tidy(box):
	'''
	输入：记录列表
	输出：整理后的记录的嵌套列表
	将每一条记录用列表的split()方法去掉中间的","	
	声明两个列表逐步建立嵌套列表
	'''
	box_mid = list()
	box_aim = list()
	for i in box:
		i = i.split(",")
		#分割后进行解包
		[a,b,c,x] = i
		box_mid = [float(a),float(b),float(c),x[:-1]]
		box_aim.append(box_mid)
	return box_aim

'''
对属性值进行离散化位两个区间
以信息增益为标准，对每个属性选择信息增益最大的区间划分点
'''

def count_frequency(input_attribute_list):
	'''
	功能:
		计算输入样本集各个类标号的频率
	输入:
		列表:样本集组成的嵌套列表
	输出:
		字典:{类标号取值:频率}
		各个类标号在所有输入样本中出现的频率(频次/所有样本总数)
	'''
	#输入的嵌套列表的大小：样本总数
	total_count = len(input_attribute_list)
	obj_dict = dict()	
	for i in input_attribute_list:
		if i[3] not in obj_dict.keys():
			obj_dict[i[3]] = 0
			obj_dict[i[3]] += 1
		else:
			obj_dict[i[3]] += 1
	#测试:频次统计
	print("类标号频次对应的字典为"+str(obj_dict))	
	for j in obj_dict.keys():	
		#这里注意必须使用float,否则除法会自动取整
		obj_dict[j] = round(float(obj_dict[j]/float(total_count)),3)
	#测试:频率统计
	print("类标号频率对应字典为"+str(obj_dict))
	return obj_dict

def entropy(input_attribute_list):
	'''
	功能:
		计算输入样本集input_attribute_list的信息熵
	输入:
		列表:样本组成的列表
	输出:
		float:样本集对应的信息熵
	'''
	#新建一个字典用于存放count_frequency返回的{类标号:频率}
	mid_dict = count_frequency(input_attribute_list)
	entro = 0
	for i in mid_dict.keys():
		entro += - mid_dict[i]*log(mid_dict[i])
	#测试:熵的计算是否准确
	print("--------------------")
	print("entropy = %f"%entro)
	return entro

def compute_mid(input_attribute_list):
	'''
	输入:
		列表:整个数据集（4列的嵌套列表）
	输出:
		列表:数据集每一列近邻值中点组成的嵌套列表
		针对数据集的每一列，统计各个属性值，计算出中点，结果作为列表输出
	'''
	mid_result = list()
	row_possibility = list()
	row = list()
	del_dup = set()
	#拿到每一列的可能取值组成的列表
	#i对应于列
	for i in range(3):
		#j对应于行
		for j in range(len(input_attribute_list)):
			#将某一列的所有值添加到一个列表中
			row_possibility.append(input_attribute_list[j][i])
		#将这个列表排序
		#row_possibility.sort()
		#将这个列表用集合del_dup去重
		del_dup = set(row_possibility)
		#再将集合赋值给一个列表，
		#set_to_list就是已经排序并且去重之后的每一列的所有可能取值组成的列表
		set_to_list = list(del_dup)
		#排序需要在去重之后来做,排序后计算相邻值才有意义
		set_to_list.sort()
		#用一个保存列的列表将去重后的值的列表保存起来
		#row[i]下标对应于原数据集的第i-1列，存储这一列已经排好序的所有可能取值
		row.append(set_to_list)
	#下面计算各个属性取值的中点
	for k in row:
		mid_list = list()
		#row的每一个元素（原数据集的一列）有count_mid_list-1个中点
		count_mid_list = len(k) - 1
		for l in range(count_mid_list):
			mid_list.append(round(float((k[l]+k[l+1])/2),3))
		mid_result.append(mid_list)
	#测试:样本集每一列(属性)相邻值的中点组成的列表
	print(mid_result)
	return mid_result

def gain(input_attribute_list,row_num,mid_point):
	'''
	输入:
		1.列表:样本组成的列表input_attribute_list
		2.int:需要处理的列标号，对应于数据集的第row_num列
		3.列表:数据集每一列近邻值中点组成的嵌套列表mid_point
	输出:
		字典:
			键：近邻值中点
			值：mid_point列表中的元素(近邻值中点)对应的信息增益
	'''
	#调用entropy计算输入样本集的信息熵
	entro = entropy(input_attribute_list)
	S_count = len(input_attribute_list)
	gainlist = dict()
	#循环用于处理每一个近邻值中点
	for i in mid_point[row_num-1]:
		#i是可能分裂值
		#需要计算每一个近邻值中点作为分裂点的样本子集的熵
		#计算Entropy可能分裂点(样本集)
		Entropy_attri_sampleset = 0
		#声明两个列表S_left和S_right是按照属性划分的样本子集
		#我们按照中点来分，所以每次总是有2个样本子集
		S_left = list()
		S_right = list()
		for j in input_attribute_list:
			#j是一条记录
			#按照可能分裂点按照范围
			#将输入的总样本集input_attribute_list分为S_left和S_right两个样本子集
			if j[row_num-1] <= i:
				S_left.append(j)
			else:
				S_right.append(j)
		Entropy_attri_sampleset = len(S_left)*entropy(S_left)/float(S_count) + len(S_right)*entropy(S_right)/float(S_count)
		Entropy_attri_sampleset = round(Entropy_attri_sampleset,5)
		gainlist[i] = (round(entro - Entropy_attri_sampleset,5))
	#测试：某一列所有可能分裂点的信息增益值组成的列表,保留5位小数
	print("=====================================")
	print("第%d列所有可能分裂点的信息增益值组成的字典是"%row_num + str(gainlist))
	return gainlist

def splitpoint(input_attribute_list,mid_point):
	'''
	输入:
		1.列表:样本组成的列表input_attribute_list
		2.列表:数据集每一列近邻值中点组成的嵌套列表mid_point
	输出:
		split_point_list，分裂点组成的列表
	'''
	split_point_list = list()
	for i in range(3):
		store = list()
		store = gain(input_attribute_list,i+1,mid_point)
		sorted(store.items(),key = lambda x:x[1],reverse = True)
		mid_box = store.keys()
		#print("+++++++"+str(mid_box))
		#print("---=====++++"+str(store))
		#store.sort(reverse = True)
		split_point_list.append(mid_box[0])
	#测试:分裂点组成的列表
	print("分裂点组成的列表是" + str(split_point_list))
	return split_point_list

def discretion(input_attribute_list,split_point_list):
	'''
	输入:
		1.样本组成的列表input_attribute_list
		2.分裂点组成的列表split_point_list
	输出:
		样本离散化之后组成的列表
	'''
	discrete_list = list()
	for i in input_attribute_list:
		if i[0] <= split_point_list[0]:
			a = "yes"
		else:
			a = "no"
		if i[1] <= split_point_list[1]:
			b = "yes"
		else:
			b = "no"		
		if i[2] <= split_point_list[2]:
			c = "yes"
		else:
			c = "no"
		record_discrete = [a,b,c,i[3]]
		discrete_list.append(record_discrete)
	return discrete_list

'''
实现naive——bayes算法
给出测试数据集中每个测试样例的预测类标
同时给出每个测试样例属于每个类别的后验概率值
	样本X在survive = "1.0"的后验概率  P(survive = "1.0"|X)
		P(survive = "1.0"|X) 
		= P(X|survive = "1.0") * P(survive = "1.0")
		= P(x1|survive = "1.0") * P(x2|survive = "1.0") * P(x3|survive = "1.0") *P(survive = "1.0") 
	[x1,x2,x3] = [a,b,c]
	P(survive = "1.0") 取 survive = "1.0"的频率
	P(x1|survive = "1.0") = P(a|survive = "1.0")
	其余三个同理
	样本X在survive = "-1.0"的后验概率 P(survive = "-1.0"|X)同理
'''


def naive_bayes(discrete_training_list,discrete_test_record):
	'''
	输入:
		1.离散化之后的训练样本集组成的列表
		2.离散化之后的测试样本
	输出:
		列表，包括：
			1.该测试样例属于每个类别的后验概率值
			2.测试样例的预测类标
			3.测试样例的实际类标
	'''
	count_total = len(discrete_training_list)
	count_survive = 0
	count_not_survive =0
	count_training = 0
	count_x1_survive = 0
	count_x1_not_survive = 0
	count_x2_survive = 0
	count_x2_not_survive = 0
	count_x3_survive = 0
	count_x3_not_survive = 0
	for j in discrete_training_list:
		if j[3] == '1.0':
			count_survive += 1
		else:
			count_not_survive += 1
	for i in discrete_training_list:
		if (i[0] == discrete_test_record[0]) and (i[3] == '1.0'):
			count_x1_survive += 1
		elif (i[0] == discrete_test_record[0]) and (i[3] == '-1.0'):
			count_x1_not_survive += 1
		if (i[1] == discrete_test_record[1]) and (i[3] == '1.0'):
			count_x2_survive += 1
		elif (i[1] == discrete_test_record[1]) and (i[3] == '-1.0'):
			count_x2_not_survive +=	1		
		if (i[2] == discrete_test_record[2]) and (i[3] == '1.0'):
			count_x3_survive += 1
		elif (i[2] == discrete_test_record[2]) and (i[3] == '-1.0'):
			count_x3_not_survive += 1
	#print(count_survive)
	#print(float(count_total))
	p_survive = round(float(count_survive/float(count_total)),5)
	p_not_survive =round(float(count_not_survive/float(count_total)),5)
	p_x1_survive = round(float(count_x1_survive/float(count_survive)),5)
	p_x1_not_survive = round(float(count_x1_not_survive/float(count_not_survive)),5)
	p_x2_survive = round(float(count_x2_survive/float(count_survive)),5)
	p_x2_not_survive = round(float(count_x2_not_survive/float(count_not_survive)),5)
	p_x3_survive = round(float(count_x3_survive/float(count_survive)),5)
	p_x3_not_survive = round(float(count_x3_not_survive/float(count_not_survive)),5)
	survive_Posterior = round(p_x1_survive*p_x2_survive*p_x3_survive*p_survive,5)
	not_survive_Posterior = round(p_x1_not_survive*p_x2_not_survive*p_x3_not_survive*p_not_survive,5)
	#print(count_survive/float(float(count_total)))
	test_result = list()
	if survive_Posterior >  not_survive_Posterior:
		a = survive_Posterior
		b = '1.0'
		c = discrete_test_record[3]
		test_result.append([a,b,c])
	else:
		a = not_survive_Posterior
		b = '-1.0'
		c = discrete_test_record[3]
		test_result.append([a,b,c])
	#print(str(test_result))
	return test_result


'''
读取训练样例
'''
f = open("","r")





#以@开头的行为注释行，剩下每一行为一个带类标的训练样例，由空格隔开的属性值构成，最后一个属性值为类标
#box1存储读取的所有行,box2存储训练样例,count1用于统计训练样例个数
box1 = f.readlines()

box2 = list()
count1 = 0
for elements in  box1:
	if elements[0] != '@':
		box2.append(elements)
		count1 += 1

print("一共有%d个样本"%count1)
'''
构建训练集与测试集
对给定的数据集随机划分，70%作为训练数据，30%作为测试数据
'''
count_training = int(count1*0.7)
count_test = int(count1*0.3)

print("count_training = %d"%count_training)
print("count_test = %d"%count_test)
#用random.shuffle随机打乱训练样例box2，random.shuffle返回none需要注意
#box_training是存储训练样例的列表，box_test是存储测试样例的列表
random.shuffle(box2)

box_training = list()

box_test = list()

count2 = 0

while count2 <= count_training:
	count2 += 1
	box_training.append(box2.pop()) #box2.pop()为从box2列表中删除的最后一个元素
#print("训练集的大小是%d"%len(box_training))
while len(box2)> 0:
	box_test.append(box2.pop())


#整理
box_test = tidy(box_test)
box_training = tidy(box_training)


discrete_box_training = discretion(box_training,splitpoint(box_training,compute_mid(box_training)))
discrete_box_test = discretion(box_test,splitpoint(box_test,compute_mid(box_test)))

#print(discrete_box_training)
#naive_bayes(discrete_box_training,discrete_box_test[0])

def main():
	forcast = list()
	for i in range(len(discrete_box_test)):
		forcast.append(naive_bayes(discrete_box_training,discrete_box_test[i]))
	accurate_point = 0
	#print(forcast)
	for j in forcast:
		if j[0][1] == j[0][2]:
			accurate_point += 1
	print("预测的准确率是%f"%(accurate_point/float(len(forcast))))

if __name__ == "__main__":
	main()
