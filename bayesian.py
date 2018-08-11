#!/usr/bin/python
# -*- coding: utf-8 -*-

# 运算拆开, 不然会有精度问题   方差和标准差都存起来

from math import pi, exp
import time

CATEGORIES = [] # 分类
ATTRS_COUNT = 0 #属性的个数
COLS = None #数据每一列的属性名
ATTR_DISPERSED = None #属性值是否是离散的

start = 0

training_data = {} #读取的训练数据, 类似{类1: [[第一组],[第二组],[第三组]...]...}
data_number = 0 #训练数据集中的数据总数

def read_data(file='./tratiningiris.data', local=True):

    global start, data_number, COLS, ATTR_DISPERSED, ATTRS_COUNT, CATEGORIES, training_data
    
    start = time.time()

    CATEGORIES = [] # 分类
    ATTRS_COUNT = 0 #属性的个数
    COLS = None #数据每一列的属性名
    ATTR_DISPERSED = None #属性值是否是离散的

    training_data = {} #读取的训练数据, 类似{类1: [[第一组],[第二组],[第三组]...]...}
    data_number = 0 #训练数据集中的数据总数

    if local:
        handle_file(file, read_data)
        return

    lines = file.readlines()
    COLS = lines[0].strip().split(',')
    ATTR_DISPERSED = [x=='dispersed' for x in lines[1].strip().split(',')]
    ATTRS_COUNT = len(COLS) - 1 #最后一个为类, 不是属性名
    for line in lines[2:]:

        data_number += 1

        tmp = line.strip().split(',')

        category = tmp[-1]
        if category not in CATEGORIES:
            CATEGORIES.append(category)

        attrs = []
        for i in range(ATTRS_COUNT):
            val = tmp[i]
            if ATTR_DISPERSED[i]:
                attrs.append(val)
            else:
                attrs.append(float(val))
        category = tmp[ATTRS_COUNT]

        if category not in training_data:
            training_data[category] = []
            
        training_data[category].append(attrs)

def handle_file(file, cb):
    with open(file, 'r') as f:
        return cb(f, False)

def get_average(category, index): #获取分类category第index+1个属性的平均值
    _sum = 0
    data = training_data[category]
    for attrs in data:
        _sum += attrs[index]
    return _sum/len(data)

def get_variance(category, index, avg): #获取分类category第index+1个属性的方差
    if not avg:
        avg = get_average(category, index)
    _sum = 0
    data = training_data[category]
    for attrs in data:
        _sum += (attrs[index] - avg)**2
    
    variance = _sum/(len(data)-1) #样本标准方差
    return variance

def get_attr_category_count(index_a,val_a, category): #获取数据中类别为category且第index_a+1个属性为val_a的个数
    data = training_data[category]
    count = 0
    for attrs in data:
        if attrs[index_a] == val_a:
            count += 1
    return count

def compute_probability(attrs):
    probabilities = {}

    for category in CATEGORIES:
        probabilities[category] = 1
        category_probabilitiy = len(training_data[category])/data_number
        density = None
        for i, val in enumerate(attrs):

            if ATTR_DISPERSED[i]:
                i_category = get_attr_category_count(i, val, category)
                density = i_category/len(training_data[category])
            else:
                val = float(val)
                avg = get_average(category, i)
                variance = get_variance(category, i, avg)
                a = (val-avg)**2
                b = -a/(2*variance)
                c = exp(b)
                d = (2*pi)**0.5
                e = variance**0.5*d
                density = c/e

            probabilities[category] *= density
        probabilities[category] *= category_probabilitiy
        
    return probabilities

def test(file='./testcar.data', local=True):
    if local:
        return handle_file(file, test)
        
    result = {}
    real = {}
    lines = file.readlines()
    for line in lines:
        line = line.strip().split(',')

        real_cat = line[-1]
        if real_cat not in real:
            real[real_cat] = 0
        real[real_cat] += 1

        tmp = compute_probability(line[:ATTRS_COUNT])
        m = CATEGORIES[0]
        for key, val in tmp.items():
            if tmp[m] < val:
                m = key
        if m not in result:
            result[m] = 0
        result[m] += 1
    print(result)

    err_num = 0
    for cat, val in real.items():
        if cat not in result:
            err_num += val
        elif val > result[cat]:
            err_num += val - result[cat]

    success_rate = (len(lines) - err_num)/len(lines)
    success_rate = round(success_rate, 4) * 100
    success_rate = str(success_rate) + '%'

    return {
        'test_result': result,
        'real_result': real,
        'error_count': err_num,
        'success_rate': success_rate,
        'test_count': len(lines),
        'time': round(time.time() - start, 4)
    }
        


def main():
    read_data(file='./trainingcar.data')
    test(file='./testcar.data')

if(__name__ == '__main__'):
    main()
