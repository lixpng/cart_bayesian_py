#!/usr/bin/python
# -*- coding: utf-8 -*-

import time

training_data = [] #格式如:[{attr1:val,attr2:val,...'category':category},...]
categories = []
COLS = None #数据每一列的属性名
ATTR_DISPERSED = None #属性值是否是离散的
ATTRS_COUNT = 0 #属性的个数
DIS_ATTR_VALS = {} #存储每一个离散属性的不同值, 如{attr1:[val1,val2...]...}
start = 0

class Node: #决策树节点
    def __init__(self, diff_loss, split_value, attr=None, length=0, left=None, right=None, category=None):
        self.diff_loss = diff_loss #差异性损失
        self.split_value = split_value #相邻值中点
        self.attr = attr #选取的属性
        self.length = length #在训练时划分到此树的数据个数
        self.left = left #左子树
        self.right = right #右子树
        self.category = category
        
def read_data(file='./tratiningiris.data'):
    global start, COLS, ATTR_DISPERSED, ATTRS_COUNT, training_data, categories, DIS_ATTR_VALS
    start = time.time()

    training_data = [] #格式如:[{attr1:val,attr2:val,...'category':category},...]
    categories = []
    COLS = None #数据每一列的属性名
    ATTR_DISPERSED = None #属性值是否是离散的
    ATTRS_COUNT = 0 #属性的个数
    DIS_ATTR_VALS = {} #存储每一个离散属性的不同值, 如{attr1:[val1,val2...]...}

    with open(file, 'r') as f:
        lines = f.readlines()
        COLS = lines[0].strip().split(',')
        ATTR_DISPERSED = [x=='dispersed' for x in lines[1].strip().split(',')]
        ATTRS_COUNT = len(COLS) - 1 #最后一个为类, 不是属性名
        for line in lines[2:]:
            tmp = line.strip().split(',')

            category = tmp[-1]
            if category not in categories:
                categories.append(category)

            values = []
            for i in range(ATTRS_COUNT+1):
                val = tmp[i]
                if ATTR_DISPERSED[i]:
                    values.append(val)
                    attr_name = COLS[i]
                    if attr_name not in DIS_ATTR_VALS:
                        DIS_ATTR_VALS[attr_name] = []
                    if val not in DIS_ATTR_VALS[attr_name]:
                        DIS_ATTR_VALS[attr_name].append(val)
                else:
                    values.append(float(val))

            samples = dict(zip(COLS, values))
            training_data.append(samples)

def get_root_gini(root_data):
    total = 0
    count_of_category = get_count_of_category(root_data)
        
    root_gini = 1
    for category, count in count_of_category.items():
        root_gini -= (count/len(root_data))**2
    
    return root_gini

def get_count_of_category(samples):
    count_of_category = {}
    for sample in samples:
        category = sample['category']
        if category not in count_of_category:
            count_of_category[category] = 0
        count_of_category[category] += 1
    return count_of_category

def get_gini_by_category_count(count_of_category, total):
    gini = 1
    for category, count in count_of_category.items():
        gini -= (count/total)**2
    return gini

def get_real_sub_set(items): #获取一个集合的真子集
    res = []
    n = len(items)  
    for i in range(1, 2**n-1):  
        combo = []  
        for j in range(n):  
            if(i >> j ) % 2 == 1:  
                combo.append(items[j])  
        res.append(combo)
    return res;

def get_diff_loss(root_data, left_tree, right_tree): #获取差异性损失
    root_gini = get_root_gini(root_data)
    samples_count = len(root_data)
    left_len = len(left_tree)
    right_len = len(right_tree)
    left_count_of_category = get_count_of_category(left_tree) #左子树中每一类的个数
    right_count_of_category = get_count_of_category(right_tree) #右子树中每一类的个数

    left_gini = get_gini_by_category_count(left_count_of_category, left_len)
    right_gini = get_gini_by_category_count(right_count_of_category, right_len)

    sub_gini = root_gini - (left_len*left_gini/samples_count) - (right_len*right_gini/samples_count)

    return sub_gini

def get_best_attr(root_data):

    best_attr = { #当前应选择的最好属性
        'attr': None,
        'diff_loss': 0, #差异性损失
        'split_value': None, #按哪个中点值或是哪个值集合划分树
        'left_tree': None,
        'right_tree': None
    }

    if not is_diff_data(root_data):
        return best_attr
    
    for x in range(ATTRS_COUNT):

        attr = COLS[x]

        if ATTR_DISPERSED[x]:
            attr_vals = DIS_ATTR_VALS[attr]
            sub_sets = get_real_sub_set(attr_vals)
            sub_sets = sub_sets[:len(sub_sets)//2] #只需前半部分即可, 因为后面就是重复的了
            for sub_set in sub_sets:
                left_tree = []
                right_tree = []
                for sample in root_data:
                    if sample[attr] in sub_set:
                        left_tree.append(sample)
                    else:
                        right_tree.append(sample)
                
                diff_loss = get_diff_loss(root_data, left_tree, right_tree)
                if best_attr['diff_loss'] < diff_loss:
                    best_attr['diff_loss'] = diff_loss
                    best_attr['attr'] = attr
                    best_attr['right_tree'] = right_tree
                    best_attr['left_tree'] = left_tree
                    best_attr['split_value'] = sub_set
        else:
            samples = sorted(root_data, key=lambda k: k[attr])

            for i in range(len(samples)-1):
                middle = (samples[i][attr] + samples[i+1][attr])/2

                left_tree = samples[:i+1]
                right_tree = samples[i+1:]

                diff_loss = get_diff_loss(root_data, left_tree, right_tree)

                if best_attr['diff_loss'] < diff_loss:
                    best_attr['diff_loss'] = diff_loss
                    best_attr['attr'] = attr
                    best_attr['right_tree'] = right_tree
                    best_attr['left_tree'] = left_tree
                    best_attr['split_value'] = middle

    return best_attr

def is_diff_data(data):
    attr = data[0]['category']
    for item in data:
        if item['category'] != attr:
            return True
    return False

def split_tree(root_data):

    best_attr = get_best_attr(root_data)
    
    root = Node(best_attr['diff_loss'], best_attr['split_value'], best_attr['attr'], len(root_data))

    if not best_attr['attr']: #不需要再拆分
        root.category = root_data[0]['category']
        return root
        
    left = best_attr['left_tree']        
    right = best_attr['right_tree']

    root.left = split_tree(left)
    root.right = split_tree(right)

    return root;
    
def print_tree(tree, count):
    if tree:
        print('>>>>>>>>>>>>>>>>>>>>>>>\ndiff_loss: ', 
            tree.diff_loss, '\split_value: ', tree.split_value,
            '\nattr: ', tree.attr, '\nlength: ', tree.length,
            '\ncount: ', count, '\ncategory: ', tree.category)
        print_tree(tree.left, count+1)
        print_tree(tree.right, count+1)
        # print(tree.left, tree.right)

def use_it(root, data):
    attr = root.attr
    if not root:
        return 'can not split'
    category = root.category
    if category:
        return category
    split_value = root.split_value
    val = data[attr]
    if isinstance(split_value, list):
        if val in split_value:
            return use_it(root.left, data)
        else:
            return use_it(root.right, data)
    else:
        val = float(val)
        if val <= root.split_value:
            return use_it(root.left, data)
        else:
            return use_it(root.right, data)

def test(file='./testiris.data'):

    tree = split_tree(training_data)

    with open(file, 'r') as f:
        result = {}
        real = {}
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split(',')

            real_cat = tmp[-1]
            if real_cat not in real:
                real[real_cat] = 0
            real[real_cat] += 1

            d = dict(zip(COLS,[x for x in tmp[:-1]]))
            m = use_it(root=tree, data=d)
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

if __name__ == '__main__':
    main()