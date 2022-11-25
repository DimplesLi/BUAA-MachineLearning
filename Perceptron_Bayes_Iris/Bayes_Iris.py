import numpy as np
import csv

def read_data():
    data = csv.reader(open('iris.data','r'))
    
    count_set = 0
    count_ver = 0
    count_vir = 0
    dataset = []
    label = []
    for line in data:
        
        # print(line)
        line = np.array(line)
        num = []
        for i in range(4):
            num.append(float(line[i]))
        dataset.append(num)
    
        if line[-1] == 'Iris-setosa':
            label.append(0)
            count_set += 1
        elif line[-1] == 'Iris-versicolor':
            label.append(1)
            count_ver += 1
        else:#Iris-virginica
            label.append(2)
            count_vir += 1
    
    # print (dataset)
    # print(label)
    # print(count_set)
    dataset = np.array(dataset)
    label = np.array(label)
    return dataset,label,count_set,count_ver,count_vir

def pre_calculate(x,y):
    
    ratio=[]
    mean = []
    var = []
    for i in range(3):
        class_i = x[np.nonzero(i == y)]
        ratio.append(len(class_i)/len(x))
        m = []
        for j in range(4):
            m.append(np.mean(class_i[:,j]))
        mean.append(m)
        
        v = []
        for j in range(4):
            v.append(np.var(class_i[:,j]))
        var.append(v)
        # print('mean:{},var:{}'.format(mean,var))
    return ratio,mean,var


def validate(x,y,ratio,mean,var):
    mean = np.array(mean)
    pred_list = []
    wrong = 0
    for i in range(len(x)):
        p1 = 1
        for j in range(4):
            p1 *= (np.exp(-(x[i][j]-mean[0][j])**2/(2*var[0][j])))/(np.sqrt(2*np.pi*var[0][j]))
        p1 = np.sum(np.log(p1))
        p2 = 1
        for j in range(4):
            p2 *= (np.exp(-(x[i][j]-mean[1][j])**2/(2*var[1][j])))/(np.sqrt(2*np.pi*var[1][j]))
        p2 = np.sum(np.log(p2))
        p3 = 1
        for j in range(4):
            p3 *= (np.exp(-(x[i][j]-mean[2][j])**2/(2*var[2][j])))/(np.sqrt(2*np.pi*var[2][j]))
        p3 = np.sum(np.log(p3))

        # print(p1,p2,p3)
        p_max = max([p1,p2,p3])

        if p_max == p1:
            pred = 0
        elif p_max == p2:
            pred = 1
        else:
            pred = 2
        pred_list.append(pred)

        if pred == y[i]:
            pass
        else:
            wrong += 1
    acc = 1 - wrong / len(x)
    print(acc)
    
    return pred_list 

dataset,label,count_set,count_ver,count_vir = read_data()
ratio,mean,var=pre_calculate(dataset,label)
# print('ratio: {} , mean: {}, variance: {}'.format(ratio,mean,var))
pred = validate(dataset,label,ratio,mean,var)
print(pred)