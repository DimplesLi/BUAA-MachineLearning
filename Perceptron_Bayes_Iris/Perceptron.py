import numpy as np
import csv

lr = 0.1


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
            label.append(1)
            count_set += 1
        elif line[-1] == 'Iris-versicolor':
            label.append(2)
            count_ver += 1
        else:#Iris-virginica
            label.append(3)
            count_vir += 1
    
    # print (dataset)
    # print(label)
    # print(count_set)
    return dataset,label,count_set,count_ver,count_vir


def train(dataset,label,count_set,count_ver,count_vir):
    
    weight = np.random.random((1,4))
    # weight = [1,0,0,0]
    print('initial weight: {} '.format(weight))

    print(label)
    label[30]=2
    # label[20]=2

    label[74]=1
    # label[82]=1

    iter = 0
    epoch = 0
    while( epoch <= 500 ):

        

        WX_list = []

        epoch += 1
        wrong = 0
        # print(count_set + count_ver)
        ran = range(count_set + count_ver)
        # ran = range(count_set,count_set+count_ver+count_vir)
        
        for i in ran:
            if label[i] == 1:
                WX = np.dot(weight, dataset[i])
                if WX <0:
                    weight += lr * np.transpose(dataset[i])
                    wrong += 1
                    iter += 1
                # print('W={}'.format(weight))
                WX_list.append(WX)
            else:
                WX = -np.dot(weight, dataset[i])
                if WX <0:
                    weight -= lr * np.transpose(dataset[i])
                    wrong += 1
                    iter += 1
                # print('W={}'.format(weight))
                WX_list.append(WX)
        
        print('acc:{}/{} at iteration {},weight = {}'.format(len(WX_list)-wrong,len(WX_list),iter,weight))
        if min(WX_list) >= 0:
            print('finished at iteration {} , epoch {}!'.format(iter,epoch))
            break

    return weight

def test(dataset,weight):

    test = np.zeros(len(dataset[0]))
    for _ in range(10):
        index = np.random.choice(range(count_set))
        # index = np.random.choice(range(count_ver,count_set+count_ver+count_vir))
        test += dataset[index]
    test /= len(dataset[0])

    print('test feature: {} label: Iris-setosa'.format(test))
    # print('test feature: {} label: Iris-versicolor'.format(test))


    WX = np.dot(weight, test)
    if WX < 0:
        print('Iris-versicolor')
        # print('Iris-virginica')
    if WX > 0:
        print('Iris-setosa')
        # print('Iris-versicolor')


dataset,label,count_set,count_ver,count_vir = read_data()
weight = train(dataset,label,count_set,count_ver,count_vir)
test(dataset,weight)
