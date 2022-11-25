import numpy as np

omega1 = [-3.9847,-3.5549,-1.2401,-0.9780,-0.7932,-2.8531,-2.7605,-3.7287,
-3.5414,-2.2692,-3.4549,-3.0752,-3.9934,-0.9780,-1.5799,-1.4885,-0.7431,
-0.4221,-1.1186,-2.3462,-1.0826,-3.4196,-1.3193,-0.8367,-0.6579,-2.9683]
omega2 = [2.8792,0.7932,1.1882,3.0682,4.2532,0.3271,0.9846,2.7648,2.6588]
loss_rate = [6,1]
x = np.array(omega1+omega2)
y1 = np.zeros(len(omega1))
y2 = np.ones(len(omega2))
y = np.append(y1,y2)


def pre_calculate(x,y):
    ratio=[]
    mean = []
    var = []
    for i in range(2):
        class_i = x[np.nonzero(i == y)]
        ratio.append(len(class_i)/len(x))
        m = np.mean(class_i)
        mean.append(m)
        v = np.var(class_i)
        var.append(v)

    return ratio,mean,var


def validate(x,y,ratio,mean,var):

    pred_list = []
    wrong = 0
    for i in range(len(x)):

        p1 = (np.exp(-(x[i]-mean[0])**2/(2*var[0])))/(np.sqrt(2*np.pi*var[0]))
        p2 = (np.exp(-(x[i]-mean[1])**2/(2*var[1])))/(np.sqrt(2*np.pi*var[1]))
        
        # p1 = np.log(p1*ratio[0])
        # p2 = np.log(p2*ratio[1])
        # # 风险决策
        p1 = np.log(p1*ratio[0]*loss_rate[0])
        p2 = np.log(p2*ratio[1]*loss_rate[1])

        if p1 > p2:
            pred = 0
        else :
            pred = 1
        pred_list.append(pred)

        if pred == y[i]:
            pass
        else:
            wrong += 1
    acc = 1 - wrong / len(x)
    print('acc:{}'.format(acc))
    
    return pred_list 

def find_split(x,ratio,mean,var):
    split = 0
    temp = 10
    for i in np.linspace(0,2,1000):

        p1 = (np.exp(-(i-mean[0])**2/(2*var[0])))/(np.sqrt(2*np.pi*var[0]))
        p2 = (np.exp(-(i-mean[1])**2/(2*var[1])))/(np.sqrt(2*np.pi*var[1]))
        
        # p1 = np.log(p1*ratio[0])
        # p2 = np.log(p2*ratio[1])
        # # 风险决策
        p1 = np.log(p1*ratio[0]*loss_rate[0])
        p2 = np.log(p2*ratio[1]*loss_rate[1])

        if abs(p1 - p2) < temp:
            temp = abs(p1 - p2)
            split = i

    return split

ratio,mean,var=pre_calculate(x,y)
ratio = [0.9, 0.1]
print('ratio: {} , mean: {}, variance: {}'.format(ratio,mean,var))
# print(np.sqrt(var[0]),np.sqrt(var[1]))
# print(2*var[0],2*var[1])
split = find_split(x,ratio,mean,var)
print('split at:{}'.format(split))
pred = validate(x,y,ratio,mean,var)
print(pred)
