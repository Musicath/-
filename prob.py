# 数据预处理部分
import math
import numpy as np
import pandas as pd
import joblib


def get_prob_add(df_outer_Face, df_outer_Imsi):
    class Data:
        def __init__(self):
            self.data = []
            self.dic = {}

    '''
    字典中，一个信息对应的是长度为n的数组
    数组中每个元素为一个长度为4的数组：位置,时间,经度,纬度
    '''
    df_Face = df_outer_Face.copy()
    df_Imsi = df_outer_Imsi.copy()

    df_Face['DeviceID'] = pd.Categorical(df_Face['DeviceID']).codes
    df_Imsi['DeviceID'] = pd.Categorical(df_Imsi['DeviceID']).codes

    df_Face['DeviceID'] = pd.to_numeric(df_Face['DeviceID'])
    df_Face['Lon'] = pd.to_numeric(df_Face['Lon'])
    df_Face['Lat'] = pd.to_numeric(df_Face['Lat'])
    df_Face['TimeStamp'] = pd.to_numeric(df_Face['TimeStamp'])

    df_Imsi['DeviceID'] = pd.to_numeric(df_Imsi['DeviceID'])
    df_Imsi['Lon'] = pd.to_numeric(df_Imsi['Lon'])
    df_Imsi['Lat'] = pd.to_numeric(df_Imsi['Lat'])
    df_Imsi['TimeStamp'] = pd.to_numeric(df_Imsi['TimeStamp'])

    device_num = len(df_Face['DeviceID'].unique())

    '''
        下面这个是核心函数
        k是参数，我这边最优解为2.4e-6，那边可以重新让它评估
        因为它改了时间戳吧，应该会变的
        p是人标志号，存在了字典中：for p in P.dic:
        c是物标志号，存在了字典中：for c in C.dic:
        d是距离矩阵，与u相关的，现在暂时取u=0，先不训练这个参数
        p_num是地点数，后来发现不同，训练集是5，测试集是6
    '''

    def data2dic(data):  # 选取特征数据并按时间排序
        dic = {}
        for i in data:
            if i[4] in dic:  # 改动的时候保证顺序是：位置，时间，经度，纬度
                dic[i[4]].append([i[0], i[6], i[1], i[2]])
            else:
                dic[i[4]] = [[i[0], i[6], i[1], i[2]]]
        for d in dic:
            dic[d].sort(key=lambda x: (x[1]))
        return dic

    def Pro(k,p,c,lmda,device_num):#极大似然估计
        s=[]
        LC=len(c)
        LP=len(p)
        ps=0
        cs=-1
        while ps<LP:
            while cs+1!=LC and c[cs+1][1]<=p[ps][1]:#更新cs，保证ct>ps或ct==LC
                cs=cs+1
            ct=cs+1
            pt=ps
            if ct==LC:#晚于记录中的最晚时间
                pros=pow(k*abs(p[ps][1]-c[cs][1])+1,-1/2)
                if p[ps][0]==c[cs][0]:#计算不在相应地点的概率
                    temp=(1-1/device_num)*(1-pros)
                else:
                    temp=(1-1/device_num)+pros/device_num
                s.append(1-math.sqrt(temp*(1-1/device_num)))
            else:#ct>ps
                while pt+1!=LP and p[pt+1][1]<c[ct][1]:#更新pt，保证pt+1>=ct
                    pt=pt+1
                if ct==0:#早于记录中的最早时间
                    prot=pow(k*abs(p[pt][1]-c[ct][1])+1,-1/2)
                    if p[pt][0]==c[ct][0]:
                        temp=(1-1/device_num)*(1-prot)
                    else:
                        temp=(1-1/device_num)+prot/device_num
                    s.append(1-math.sqrt(temp*(1-1/device_num)))
                else:#指数凸组合
                    pros=pow(k*abs(p[ps][1]-c[cs][1])+1,-1/2)
                    prot=pow(k*abs(p[pt][1]-c[ct][1])+1,-1/2)
                    if p[ps][0]==c[cs][0]:
                        temps=(1-1/device_num)*(1-pros)
                    else:
                        temps=(1-1/device_num)+pros/device_num
                    if p[pt][0]==c[ct][0]:
                        tempt=(1-1/device_num)*(1-prot)
                    else:
                        tempt=(1-1/device_num)+prot/device_num
                    s.append(1-math.sqrt(temps*tempt))
            ps=pt+1
        n_s=len(s)
        s_=1
        for i in s:
            s_=s_*i
        s_=pow(s_,1/n_s)+lmda*pow(n_s,-1/2)
        return s_

    C = Data()
    P = Data()
    C.data = np.array(df_Imsi)
    P.data = np.array(df_Face)

    # data[0,1,2,4,6]=[位置，经度，纬度，目标，时间]
    C.dic = data2dic(C.data)
    P.dic = data2dic(P.data)
    # dic[目标]=[[位置,时间,经度,纬度]]

    k = 5e-6
    lmda = -5

    prob_list = []

    i = 0
    for p in P.dic:
        i = i + 1
        for c in C.dic:
            s = Pro(k, P.dic[p], C.dic[c],lmda,device_num)
            prob_list.append([p, c, s])
        if i % 10 == 0:
            print('\r', 'now calculating prob for person NO.' + str(i), end='', flush=True)

    pro_df = pd.DataFrame(prob_list, columns=['FaceLabel', 'Code', 'Prob'])
    return pro_df

def get_prob_ucb(df_outer_Face, df_outer_Imsi):
    class Data:
        def __init__(self):
            self.data = []
            self.dic = {}

    '''
    字典中，一个信息对应的是长度为n的数组
    数组中每个元素为一个长度为4的数组：位置,时间,经度,纬度
    '''
    df_Face = df_outer_Face.copy()
    df_Imsi = df_outer_Imsi.copy()

    df_Face['DeviceID'] = pd.Categorical(df_Face['DeviceID']).codes
    df_Imsi['DeviceID'] = pd.Categorical(df_Imsi['DeviceID']).codes

    df_Face['DeviceID'] = pd.to_numeric(df_Face['DeviceID'])
    df_Face['Lon'] = pd.to_numeric(df_Face['Lon'])
    df_Face['Lat'] = pd.to_numeric(df_Face['Lat'])
    df_Face['TimeStamp'] = pd.to_numeric(df_Face['TimeStamp'])

    df_Imsi['DeviceID'] = pd.to_numeric(df_Imsi['DeviceID'])
    df_Imsi['Lon'] = pd.to_numeric(df_Imsi['Lon'])
    df_Imsi['Lat'] = pd.to_numeric(df_Imsi['Lat'])
    df_Imsi['TimeStamp'] = pd.to_numeric(df_Imsi['TimeStamp'])

    device_num = len(df_Face['DeviceID'].unique())

    '''
        下面这个是核心函数
        k是参数，我这边最优解为2.4e-6，那边可以重新让它评估
        因为它改了时间戳吧，应该会变的
        p是人标志号，存在了字典中：for p in P.dic:
        c是物标志号，存在了字典中：for c in C.dic:
        d是距离矩阵，与u相关的，现在暂时取u=0，先不训练这个参数
        p_num是地点数，后来发现不同，训练集是5，测试集是6
    '''

    def data2dic(data):  # 选取特征数据并按时间排序
        dic = {}
        for i in data:
            if i[4] in dic:  # 改动的时候保证顺序是：位置，时间，经度，纬度
                dic[i[4]].append([i[0], i[6], i[1], i[2]])
            else:
                dic[i[4]] = [[i[0], i[6], i[1], i[2]]]
        for d in dic:
            dic[d].sort(key=lambda x: (x[1]))
        return dic

    def Pro(k,p,c,lmda,device_num):#极大似然估计
        s=[]
        LC=len(c)
        LP=len(p)
        ps=0
        cs=-1
        while ps<LP:
            while cs+1!=LC and c[cs+1][1]<=p[ps][1]:#更新cs，保证ct>ps或ct==LC
                cs=cs+1
            ct=cs+1
            pt=ps
            if ct==LC:#晚于记录中的最晚时间
                s.append(1)
            else:#ct>ps
                while pt+1!=LP and p[pt+1][1]<c[ct][1]:#更新pt，保证pt+1>=ct
                    pt=pt+1
                if ct==0:#早于记录中的最早时间
                    s.append(1)
                else:#指数凸组合
                    s.append(1)
            ps=pt+1
        n_s=len(s)
        return pow(n_s,-1/2)

    C = Data()
    P = Data()
    C.data = np.array(df_Imsi)
    P.data = np.array(df_Face)

    # data[0,1,2,4,6]=[位置，经度，纬度，目标，时间]
    C.dic = data2dic(C.data)
    P.dic = data2dic(P.data)
    # dic[目标]=[[位置,时间,经度,纬度]]

    k = 5e-6
    lmda = -5

    prob_list = []

    i = 0
    for p in P.dic:
        i = i + 1
        for c in C.dic:
            s = Pro(k, P.dic[p], C.dic[c],lmda,device_num)
            prob_list.append([p, c, s])
        if i % 10 == 0:
            print('\r', 'now calculating prob for person NO.' + str(i), end='', flush=True)

    pro_df = pd.DataFrame(prob_list, columns=['FaceLabel', 'Code', 'Prob'])
    return pro_df

def get_prob_best(df_outer_Face, df_outer_Imsi):
    class Data:
        def __init__(self):
            self.data = []
            self.dic = {}

    '''
    字典中，一个信息对应的是长度为n的数组
    数组中每个元素为一个长度为4的数组：位置,时间,经度,纬度
    '''
    df_Face = df_outer_Face.copy()
    df_Imsi = df_outer_Imsi.copy()

    df_Face['DeviceID'] = pd.Categorical(df_Face['DeviceID']).codes
    df_Imsi['DeviceID'] = pd.Categorical(df_Imsi['DeviceID']).codes

    df_Face['DeviceID'] = pd.to_numeric(df_Face['DeviceID'])
    df_Face['Lon'] = pd.to_numeric(df_Face['Lon'])
    df_Face['Lat'] = pd.to_numeric(df_Face['Lat'])
    df_Face['TimeStamp'] = pd.to_numeric(df_Face['TimeStamp'])

    df_Imsi['DeviceID'] = pd.to_numeric(df_Imsi['DeviceID'])
    df_Imsi['Lon'] = pd.to_numeric(df_Imsi['Lon'])
    df_Imsi['Lat'] = pd.to_numeric(df_Imsi['Lat'])
    df_Imsi['TimeStamp'] = pd.to_numeric(df_Imsi['TimeStamp'])

    device_num = len(df_Face['DeviceID'].unique())

    '''
        下面这个是核心函数
        k是参数，我这边最优解为2.4e-6，那边可以重新让它评估
        因为它改了时间戳吧，应该会变的
        p是人标志号，存在了字典中：for p in P.dic:
        c是物标志号，存在了字典中：for c in C.dic:
        d是距离矩阵，与u相关的，现在暂时取u=0，先不训练这个参数
        p_num是地点数，后来发现不同，训练集是5，测试集是6
    '''

    def data2dic(data):  # 选取特征数据并按时间排序
        dic = {}
        for i in data:
            if i[4] in dic:  # 改动的时候保证顺序是：位置，时间，经度，纬度
                dic[i[4]].append([i[0], i[6], i[1], i[2]])
            else:
                dic[i[4]] = [[i[0], i[6], i[1], i[2]]]
        for d in dic:
            dic[d].sort(key=lambda x: (x[1]))
        return dic

    def Pro(k,p,c,lmda,device_num):#极大似然估计
        s=[]
        LC=len(c)
        LP=len(p)
        ps=0
        cs=-1
        while ps<LP:
            while cs+1!=LC and c[cs+1][1]<=p[ps][1]:#更新cs，保证ct>ps或ct==LC
                cs=cs+1
            ct=cs+1
            pt=ps
            if ct==LC:#晚于记录中的最晚时间
                pros=math.exp(-k*abs(p[ps][1]-c[cs][1])**2.2)
                if p[ps][0]==c[cs][0]:#计算不在相应地点的概率
                    temp=(1-1/device_num)*(1-pros)
                else:
                    temp=(1-1/device_num)+pros/device_num
                s.append(1-math.sqrt(temp*(1-1/device_num)))
            else:#ct>ps
                while pt+1!=LP and p[pt+1][1]<c[ct][1]:#更新pt，保证pt+1>=ct
                    pt=pt+1
                if ct==0:#早于记录中的最早时间
                    prot=math.exp(-k*abs(p[pt][1]-c[ct][1])**2.2)
                    if p[pt][0]==c[ct][0]:
                        temp=(1-1/device_num)*(1-prot)
                    else:
                        temp=(1-1/device_num)+prot/device_num
                    s.append(1-math.sqrt(temp*(1-1/device_num)))
                else:#指数凸组合
                    pros=math.exp(-k*abs(p[ps][1]-c[cs][1])**2.2)
                    prot=math.exp(-k*abs(p[pt][1]-c[ct][1])**2.2)
                    if p[ps][0]==c[cs][0]:
                        temps=(1-1/device_num)*(1-pros)
                    else:
                        temps=(1-1/device_num)+pros/device_num
                    if p[pt][0]==c[ct][0]:
                        tempt=(1-1/device_num)*(1-prot)
                    else:
                        tempt=(1-1/device_num)+prot/device_num
                    s.append(1-math.sqrt(temps*tempt))
            ps=pt+1
        n_s=len(s)
        s_=1
        for i in s:
            s_=s_*i
        s_=pow(s_,1/n_s)+lmda*pow(n_s,-1/2)
        return s_

    C = Data()
    P = Data()
    C.data = np.array(df_Imsi)
    P.data = np.array(df_Face)

    # data[0,1,2,4,6]=[位置，经度，纬度，目标，时间]
    C.dic = data2dic(C.data)
    P.dic = data2dic(P.data)
    # dic[目标]=[[位置,时间,经度,纬度]]

    k = 5e-6
    lmda = -5

    prob_list = []

    i = 0
    for p in P.dic:
        i = i + 1
        for c in C.dic:
            s = Pro(k, P.dic[p], C.dic[c],lmda,device_num)
            prob_list.append([p, c, s])
        if i % 10 == 0:
            print('\r', 'now calculating prob for person NO.' + str(i), end='', flush=True)

    pro_df = pd.DataFrame(prob_list, columns=['FaceLabel', 'Code', 'Prob'])
    return pro_df


