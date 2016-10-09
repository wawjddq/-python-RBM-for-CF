# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:02:35 2016

@author: ddq
RBM for CF using numpy
"""
from __future__ import print_function
import numpy as np
import utils_RBM 


class RBM_CF(object):
    W = None
    hbias = None
    vbias = None
   # params = []
    def __init__(
        self,
        input=None,
        n_visible=200,
        K = 5, 
        n_hidden=200,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None
    ):
        '''
        n_visible表示可见单元的数目
        n_hidden表示隐藏单元的数目
        k表示评分区间
        w表示v和h层的权重，w是一个n_visible*k*n_hidden的三维矩阵
        hbias表示隐藏单元的偏置，是一个元素数量为n_hidden的向量
        vbias表示显层单元的偏置，是一个n_visible*k的矩阵
        numpy_rng是产生随机数的产生器
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.K = K;
        if numpy_rng is None:
            # create a number generator
            #产生1234个方法，这些方法是从各种不同类别的分布挑选出来的，每个方法获取一个关键参数后
            #就可以产生服从它的分布的随机数
           numpy_rng = np.random.RandomState(1234)
        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = np.asarray(
                    numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    #设置返回的形状，这里返回的是n_visible*n_hidden
                    size=(n_visible, K, n_hidden))
                    )
        #将隐层单元都初始化为0
        if hbias is None:
           hbias = np.zeros(n_hidden)
        #显层这里为了简便暂时也先初始化为0
        if vbias is None:
           vbias = np.zeros([n_visible,K])
    
        #用户的评分肯定是整数    
        self.input = input
        
        #如果input为空，o则默认用户对所有的评分均为0
        if input is None:
            self.input = np.zeros([n_visible, K])
        
        if RBM_CF.W is None: RBM_CF.W = initial_W
        if RBM_CF.hbias is None: RBM_CF.hbias = hbias
        if RBM_CF.vbias is None: RBM_CF.vbias= vbias
        self.numpy_rng = numpy_rng
       # RBM_CF.params = [self.W, self.hbias, self.vbias]
        
    #计算p(hj = 1|v)，在v确定的条件下，h的状态用logistic回归判断
    #函数返回的是ph[j] = 1的概率
    def propup(self,vis):
        pre_activation = np.zeros(self.n_hidden)
        ph = np.zeros(self.n_hidden)
        for j in range(self.n_hidden):
            for i in range(self.n_visible):
                for k in range(self.K):
                    pre_activation[j] += self.W[i][k][j]*vis[i][k]
            pre_activation[j] += self.hbias[j]
            ph[j] = utils_RBM.sigmoid(pre_activation[j])
        return ph
        
    #计算P(v|h),在h确定的条件下，v的状态用softmax回归来判断
    def propdown(self, hid):
        pv = np.zeros([self.n_visible, self.K])
        for i in range(self.n_visible):
            den = 0
            for k in range(self.K):
                temp = 0
                for j in range(self.n_hidden):
                    temp += (self.W[i][k][j]*hid[j])
                pv[i][k] = np.exp(self.vbias[i][k] + temp)
                den +=  pv[i][k]
            if den: pv[i] /= den
        return pv
    
    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        #在给定v层时，h层的每个单元应该是服从伯努利分布的
        # compute the activation of the hidden units given a sample of
        # the visibles
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(n=1, p=h1_mean, size=h1_mean.shape)
        
        return h1_mean, h1_sample
        
    '''说明numpy.random.binomial(n, p, size=None)
    Samples are drawn from a binomial distribution with specified parameters,
    n trials and p probability of success where n an integer >= 0 and p is
    in the interval [0,1].
    (n may be input as a float, but it is truncated to an integer in use)
    Returns:	samples : ndarray or scalar
    where the values are all integers in [0, n].
    '''  
    def sample_v_given_h(self, h0_sample):
        #给定h层，v层的每个单元服从二项式分布（这里也就相当于k重伯努利分布）
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(n=1, p=v1_mean, size=v1_mean.shape)
        
        return v1_mean, v1_sample
    
    #下面可以进行Gibbs Sampling了
    def gibbs_hvh(self, h0_sample):
        '''
        This function implements one step of Gibbs sampling,
        starting from the hidden state
        '''  
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]
    
    def gibbs_vhv(self, v0_sample):
        '''
        This function implements one step of Gibbs sampling,
        starting from the visible state
        '''  
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [h1_mean, h1_sample, v1_mean, v1_sample]
    
    #通过gibbs采样后我们可以通过reconstruct重构后的分布来拟合原来的model分布
    #从而对参数进行更新
    
    #下面是用CD-k的方法对参数进行更新，每次针对一个样本进行更新
    #lr是学习率，persistent保存的是上一次gibbs_sampling的end_chain,可以作为
    #这次sample的start_chain,k=1表示只需要进行一次gibbs_sampling
    def param_update(self, lr=0.1, persistent=None):
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        v1_mean, v1_sample, h1_mean, h1_sample = self.gibbs_hvh(chain_start)
        persistent = h1_sample
        
        #update参数
        #更新w
        for i in range(self.n_visible):
            for k in range(self.K):
                for j in range(self.n_hidden):
                    '''
                    RBM_CF.W[i][k][j] -= (lr * (ph_sample[j]*ph_mean[j]*self.input[i][k] 
                                        -h1_sample[j]*h1_mean[j]*self.input[i][k]))
                    '''
                    RBM_CF.W[i][k][j] -= (lr * (ph_sample[j]*ph_mean[j]*self.input[i][k] 
                                        -h1_sample[j]*h1_mean[j]*v1_sample[i][k]))
        #更新v层偏置
        RBM_CF.vbias -= (lr*(self.input - v1_sample))
        
        #更新h层偏置
        RBM_CF.hbias -= (lr*(ph_mean - h1_mean))
        return persistent
    
    #对所有visible units进行评分预测，但是RMSE是针对test_set中已经评分的movie的，
    # missing value不算在内
    def make_rec1(self, probability=0.5):
        #计算p(h|v)
        ph = self.propup(self.input)
        hidden = np.zeros(self.n_hidden)
        for i in range(self.n_hidden):
            if ph[i] > probability: hidden[i] = 1
            else: hidden[i] = 0
        pv = self.propdown(hidden)
        #根据预测的v层的分布算出v层的得分期望
        value = np.zeros(self.n_visible)
        for i in range(self.n_visible):
            for k in range(self.K):
                value[i] += int(round(((k+1) * pv[i][k])))
        return value
    
    #评分预测方法二
    def make_rec2(self):
        h1_mean, h1_sample, v1_mean, v1_sample=self.gibbs_vhv(self.input)
        #根据预测的v层的分布算出v层的得分期望
        value = np.zeros(self.n_visible)
        for i in range(self.n_visible):
            for k in range(self.K):
                if v1_sample[i][k] > 0:
                    value[i] = k+1
                    break;
        return value

#下面是对多个RBM进行训练，建立多个user的评分模型，采用minibatch的方法
def train_RBM(learning_rate=0.1, training_epochs=1,
              n_hidden=100,dataset='.\u_data.txt'
             ):
    m, n, dataset = utils_RBM.load_data(dataset)
    persistent = None
    train_set, test_set = utils_RBM.splitData(dataset)
    input = np.zeros([n,5])
    print("******")
   # print(train_set)
    #print(test_set)
    #用训练集对param进行更新
    for k in range(training_epochs):
        #batch_set[i]
        for i in range(len(train_set)):
            for j in range(n):
                value = train_set[i][j]
                if value > 0: input[j][value-1] = 1
            r = RBM_CF(input,n,5,n_hidden)
            persistent = r.param_update(learning_rate,persistent)
        learning_rate *= 0.8  #学习率以0.8的速度衰减
            #print(persistent)
    #用测试集测试误差，计算RMSE
    input = np.zeros([n,5])
    RMSE = 0
    MAE = 0
    n_compare = 0  #表示预测了多少个评分
    for i in range(len(test_set)):
        for j in range(n):
            value = test_set[i][j]
            if value > 0: input[j][value-1] = 1
        r = RBM_CF(input,n,5,n_hidden)
        #预测测试集中第i个用户的得分
        rating = r.make_rec2()
       # print("第%d个用户对电影的实际评分为:"%i)
       # print(r.input)
      #  print("第%d个用户对电影的预测评分为:" % i)
      #  print(rating)
        temp = 0
        temp1 = 0
        for j in range(n):
            for k in range(5):
                #如果用户对第j个电影的实际评分为k，则计算预测的评分与实际评分的差值
                if r.input[j][k] > 0:
                   temp += ((k+1) - rating[j])**2
                   temp1 += (np.abs((k+1) - rating[j]))
                   n_compare += 1
                   break
        #print("第%d个测试用户对电影的评分误差为:%d"%(i,temp))
        #print("第%d个用户的电影比较次数:%d"%(i,n_compare))
        RMSE += temp
        MAE += temp1
    RMSE = np.sqrt(1.0 * RMSE / n_compare)
    MAE = MAE / n_compare
    return RMSE,MAE
    

#最后利用numpy的matplotlib库来画出折线图
from matplotlib import pyplot as plt
def draw(epochs):
    num = epochs+1
    y1 = []
    y2 = []
    x = [i for i in range(1,num)]
    for i in range(1,num):
        RMSE,MAE = train_RBM(0.01,i)
        print("RMSE:%f" % RMSE)
        print("MAE:%f"% MAE)
        y1.append(RMSE)
        y2.append(MAE)
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.plot(x, y1, 'bo-', label='RBM', linewidth=2)
    plt.axis([1, epochs, 0.6, 2.0])
    plt.legend()
  #  plt.ylabel("MAE")
   # plt.plot(x, y2, 'bo-', label='RBM', linewidth=2)
   # plt.axis([1, epochs, 0.6, 2.0])
   # plt.legend()


draw(15)