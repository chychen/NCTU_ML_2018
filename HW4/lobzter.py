
# coding: utf-8

# # Logistic Regression 
#  
# - **INPUT**
#     - n (number of data point, D)
#     - mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 (m: mean, v: variance)
#     
# - **FUNCTION**
#     - Generate n data point: D1= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, where x and y are independently sampled from N(mx1, vx1) and N(my1, vy1) respectively. (use the Gaussian random number generator you did for homework 3.).
#     - Generate n data point: D2= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, where x and y are independently sampled from N(mx2, vx2) and N(my2, vy2) respectively. 
#     - Use Logistic regression to separate D1 and D2. You should implement both Newton's and steepest gradient descent method during optimization
#         - When the Hessian is singular, use steepest descent for instead. 
#         - You should come up with a reasonable rule to determine convergence. (a simple run out of the loop should be used as the ultimatum) 
#     
# - **OUTPUT** 
#     - The confusion matrix and the sensitivity and specificity of the logistic regression applied to the training data D.
# 
# 

# ### Gaussian random number generator

# In[3]:

import math
import random

def gaussianRandom(mean, varience):
    res = 0.0
    for i in xrange(12):
        res += random.uniform(0, 1)
    res -= 6.0
    return res * math.sqrt(varience) + mean


# $$
#    Activation\ fuction\ (Sigmoid\ function)\quad\quad p(x) = \frac{1}{1+e^{-w^Tx}}
# $$

# $$
#     \Delta f = \sum_{i=1}^{n}(\frac{1}{1+e^{-W^{T}x_{i}}}-y_{i})x_{i} = X^T(\frac{1}{1+e^{-W^{T}X}}-y)
# $$

# $$
#     H(f) = \sum_{i=1}^np(x_{i})(1-p(x_{i}))x_{i}x_{i}^T = X^TSX\quad,\quad S = diag(p(x_{i})(1-p(x_{i})))
# $$

# $$
#     Steepest\ gradient\ descent \quad\quad W_{k+1} = W_k - \lambda\frac{\Delta f}{\lVert \Delta f \rVert}
# $$

# $$
#     Newton's\ method \quad\quad W_{k+1} = W_k - H^{-1}(f) \Delta(f)
# $$

# In[36]:

class LogisticRegression:
    
    def __init__(self, n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
        self.n = n
        self.D = []
        self.z = []
        self.w = [0.0, 0.0]
        
        # generate two data point sets
        for i in xrange(n):
            x1 = gaussianRandom(mx1, vx1)
            y1 = gaussianRandom(my1, vy1)
            x2 = gaussianRandom(mx2, vx2)
            y2 = gaussianRandom(my2, vy2)
            self.D.append([x1, y1])
            self.D.append([x2, y2])
            self.z.append(0.0)
            self.z.append(1.0)
    
    def sigmoid(self, x):
        res = x[0] * self.w[0] + x[1] * self.w[1]
        res = math.exp(-res)
        res = 1.0 + res
        res = 1.0 / res
        
        return res
        
    def gradient(self, activation):
        activation_err = [activation[i] - self.z[i] for i in xrange(2 * self.n)]
        
        res = [0.0, 0.0]
        for i in xrange(2 * self.n):
            res[0] += self.D[i][0] * activation_err[i]
            res[1] += self.D[i][1] * activation_err[i]
        
        # normalize
        len_res = math.sqrt(res[0] * res[0] + res[1] * res[1])
        res[0] /= len_res
        res[1] /= len_res
            
        return res
    
    def hessian(self, activation):
        diag = [activation[i] * (1.0 - activation[i]) for i in xrange(2 * self.n)]
        
        ATD = [[diag[i] * self.D[i][0], diag[i] * self.D[i][1]] for i in xrange(2 * self.n)]
        ATDA = [[0.0, 0.0], [0.0, 0.0]]
        
        
        for i in xrange(2 * self.n):
            ATDA[0][0] += ATD[i][0] * self.D[i][0]
            ATDA[0][1] += ATD[i][0] * self.D[i][1] 
            ATDA[1][0] += ATD[i][1] * self.D[i][0] 
            ATDA[1][1] += ATD[i][1] * self.D[i][1] 
        
        return ATDA
        
    def steepestGradientDecent(self, g):
        self.w[0] -= 0.1 * g[0]
        self.w[1] -= 0.1 * g[1]
        
    def netwonsMethod(self, H, g):
        det = (H[0][0] * H[1][1]) - (H[0][1] * H[0][1])
        Hinverse = H
        Hinverse[0][0] = H[1][1] / det
        Hinverse[0][1] = H[0][1] / (-det)
        Hinverse[1][0] = H[1][0] / (-det)
        Hinverse[1][1] = H[0][0] / det
        
        res = [0.0, 0.0]
        res[0] = Hinverse[0][0] * g[0] + Hinverse[0][1] * g[1]
        res[1] = Hinverse[1][0] * g[0] + Hinverse[1][1] * g[1]
        
        self.w[0] -= res[0]
        self.w[1] -= res[1]
        
    def optimize(self):
        
        for k in xrange(100):
            print "w:",
            print self.w
            activation = [self.sigmoid(self.D[i]) for i in xrange(2 * self.n)]
            
            prediction = [0.0 if activation[i] < 0.5 else 1.0 for i in xrange(2 * self.n)]
            
            # confusion matrix
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in xrange(2 * self.n):
                if prediction[i] == 0.0 and self.z[i] == 0.0:
                    TP += 1
                if prediction[i] == 0.0 and self.z[i] == 1.0:
                    FP += 1
                if prediction[i] == 1.0 and self.z[i] == 0.0:
                    FN += 1
                if prediction[i] == 1.0 and self.z[i] == 1.0:
                    TN += 1
            
            print "TP:",
            print TP
            print "FP:",
            print FP
            print "FN:",
            print FN
            print "TN:",
            print TN
            
            sensitivity = float(TP) / float(TP + FN)
            specificity = float(TN) / float(TN + FP)
            accuracy = float(TP + TN) / float(TP + FP + FN + TN)
            
            print "sensitivity:",
            print sensitivity
            print "specificity:",
            print specificity
            print "accuray:",
            print accuracy
            
            if accuracy > 0.85:
                break
            
            # gradient descent
            g = self.gradient(activation)
            H = self.hessian(activation)

            det = (H[0][0] * H[1][1]) - (H[0][1] * H[0][1])
            if det == 0:
                print "steepestGradientDecent"
                self.steepestGradientDecent(g)
            else:
                print "netwonsMethod"
                self.netwonsMethod(H, g)
        


# In[40]:

model = LogisticRegression(1000, -2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0)
model.optimize()                           


# # EM algorithm
# - **INPUT** 
#     - MNIST training data and label sets.
# - **FUNCTION**
#     - Binning the gray level value into two bins. Treating all pixels as random variables following Bernoulli distributions. Note that each pixel follows a different Binomial distribution independent to others.
#     - Use EM algorithm to cluster each image into ten groups. You should come up with a reasonable rule to determine convergence. (a simple run out of the loop should be used as the ultimatum) 
# - **OUTPUT**
#     - For each digit, output a confusion matrix and the sensitivity and specificity of the clustering applied to the training data.

# $$
# Exceptaition\ step\quad\quad z_{n,k} = \frac{\pi_k\prod_{i=1}^{D}\mu_{k,i}^{x_{n,i}}(1-\mu_{k,i})^{1-x_{n,i}}}{\Sigma_{m=1}^{k}\pi_m\prod_{i=1}^{D}\mu_{k,i}^{x_{n,i}}(1-\mu_{k,i})^{1-x_{n,i}}}
# $$

# $$Maximization\ step\quad\quad  N_m = \Sigma_{n=1}^{N}z_{n,m}$$
# $$\mu_m = \frac{1}{N_m}\Sigma_{n=1}^{N}z_{n,m}\mu_{n}$$
# $$\pi_m = \frac{N_m}{N}$$

# ### read MNIST dataset and bin into two bins

# In[45]:

import gzip
import struct

dir_path = './'
X_train_path = dir_path + 'train-images.idx3-ubyte'
y_train_path = dir_path + 'train-labels.idx1-ubyte'
X_test_path = dir_path + 't10k-images.idx3-ubyte'
y_test_path = dir_path + 't10k-labels.idx1-ubyte'

# read files
y_train = []
with open(y_train_path, 'rb') as f:
    y_train_magic, y_train_size = struct.unpack(">II", f.read(8))
    for idx in xrange(y_train_size):
        label = ord(f.read(1))
        y_train.append(label)

X_train = []
with open(X_train_path, 'rb') as f:
    X_train_magic, X_train_size, X_train_row, X_train_col = struct.unpack(">IIII", f.read(16))
    for idx in xrange(X_train_size):
        img_px = []
        for pxIdx in xrange(X_train_row * X_train_col):
            grey_scale = ord(f.read(1))
            grey_scale = grey_scale * 2 / 256 
            img_px.append(grey_scale)
        X_train.append(img_px)

# y_test = []
# with gzip.open(y_test_path, 'rb') as f:
#     y_test_magic, y_test_size = struct.unpack(">II", f.read(8))
#     for idx in xrange(y_test_size):
#         label = ord(f.read(1))
#         y_test.append(label)

# X_test = []
# with gzip.open(X_test_path, 'rb') as f:
#     X_test_magic, X_test_size, X_test_row, X_test_col = struct.unpack(">IIII", f.read(16))
#     for i in xrange(X_test_size):
#         img_px = []
#         for pxIdx in xrange(X_test_row * X_test_col):
#             grey_scale = ord(f.read(1))
#             grey_scale = grey_scale * 2 / 256 
#             img_px.append(grey_scale)
#         X_test.append(img_px)


# In[43]:

### output with two bins
for i in xrange(28):
    for j in xrange(28):
        print X_train[10][i*28+j],
    print ""


# In[ ]:

cluster_n = 10
px_n = 784
img_n = 100
alpha = 0.00000008

pi = [0.1 for i in xrange(cluster_n)]
mu = [[0.5 for i in xrange(px_n)] for j in xrange(cluster_n)]
z  = [[0.1 for i in xrange(cluster_n)] for j in xrange(img_n)]

for idx in xrange(20):

    Nm = [0 for i in xrange(cluster_n)]
    mean = [[0.0 for i in xrange(px_n)] for j in xrange(cluster_n)]
    predict = [-1 for i in xrange(img_n)]
    
    for n in xrange(img_n):

        # Expectation step
        for k in xrange(cluster_n):
            for i in xrange(px_n):
                z[n][k] += X_train[n][i] * math.log(mu[k][i]) + (1 - X_train[n][i]) * math.log(1.0 - mu[k][i])
            z[n][k] += math.log(pi[k])

        z_0 = z[n][0]
        # divided by first element
        temp_max = max(z[n])
        temp_max_idx = 0
        for k in xrange(cluster_n):
            if temp_max == z[n][k]:
                temp_max_idx = k

        for k in xrange(cluster_n):
            z[n][k] -= z[n][temp_max_idx]
            
        for k in xrange(cluster_n):
            try:
                z[n][k] = math.exp(z[n][k])
            except:
                z[n][k] = 0.0
            print(z[n][k])
        
        # normalize
        z_sum = sum(z[n])
        z[n] = [z_k / z_sum for z_k in z[n]]

        # predict
        temp_max = 0
        predict_k = 0
        for k in xrange(1, cluster_n):
            if z[n][k] > temp_max:
                temp_max = z[n][k]
                predict_k = k
                
        predict[n] = predict_k

        # Maximization step
        for k in xrange(cluster_n):
            Nm[k] += z[n][k]
            for i in xrange(px_n):
                mean[k][i] += z[n][k] * X_train[n][i]
    
    for k in xrange(cluster_n):
        pi[k] = float(Nm[k] + alpha) / float(img_n + alpha * cluster_n)
        for i in xrange(px_n):
            mean[k][i] += alpha
            mean[k][i] /= Nm[k] + alpha * img_n

            
    mu = mean


# In[65]:

### output with two bins
for k in xrange(cluster_n):
    for i in xrange(28):
        for j in xrange(28):
            if mu[k][i*28+j] > 0.5:
                print " ",
            else:
                print 0,
        print ""
    print "--"*28

