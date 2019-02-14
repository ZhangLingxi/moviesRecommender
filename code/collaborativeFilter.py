import scipy.io as sio
import numpy as np
from scipy import optimize

data = sio.loadmat('../data/ex8_movies.mat')
Y = data['Y'] # 1682x943，1682 movies*943 users
R = data['R'] # R(i,j) = 1 when user j gave a rating to movie i
params = sio.loadmat('../data/ex8_movieParams.mat')
X = params['X']
Theta = params['Theta']
num_users = params['num_users']
num_movies = params['num_movies']
num_features = params['num_features']

#导入电影数据
def load_data(filename):
    movieList = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(' ')
        col_num = len(lineArr)
        temp = ''
        for i in range(col_num):
            if i == 0:
                continue
            temp = temp + lineArr[i] + ' '
        movieList.append(temp)
    return movieList

# 没有x0=1,也没有除以m
def computeCost(x, theta, y, r, reg=0):
    j = 0
    x_grad = np.zeros_like(x)
    theta_grad = np.zeros_like(theta)

    j_temp = (x.dot(theta.T) - y) ** 2
    j = 0.5 * np.sum(j_temp[r == 1]) + 0.5 * reg * np.sum(theta ** 2) + 0.5 * reg * np.sum(x ** 2)

    x_grad = np.dot(((x.dot(theta.T) - y) * r), theta) + reg * x
    theta_grad = np.dot(((x.dot(theta.T) - y) * r).T, x) + reg * theta

    return j, x_grad, theta_grad

# 计算数值梯度
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  
        fxph = f(x)  # f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # f(x - h)
        x[ix] = oldval  
        grad[ix] = (fxph - fxmh) / (2 * h) 
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

# 进行数值归一化
def normalizeRatings(y, r):
    m, n = y.shape
    ymean = np.zeros((m, 1))
    ynorm = np.zeros_like(y)
    for i in range(m):
        idx = np.where(r[i, :] == 1)
        ymean[i] = np.mean(y[i, idx])
        ynorm[i, idx] = y[i, idx] - ymean[i]
    return ymean, ynorm

movieList = load_data('../data/movie_ids.txt')

#新用户数据
my_ratings = np.zeros((1682,1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# 将新的用户的评分数据放入数据集合,进行训练
YY = np.hstack((my_ratings, Y))
RR = np.hstack((my_ratings != 0, R))
Ymean, Ynorm = normalizeRatings(YY, RR)
num_users1 = YY.shape[1]
num_movies1 = YY.shape[0]
num_features1 = 10
XX = np.random.randn(num_movies1, num_features1)
TTheta = np.random.randn(num_users1, num_features1)
args = (Ynorm, RR, num_users1, num_movies1, num_features1, 1.5)
params = np.hstack((XX.ravel(), TTheta.ravel())).ravel()

def Cost(params, *args):
    y, r, nu, nm, nf, reg = args
    j = 0
    x = params[0:nm * nf].reshape(nm, nf)
    theta = params[nm * nf:].reshape(nu, nf)
    j_temp = (x.dot(theta.T) - y) ** 2
    j = 0.5 * np.sum(j_temp[r == 1]) + 0.5 * reg * np.sum(theta ** 2) + 0.5 * reg * np.sum(x ** 2)
    return j

def grad(params, *args):
    y, r, nu, nm, nf, reg = args
    x = params[0:nm * nf].reshape(nm, nf)
    theta = params[nm * nf:].reshape(nu, nf)
    x_grad = np.zeros_like(x)
    theta_grad = np.zeros_like(theta)
    x_grad = np.dot(((x.dot(theta.T) - y) * r), theta) + reg * x
    theta_grad = np.dot(((x.dot(theta.T) - y) * r).T, x) + reg * theta
    g = np.hstack((x_grad.ravel(), theta_grad.ravel())).ravel()
    return g

res = optimize.fmin_cg(Cost, x0=params, fprime=grad, args=args, maxiter=100)
#print(res)
bestX = res[0:num_movies1*num_features1].reshape(num_movies1,num_features1)
bestTheta = res[num_movies1*num_features1:].reshape(num_users1,num_features1)

#预测分数
score = bestX.dot(bestTheta.T) + Ymean
my_score = score[:, 0] 

# 排序，推荐最高的分数的电影给新用户
sort_index = my_score.argsort()
favorite = 10
for i in range(favorite):
    print ("推荐%d：%s，分数%d" \
          %(i+1,movieList[sort_index[-(i+1)]],my_score[sort_index[-(i+1)]]))

