import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 500

mu1 = np.array([2,5])
mu2 = np.array([8,1])
mu3 = np.array([5,3])
means = [None] * 3
means[0] = mu1
means[1] = mu2
means[2] = mu3

cov1 = np.array([[2,0],[0,2]])
cov2 = np.array([[3,1],[1,3]])
cov3 = np.array([[2,1],[1,2]])
covariances = [None] * 3
covariances[0] = cov1
covariances[1] = cov2
covariances[2] = cov3

class1 = np.random.multivariate_normal(mu1, cov1, N)
class2 = np.random.multivariate_normal(mu2, cov2, N)
class3 = np.random.multivariate_normal(mu3, cov3, N)

data = np.concatenate([class1, class2 , class3 ], axis=0)

color = ['blue','red','green']

def plot_true_2d(x,Mu,sigma):
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    fig1.suptitle('2D true density')
    
    f0 = np.linspace(x[:,0].min(),x[:,0].max())
    f1 = np.linspace(x[:,1].min(),x[:,1].max())
    X, Y = np.meshgrid(f0,f1)
    
    for c in range(3):
        def pdf(point):
            part1 = 1 / (2* np.pi) * (np.linalg.det(sigma[c])**(1/2))
            part2 = (-1/2) * ((point - Mu[c]).T @ (np.linalg.inv(sigma[c]))) @((point-Mu[c]))
            return float(part1 * np.exp(part2))
        z = np.array([pdf(np.array(ponit)) for ponit in zip(np.ravel(X),np.ravel(Y))])
        Z = z.reshape(X.shape)
        ax1.contour(X, Y, Z,colors=color[c])
        
def plot_true_3d(x,Mu,sigma):

    fig2 = plt.figure(figsize=(6,6))
    ax2 = fig2.add_subplot(projection = '3d')
    fig2.suptitle('3D true density')
    ax2.view_init(10,-100)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("P(X)")
    
    f0 = np.linspace(x[:,0].min(),x[:,0].max())
    f1 = np.linspace(x[:,1].min(),x[:,1].max())
    X, Y = np.meshgrid(f0,f1)

    for c in range(3):
        def pdf(point):
            part1 = 1 / (2* np.pi) * (np.linalg.det(sigma[c])**(1/2))
            part2 = (-1/2) * ((point - Mu[c]).T @ (np.linalg.inv(sigma[c]))) @((point-Mu[c]))
            return float(part1 * np.exp(part2))
        z = np.array([pdf(np.array(ponit)) for ponit in zip(np.ravel(X),np.ravel(Y))])
        Z = z.reshape(X.shape)
        ax2.contour3D(X, Y, Z,60, colors=color[c])
        

def area(data):
    size = 30
    X = []
    a = min(data[:,0].min(),data[:,1].min())
    b = max(data[:,0].max(),data[:,1].max())
    
    for i in range(data.shape[1]):
        x=np.linspace(a,b, size)
        X.append(x)
    return np.array(X)

def knn(data, k):
    X = area(data)
    size = [len(X[0]), len(X[1])]
    knnpdf = np.zeros(size)
    n=len(data)
    for i in range(size[0]):
        for j in range(size[1]):
            x = np.array([X[0][i],X[1][j]])
            ds = [np.linalg.norm(x-y) for y in data]
            ds.sort()
            v = math.pi*ds[k-1]*ds[k-1]
            if v == 0:
                knnpdf[i,j] = 1
            else:
                knnpdf[i,j] = k/(n*v)
    return X, knnpdf


def plotknn(class1,class2,class3):   
    k_set = [1,9,99]

    fig2d = plt.figure()
    fig3d = plt.figure()
    fig2d.suptitle('2D KNN')
    fig3d.suptitle('3D KNN')

    pos=1

    for k in k_set :
        title = "k = %d" % (k)
        print(title,"waiting...")
        ax2d = fig2d.add_subplot(1, 3, pos)
        ax3d = fig3d.add_subplot(1, 3, pos,projection='3d')
        ax3d.view_init(10,-100)
        pos = pos + 1
        
        for i in range(0,3):
            if(i==0):
                X,P = knn(class1, k)
                px, py = np.meshgrid(X[0], X[1])
                ax2d.contour(px, py, P,colors=color[i])
                ax3d.plot_surface(px, py,P,alpha=.3,rstride=1,cstride=1,color=color[i],edgecolor='none')
                ax3d.contour3D(px, py,P,60, colors=color[i])
            
            if(i==1):
                X,P = knn(class2, k)
                px, py = np.meshgrid(X[0], X[1])
                ax2d.contour(px, py, P,colors=color[i])
                ax3d.plot_surface(px, py,P,alpha=.3,rstride=1,cstride=1,color=color[i],edgecolor='none')
                ax3d.contour3D(px, py,P,60, colors=color[i])
                
            if(i==2):
                X,P = knn(class3, k)
                px, py = np.meshgrid(X[0], X[1])
                ax2d.contour(px, py, P,colors=color[i])
                ax3d.plot_surface(px, py,P,alpha=.3,rstride=1,cstride=1,color=color[i],edgecolor='none')
                ax3d.contour3D(px, py,P,60, colors=color[i])
            

        ax2d.set_xlabel('X1')
        ax2d.set_ylabel('X2')
        ax2d.set_title(title)
        ax3d.set_zlabel('P(X)')
        ax3d.set_ylabel('Y')
        ax3d.set_xlabel('X')
        ax3d.set_title(title)

plot_true_2d(data,means,covariances)
plot_true_3d(data,means,covariances)
plotknn(class1,class2,class3)
plt.show()