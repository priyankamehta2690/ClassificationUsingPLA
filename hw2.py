import numpy as np
import random 

def main():
    N=input("Enter the value of N")
    X,Y=generateData(N)
    #calculating the number of iterations with weight vector initialized to 0
    w,iters=pla(X,Y)
    print w,iters
    #calculating the number of iterations with weight vector initialized using Linear Regression
    w=pseudoinverse(X, Y)
    w,iters=pla(X,Y,w)
    print w,iters    
    
#generates N*2 matrix of uniformly sampled points and N*1 vector of corresponding labels    
def generateData(N):
    Y=[[0 for x in range(1)] for x in range(N)] 
    X=np.random.uniform(-1.0,1.0,(N,2)) 
    f=np.random.uniform(-1.0,1.0,(2,2))
    #taking the cross product of target function and each point in the 
    #N*2 matrix to map each point to one of the two sides of the line
    for i in range(0,N):
        matrix=[[(f[1,0]-f[0,0]),(X[i,0]-f[0,0])],
                 [(f[1,1]-f[0,1]),(X[i,1]-f[0,1])]]
        if(np.linalg.det(matrix)>0):
            Y[i]=1
        elif(np.linalg.det(matrix)<0):
            Y[i]=-1
    return X,Y      

#returns a learned weight vector and number of iterations for the perceptron learning algorithm
#w0 is the (optional) initial set of weights, initialized to 0
def pla(X, Y, w0=np.zeros(3)):
    iters=0
    w=np.array([0,w0[0],w0[1]])
    x=[[0 for x in range(3)] for x in range(len(X))] 
    #adding artificial coordinate x0=1 
    for i in range(0,len(X)):
        x[i][0]=1
        x[i][1]=X[i,0]
        x[i][2]=X[i,1]
    misClassifiedPoints=calculateMisclassifiedPoints(x,Y,w)
    #picking a misclassified point randomly and updating the weight vector
    while(len(misClassifiedPoints)>0):
        iters=iters+1
        randomIndex = random.randint(0,len(misClassifiedPoints)-1)
        misClassifiedPoint=misClassifiedPoints[randomIndex]
        w=w+np.dot(Y[misClassifiedPoint],x[misClassifiedPoint])
        misClassifiedPoints=calculateMisclassifiedPoints(x,Y,w)
    return w,iters

#returns the learned weight vector for the pseudoinverse algorithm for linear regression
def pseudoinverse(X, Y):
    xT=np.matrix.transpose(X)
    pseudoInverse=np.dot(np.linalg.inv(np.dot(xT,X)),xT)
    w=np.dot(pseudoInverse,Y)
    return w

#finds out all the misclassified points for a particular value of weight vector    
def calculateMisclassifiedPoints(X,Y,w):
    misClassifiedPoints=[]
    for i in range(0,len(X)):
        wT=np.matrix.transpose(w)
        product=np.sign(np.dot(wT,X[i]))
        if(Y[i] != product):
            misClassifiedPoints.append(i)
    return misClassifiedPoints
    
if __name__=="__main__":
    main()
