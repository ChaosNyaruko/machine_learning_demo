import numpy as np
from astropy.units import Ybarn
import math

def computeCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    varX = 0
    varY = 0
    SSR = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    return SSR/SST

def polyfit(x, y, degree):
    results = {} 
    coeffs = np.polyfit(x, y, degree)
    print("coeffs:",coeffs)

    results['polynomial'] = coeffs.tolist()
    
    p = np.poly1d(coeffs)
    print("p(after poly1d):", p)

    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    results['determination'] = ssreg/sstot

    return results


testX = [1,3,8,7,9]
testY = [10,12,24,21,34]

print("r=", computeCorrelation(testX, testY))
print("R^2=",str(computeCorrelation(testX, testY) ** 2))

print(polyfit(testX, testY, 1)["determination"])

