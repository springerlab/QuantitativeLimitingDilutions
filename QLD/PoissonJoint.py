"""
Author: Amogh Jalihal
Date: 2024-11-18
"""
from scipy.optimize import minimize
from itertools import product
import numpy as np
def PoissonJoint(x, plate, fold):
    """
    Takes a number of "cells" x and a "plate" (pandas DataFrame)  with rows as replicates
    and columns as serial dilutions.
    Returns the likelihood of the number of cells explaining the observed data.
    """
    rowterm = 1
    denominator = 0

    ## Construct the joint probability
    for _, row in plate.iterrows():
        numerator = 1
        for i, well in enumerate(row.values):
            if well == 0:
                numerator = numerator*(np.exp(-x/np.power(fold,i)))
            else:
                numerator = numerator*(1-np.exp(-x/np.power(fold, i)))
        rowterm = rowterm*(numerator) #/denominator) 
    return(-np.log(rowterm))


def quantifyInputFromSerialDilution(serialDilutionTable, foldDilution=10, 
                                    initialGuess=1.0,
                                    maxCellRange=30000):
    """
    Takes as input a Serial Dilution Table, a pandas dataframe with
    each row as a replicate, each column as a serial dilution, and values (0,1) indicating 
    whether or not the target was found in the well.
    For example, the following table shows an observed growth pattern from a 10 fold serial dilution
    with 4 replicates
    |   | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
    |---+---+---+---+---+---+---+---+---+---+----+----+---|
    | A | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0  | 0  | 0 |
    | B | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0  | 0  | 0 |
    | C | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0  | 0  | 0 |
    | D | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0  | 0  | 0 |
    quantifyInput maximizes the Joint Possion likelihood to estimate the number of targets that can produce
    the data above.
    """
    sol = minimize(PoissonJoint, initialGuess, method="Nelder-Mead",args=(serialDilutionTable, foldDilution))
    lower = np.nan
    upper = np.nan
    if sol.success:
        X = np.linspace(1,maxCellRange,maxCellRange)
        Y = np.exp(-PoissonJoint(X, serialDilutionTable, fold=foldDilution))
        total_area = sum(Y)
        lower, upper, area = get_ci(sol.x[0], max(Y), X, Y, serialDilutionTable, 
               total_area, fold=foldDilution)
    return(sol, lower, upper)
    
def get_ci(xmax, pmax, X, Y, plate, total_area, fold):
    """
    Estimate 95% Confidence Interval bounds for the full probability distribution.
    xmax : estimated count from maximum likeihood estimation
    pmax : likelihood of xmax
    X : list of range of counts over which to estimate the 95%CI. 
    Y : list of liklihoods associated with X.
    total_area : Area under the distribution over X
    fold : fold dilution
    """
    area_within_lim = 0
    xlow, xhigh = xmax, xmax
    currentp = pmax
    while area_within_lim < 0.95*total_area:
        currentp = 0.95*currentp
        scanidx = [idx for idx, (x,y) in enumerate(zip(X,Y))\
                   if abs(y - currentp) < np.power(10., -int(-np.log10(pmax))-3)]
        lowidx, highidx = scanidx[0], scanidx[-1]
        area_within_lim  = sum([Y[i] for i in range(lowidx, highidx)])
    ax.plot([X[lowidx],X[highidx]], [Y[lowidx],Y[highidx]],color="C0")
    return(X[lowidx], X[highidx], area_within_lim)
