"""
Author: Amogh Jalihal
Date: 2024-11-18
"""
from scipy.optimize import minimize
from itertools import product
import numpy as np
import matplotlib.pyplot as plt 
from numpy.random import binomial
import pandas as pd
def PoissonJoint(x, plate, fold):
    """
    Takes a number of "cells" x and a "plate" (pandas DataFrame)  with rows as replicates
    and columns as serial dilutions.
    Returns the negative log likelihood of the number of cells explaining the observed data.
    """
    loglikelihood = 0
    ## Construct the joint probability
    loglikelihood = 0
    for _, replicate in plate.iterrows():
        for dilution, growth in enumerate(replicate.values):
            if growth == 0:
                v_rd = 0
                term = - (x/np.power(fold, dilution))
            else:
                term =  np.log(1 - np.exp(-x/np.power(fold, dilution))) 
            loglikelihood = loglikelihood + (term)
    return(-loglikelihood)



def PoissonJointSecondDerivative(x, plate,fold):
    """
    Takes a number of "cells" x and a "plate" (pandas DataFrame)  with rows as replicates
    and columns as serial dilutions.
    Returns the second derivative of the loglikelihood function. 
    When evaluated at the maximum likelihood estimate, this gives the (lower bound) of the variance.
    """
    second = 0
    for _, replicate in plate.iterrows():
        for dilution, growth in enumerate(replicate.values):
            if growth == 0:
                v_rd = 0
            else:
                v_rd = 1

            if v_rd == 0:
                term = 0 
            else:
                term = -v_rd*np.exp(-x/fold**dilution)/(fold**(2*dilution)*(1 - np.exp(-x/fold**dilution)))\
                    - v_rd*np.exp(-2*x/fold**dilution)/(fold**(2*dilution)*(1 - np.exp(-x/fold**dilution))**2)
            second = second + (term)

    return(second)


def quantifyInputFromSerialDilution(serialDilutionTable, foldDilution=10, 
                                    initialGuess=1.0,
                                    maxCellRange=30000,visualize=False):
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
    for attempt in range(10):
        sol = minimize(PoissonJoint,np.power(10, attempt), method="Nelder-Mead",args=(serialDilutionTable, foldDilution))
        if sol.success:
            break

    lower = np.nan
    upper = np.nan
    MLE = 0
    if sol.success:
        MLE = sol.x[0]
        try:
            variance = -1/PoissonJointSecondDerivative(MLE, serialDilutionTable, foldDilution)
            
        except:
            variance = np.nan
        lower, upper = MLE - 1.96*np.sqrt(variance), MLE + 1.96*np.sqrt(variance)
        # lower, upper, area, ci_success = get_ci(sol.x[0], max(Y), X, Y, serialDilutionTable, 
        #        total_area, fold=foldDilution, ax=ax)
        ax= None
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        if visualize:
            ax.plot(X, Y)
            ax.axvline(sol.x[0])
            ax.set_xlabel("cell counts")
            ax.set_ylabel("Prob")
            if (not np.isnan(lower)) and ((not np.isnan(lower))):
                ax.set_title(f"Estimate={round(sol.x[0])},95%CI=[{round(lower)}, {round(upper)}]")
        plt.show()

    return(MLE, lower, upper, variance)

def get_ci_numerical(xmax, pmax, X, Y, plate, total_area, fold, maxiter=1000,ax=None):
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
    eps = np.power(10., -int(-np.log10(pmax))-1.5)
    itercount = 0
    while area_within_lim < 0.95*total_area:
        currentp = currentp - 0.05*pmax
        scanidx = [idx for idx, (x,y) in enumerate(zip(X,Y))\
                   if abs(y - currentp) < eps]
        if len(scanidx) == 0:
            return(np.nan, np.nan, np.nan, False)

        lowidx, highidx = scanidx[0], scanidx[-1]
        area_within_lim  = sum([Y[i] for i in range(lowidx, highidx)])
        itercount +=1
        # if ax is not None:
        #     if (X[lowidx] < xlow ) and (X[highidx] > xmax):
        #         ax.plot([X[lowidx],X[highidx]],[Y[lowidx],Y[highidx]],alpha=0.1, color="k")
        if itercount > maxiter:
            return(np.nan, np.nan, np.nan, False)
    return(X[lowidx], X[highidx], area_within_lim, True)    

def simulate_plate(ideal_initial, rows, 
                   numdilutions = 12,
                   fold=10):
    initial = ideal_initial
    plate = []
    for i in range(rows):
        initial = ideal_initial
        #initial = int(ideal_initial*(1 + 1./(fold-1)))
        # initial = ideal_initial
        row = [initial]
        ## do noisy serial dilutions
        for i in range(numdilutions - 1):
            row.append( binomial(row[-1], float(1./fold)))
            # if i == 0:
            #     row[0] = initial*2 - row[-1]
            row[-2] = row[-2] - row[-1]
        plate.append(row)
    return(pd.DataFrame(plate,
                        columns=list(range(1,numdilutions+1)),
                        index=pd.Index(list(range(1,rows+1)))))
