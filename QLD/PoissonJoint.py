"""
Author: Amogh Jalihal
Date: 2024-11-18
History:
2026-08-19 Cleaned up alternate model implementations
"""
from scipy.optimize import minimize, Bounds
from itertools import product
import numpy as np
from math import factorial
#import matplotlib.pyplot as plt 
from numpy.random import binomial
import pandas as pd
import warnings

def choose(n, r):
    return(factorial(n)/(factorial(r)*(factorial(n-r))))

def dilutionmodel_ml(L, growth, fold, discardFinal=False):
    """
    Accurate model of a serial dilution.
    The main difference is the cells left over in a well:
    The Poisson model assumes the number of cells at dilution i is L/fold^i
    In reality, a fraction of those cells have been moved to the next well.
    Thus, at the end of the serial dilution, the number of cells at dilution i is
    L/(fold^i) - L/(fold^(i+1)) = (L/(fold^i))* (1 - 1/fold).
    This multiiplicative term becomes insignificant as the fold dilution grows.

    """
    ## Validate input
    if fold <=0:
        print("Fold dilution cannot be negative or zero.")
        raise ValueError
    if L <=0:
        print("Number of cells cannot be negative or zero.")
        raise ValueError
    d = 1./fold
    numreps = growth.shape[0]
    numdilutions = growth.shape[1]
    positives = growth.sum(axis=0)
    nll = 0
    if discardFinal:
        for dilid, numpos in enumerate(positives):
            ## if numpos > 0
            if numpos == 0 :
                nll +=  -(np.log(choose(numreps, numreps-numpos))\
                - (numreps)*(L*(np.power(d, float(dilid)))*(1.-d)))
            else:
                term_a = np.log(choose(numreps, numreps-numpos))
                term_b = (numreps - numpos)*(L*(np.power(d, float(dilid)))*(1.-d))
                lambd = (np.power(d, float(dilid))*L*(1.-d))
                if lambd == 0:
                    term_c = -np.inf
                else:
                    if lambd <= np.log(2):
                        term_c = numpos*np.log(-np.expm1(-lambd))
                    else:
                        term_c = numpos*np.log1p((-np.exp(- lambd)))
                nll += -(term_a \
                         - term_b\
                         + term_c)
    else:
        for dilid, numpos in enumerate(positives.values[:-1]):
            if numpos == 0 :
                nll +=  -(np.log(choose(numreps, numreps-numpos))\
                - (numreps)*(L*(np.power(d, float(dilid)))*(1.-d)))
            else:
                term_a = np.log(choose(numreps, numreps-numpos))
                term_b = (numreps - numpos)*(L*(np.power(d, float(dilid)))*(1.-d))
                lambd = (np.power(d, float(dilid))*L*(1.-d))
                if lambd == 0:
                    term_c = -np.inf
                else:
                    if lambd <= np.log(2):
                        term_c = numpos*np.log(-np.expm1(-lambd))
                    else:
                        term_c = numpos*np.log1p((-np.exp(- lambd)))
                nll += -(term_a \
                         - term_b\
                         + term_c)

        numposlast = positives.values[-1]
        dilid = dilid +1 ## Increment the last seen dilid, i.e. the last column
        if numposlast == 0:
            nll +=  -(np.log(choose(numreps, numreps-numposlast))\
                      - (numreps)*(L*(np.power(d, float(dilid)))*(1.-d)))
        else:
            term_a = np.log(choose(numreps, numreps-numposlast))
            term_b = (numreps - numposlast)*(L*(np.power(d, float(dilid))))
            lambd = (np.power(d, float(dilid))*L)
            if lambd == 0:
                term_c = -np.inf
            else:
                if lambd <= np.log(2):
                    term_c = numpos*np.log(-np.expm1(-lambd))
                else:
                    term_c = numpos*np.log1p((-np.exp(- lambd)))
            nll += -(term_a \
                     - term_b\
                     + term_c)
    return(nll)

def FisherInformation(mle, plate, fold, logging=False, discardFinal = False):
    """
    Fisher information of the dilution model 
    I_r(n,d) = (1 - d)^{2} Sum{i=1}^D (d^(2 i - 2) r_i) ( ( - e^({- d^{i - 1} x (1 - d)}))/ ((1 - e^({- d^{i - 1} x (1 - d))))^2)
    """
    fi = 0
    numreps = plate.shape[0]
    numdilutions = plate.shape[1]
    positives = plate.sum(axis=0)

    if discardFinal:
        for dilid, numpos in enumerate(positives):
            d = 1./float(fold)
            nume = (-np.exp(-d**(dilid)*mle*(1-d)))
            denom = (1-np.exp(-d**(dilid)*mle*(1-d)))**2
            fi_ = -(1-d)**2 * (d**(2*(dilid))) *numpos * (nume)/(denom)
            fi = fi+ fi_
    else:
        for dilid, numpos in enumerate(positives[:-1]):
            d = 1./float(fold)
            nume = (-np.exp(-d**(dilid)*mle*(1-d)))
            denom = (1-np.exp(-d**(dilid)*mle*(1-d)))**2
            fi_ = -(1-d)**2 * (d**(2*(dilid))) *numpos * (nume)/(denom)
            fi = fi+ fi_

        number_positive = positives.values[-1]
        dilid = dilid + 1
        d = 1./float(fold)
        nume = (-np.exp(-d**(dilid)*mle))
        denom = (1-np.exp(-d**(dilid)*mle))**2
        fi_ =  -(d**(2*(dilid))) *number_positive * (nume)/(denom)
        fi = fi+ fi_
    return(fi)

def FisherInformation_theoretical(mle, plate, fold, discardFinal = False):
    """
    Fisher information is given by
    d_i ^ 2 exp(-mle*d_i)/(1-exp(-mle*d_i))
    """
    fi = 0
    if discardFinal:
        for dilution in plate.columns:
            d = np.power(float(fold), -float(dilution-1))
            fi_ = plate.shape[0]*(d**2) * (np.exp(-mle*d))/(1-np.exp(-mle*d))
            fi = fi+ fi_
    else:
        for dilution in plate.columns:
            d = np.power(float(fold), -float(dilution-1))
            fi_ = plate.shape[0]*(d**2) * (np.exp(-mle*d))/(1-np.exp(-mle*d))
            fi = fi+ fi_
    return(fi)

def quantifyInputFromSerialDilution(serialDilutionTable, foldDilution, 
                                    backFill = True,
                                    logging=False,
                                    visualize=False,
                                    discardFinal = False):
    """
    Main user facing entry point which validates the input and
    uses heuristics to handle various edge cases.

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
    Arguments:
    serialDilutionTable: pandas DataFrame with non negative entries, with 0 denoting no growth, and number > 0 denoting growth, typically 1.
    foldDilution: float, the fold dilution used in the serial transfers. 
    backFill: This copies and error corrects the input growth table. A major assumption with this is the premise of a limiting dilution assay is not violated, namely SHPM still holds. Default=True.
    discardFinal: boolean, indicating if the final well retained the volume transferred into it, or if it was
    mixed in and discarded. Default = False. 
    """
    growthTable = serialDilutionTable > 0
    ## Initialize return values
    lower = np.nan
    upper = np.nan
    variance = np.nan
    MLE = 0

    ### Trivial cases:
    ### Check if growth table is completely empty:
    if sum(growthTable.values.flatten()) == 0:
        return(MLE, lower, upper, variance)
    ### Check if growth table is completely filled:
    if sum(growthTable.values.flatten()) == growthTable.shape[0]*growthTable.shape[1]:
        """
        See logic for estimating lower bound based on number of replicates  below.
        """
        lowerBound = np.log(growthTable.shape[0])*np.power(foldDilution, growthTable.shape[1]-1)
        return(lowerBound, lower, upper, variance)
        
    if backFill:
        """
        Back filling logic sanity checks "holes" in the growth table.
        In realistic data generated from qPCR or plate scan measurements, depending on 
        the specific thresholds used, there can appear holes or no-growth wells randomly
        in the growth table that are obviously not correct. For example = the first dilution has a 
        skipped well when the number of cells is ~1e7. This is clearly a measurement error.

        However, "skipped" wells/no-growth wells are statisticaly valid depending
        on the fold dilution used, and the confidence we have in seeing the no-growth given
        the number of replicates used. 
        
        If we assume that none of the assumptions of LD assays are violated, primarily SHPM,
        then we can confidently backfill holes in the input table given the fold dilution and number of replicates.
        This error correction step helps make the viability assessment pipeline robust to experimental noise.

        """
        ## First get an initial estimate using the full dilution table
        ### BAck filling strategy
        _p = pd.DataFrame(growthTable.sum(axis=0))
        backfilled = False
        if logging:
            outstr = ""
            outstr += "----------------\nWorking on\n"
            outstr += growthTable.to_string()

        for dil in reversed(range(1,len(growthTable.columns))):
            """
            Calculate the growth fractions strating from the end of the plate
            going backwards.
            """
            prevFrac, currFrac = growthTable[growthTable.columns[dil-1]].sum()/growthTable.shape[0],\
                growthTable[growthTable.columns[dil]].sum()/growthTable.shape[0]

            """
            A no growth well at dilution i when all reps have grown in dilution i+1 
            might be suspicious
            """
            if (currFrac == 1) and (prevFrac < currFrac):
                if logging:
                    outstr += f"\nP(growth|{dil+1})={currFrac}, P(growth|{dil}) = {prevFrac}"
                """
                Estimate the minimum number of cells required to produce the observed growth pattern
                at current dilution.
                Since currFrac = 1, this well is uninformative, EXCEPT, we can
                calculate a /lower bound/ of the number of cells.
                Logic is as follows:
                if there were n-1 grown wells, the poisson estimate from this dilution
                in isolation would be:
                (n-1)/n = 1- e^(-lambda)
                => lambda = log(n)  ## This by itself is a neat result.
                BUT, we have n grown wells, so our estimate must be greater than this lambda.
                Thus, our lower bound is log(n) cells for the current dilution.
                """
                lb_count_curr = np.log(growthTable.shape[0]) #approx_count/(foldDilution ** (dil-1))
                
                """
                the expected cell count at dilution i when starting from X cells is
                xhat_i = X d^(i-1) (1-d)
                If we have an estimate for cells at dilution i, we can work back to an estimate at i-1
                xhat_(i-1)/xhat_i = d^(i-2) (1-d)/(d^(i-1) (1-d))
                => xhat_(i-1) = xhat_i/d
                => xhat_(i-1) = xhat_i*fold
                """
                lb_prev_count_expect = lb_count_curr*foldDilution
                if logging:
                    outstr += f"\n\tE({dil}) = {round(lb_count_curr,2)},E({dil-1}) = {round(lb_prev_count_expect,2)}"

                """
                Next, we compute the estimate for the previous dilution from the observations.
                """
                prev_count_observed = -np.log(1-prevFrac)

                """
                We use poisson statistics. If the mean is X, then std of the distribution is sqrt(X).
                We will impose a strict condition that our "observed count" be within 1 std of the 
                inferred count at the previous dilution.
                If this condition is satisfied, then the "dropout" is statistically valid. If not, we 
                reject this datapoint, and back fill the /entire dilution/ with growth. This is an aggressive
                backfill.
                TODO Some variants to consider are:
                1. Fill the minimum number of wells needed to make the previous well statistically valid.
                2. Fill the number of wells needed to match the lb_prev_count_expect.
                """
                if (prev_count_observed > (lb_prev_count_expect - np.sqrt(lb_prev_count_expect)))\
                   and (prev_count_observed < (lb_prev_count_expect + np.sqrt(lb_prev_count_expect))):
                    if logging:
                        outstr += "\nvalid: "
                        outstr += str(round(prev_count_observed,2)) + " " + str(round(lb_prev_count_expect,2))
                else:
                    if logging:
                        outstr += "\ninvalid,backfilling: "
                        outstr += str(round(prev_count_observed,2)) + " " + str( round(lb_prev_count_expect,2))
                    ### Aggressively backfill the entire column.
                    growthTable[growthTable.columns[dil-1]] = True
                    backfilled = True
        if backfilled:
            if logging:
                outstr += "\n" + growthTable.to_string()
                print(outstr)

    b = Bounds(lb=0.01, ub = np.inf)

    for attempt in range(10):
        sol = minimize(dilutionmodel_ml, np.power(10, attempt), method="Nelder-Mead",
                       args=(growthTable, foldDilution, discardFinal), bounds=b)
        if sol.success:
            break
    if not sol.success:
        return(MLE, lower, upper, variance)

    ## Optimization successful
    MLE = sol.x[0]
    variance = 1/FisherInformation(MLE, growthTable, foldDilution, discardFinal=discardFinal)
    

    ## Compute the Wilks likelihood ratio intervals
    nll_mle = dilutionmodel_ml(MLE, growthTable, foldDilution, discardFinal = discardFinal)
    WCUTOFF = 3.841458820694124
    def llr_stat(x, nll_mle, growthTable, foldDilution, discardFinal):
        return(2*(dilutionmodel_ml(x, growthTable, foldDilution, discardFinal = discardFinal) - nll_mle))
    def error(x, nll_mle, growthTable, foldDilution, discardFinal):
        stat= llr_stat(x,nll_mle, growthTable, foldDilution, discardFinal)
        return(abs(stat - WCUTOFF))
    ### compute lower interval
    ### Translting from 95%CI to Chi Squared with 1 degree of freedom:
    ### we want the likelihood ratio statistic W <= 3.8415 

    ## Compute the lower interval
    b = Bounds(lb=1e-5, ub = MLE)
    lowersol = minimize(error, 1e-3, args=(nll_mle, growthTable, foldDilution, discardFinal, ), method="Nelder-Mead", bounds=b)

    interval_type = "Wilks"
    ## Rigorous sanity check to make sure the two different optimizations yield solutions on either side of the MLE
    if lowersol.success:
        lowererr = error(lowersol.x[0], nll_mle, growthTable, foldDilution, discardFinal=False)
        if (lowererr < 1e-4) and (lowersol.x[0] < MLE):
            lower = lowersol.x[0]
    
    b = Bounds(lb=MLE, ub = np.inf)
    uppersol = minimize(error, MLE, args=(nll_mle, growthTable, foldDilution, discardFinal, ), 
                        method="Nelder-Mead", bounds=b)
    if uppersol.success:
        uppererr = error(uppersol.x[0], nll_mle, growthTable, foldDilution, discardFinal=False)
        if (uppererr < 1e-4) and (uppersol.x[0] > MLE):
            upper = uppersol.x[0]

    if (not lowersol.success) and (not uppersol.success):
        ## symmetric Wald interval
        interval_type = "Wald"
        lower, upper = MLE - 1.96*np.sqrt(variance), MLE + 1.96*np.sqrt(variance)

    if visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        X = np.logspace(0,np.log10(MLE)+1, 100)
        Y = [np.exp(-dilutionmodel_ml(x, growthTable, foldDilution, discardFinal=discardFinal)) for x in X]
        ax = fig.add_subplot(1,1,1)
        ax.plot(X, Y)
        ax.axvline(sol.x[0])
        ax.axvline(uppersol.x[0], color='g')
        ax.axvline(lowersol.x[0],color='g')
        ax.axvline(upper, color='r')
        ax.axvline(lower,color='r')
        ax.set_xscale("log")
        ax.set_xlabel("cell counts")
        ax.set_ylabel("likelihood")
        if (not np.isnan(lower)) and ((not np.isnan(upper))):
            ax.set_title(f"Estimate={round(sol.x[0])},95%CI=[{round(lower)}, {round(upper)}]")
        plt.show()
    return(MLE, lower, upper, variance, {"interval_type":interval_type,})

# def get_ci_numerical(xmax, pmax, X, Y, plate, total_area, fold, maxiter=1000,ax=None):
#     """
#     Estimate 95% Confidence Interval bounds for the full probability distribution.
#     xmax : estimated count from maximum likeihood estimation
#     pmax : likelihood of xmax
#     X : list of range of counts over which to estimate the 95%CI. 
#     Y : list of liklihoods associated with X.
#     total_area : Area under the distribution over X
#     fold : fold dilution
#     """
#     area_within_lim = 0
#     xlow, xhigh = xmax, xmax
#     currentp = pmax
#     eps = np.power(10., -int(-np.log10(pmax))-1.5)
#     itercount = 0
#     while area_within_lim < 0.95*total_area:
#         currentp = currentp - 0.05*pmax
#         scanidx = [idx for idx, (x,y) in enumerate(zip(X,Y))\
#                    if abs(y - currentp) < eps]
#         if len(scanidx) == 0:
#             return(np.nan, np.nan, np.nan, False)

#         lowidx, highidx = scanidx[0], scanidx[-1]
#         area_within_lim  = sum([Y[i] for i in range(lowidx, highidx)])
#         itercount +=1
#         # if ax is not None:
#         #     if (X[lowidx] < xlow ) and (X[highidx] > xmax):
#         #         ax.plot([X[lowidx],X[highidx]],[Y[lowidx],Y[highidx]],alpha=0.1, color="k")
#         if itercount > maxiter:
#             return(np.nan, np.nan, np.nan, False)
#     return(X[lowidx], X[highidx], area_within_lim, True)    

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


########################################################
# Alternate model formulations and implementation.     #
# These are NOT used in the serial dilution QLD method #
########################################################

def FisherInformation_old(mle, plate, fold):
    """
    Fisher information is given by
    d_i ^ 2 exp(-mle*d_i)/(1-exp(-mle*d_i))
    """
    fi = 0
    for dilution in plate.columns:
        d = np.power(float(fold), -float(dilution-1))
        fi_ = plate.shape[0]*(d**2) * (np.exp(-mle*d))/(1-np.exp(-mle*d))
        fi = fi+ fi_
    return(fi)
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


def poisson_ls(L, plate, fold):
    """
    For each dilution plot the expected positive fraction (1-e^(-mle/fold^d))) vs observed positive fraction p
    Returns the RMSE
    """
    Yexp = [(1- np.exp(-L/(np.power(fold, float(d-1))))) for d in plate.columns]
    Yobs = [(plate[c].sum())/plate.shape[0] for c in plate.columns]
    rmse = np.sqrt(np.mean([np.power(yexp - yobs, 2) for yexp, yobs  in zip(Yexp, Yobs)]))
    return(rmse)


def dilutionmodel_ls(L, growth, fold):
    """
    Returns the RMSE
    """
    D = 1./fold
    Yexp = [(1- np.exp(-L*(np.power(D, dilid)*(1-D)))) for dilid in range(growth.shape[1])]
    Yobs = [(growth[c].sum())/growth.shape[0] for c in growth.columns]
    rmse = np.sqrt(np.mean([np.power(yexp - yobs, 2) for yexp, yobs  in zip(Yexp, Yobs)]))
    return(rmse)


def dilutionmodel_bin_ls(L, growth, fold):
    """
    Returns the RMSE
    """
    D = 1./fold
    Yexp = [(1- np.power((1-D), L*(np.power(D, dilid)*(1-D)))) for dilid in range(growth.shape[1])]
    Yobs = [(growth[c].sum())/growth.shape[0] for c in growth.columns]
    rmse = np.sqrt(np.mean([np.power(yexp - yobs, 2) for yexp, yobs  in zip(Yexp, Yobs)]))
    return(rmse)

def poisson_ml(L, growth, fold):
    """
    Implementation of alternate model:
    Taswell 1981. 
    returns the negative log likelihood.
    Comments:
    - This optimiziation is notoriously unstable, especially the last term
    - I recommend checking this against some trivial growth tables to find the edge cases.
    """
    numreps = growth.shape[0]
    positives = growth.sum(axis=0)
    negloglikelihood = 0
    D = 1./fold
    # if L < 1e-2 :
    #     return np.nan
    for i, numpos in enumerate(positives):
        EXP = np.exp(-(L*np.power(D, i)))
        if EXP >= 1:
            lastterm = 1e-25
        else:
            lastterm = numpos*np.log((1-EXP))
        # try:
        #     term3 = numpos*np.log((1-EXP))
        # except RuntimeWarning:
        #     print(L, fold, i)
        negloglikelihood = negloglikelihood\
            - np.log(choose(numreps, numreps-numpos))\
            - (numreps - numpos)*(L/np.power(fold, i))\
            + lastterm

    return(negloglikelihood)


