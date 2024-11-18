"""
Author: Amogh Jalihal
Date: 2024-11-18
"""
from scipy.optimize import minimize
def PoissonJoint(x, plate, fold):
    """
    Takes a number of "cells" x and a "plate" (pandas DataFrame)  with rows as replicates
    and columns as serial dilutions.
    Returns the likelihood of the number of cells explaining the observed data.
    """
    rowterm = 1
    denominator = 0

    ## Iterate over all possible states of growth and no growth for the number of dilutions 
    ## which correspond to the number of columns of the plate.
    
    for seq in product(["growth","nogrowth"], repeat=plate.shape[1]):
        _denom = 1
        for i, term in enumerate(seq):
            if term == "growth":
                _denomn = _denom*(1 - np.exp(-x/np.power(fold, i)))
            else:
                _denom = _denom*( np.exp(-x/np.power(fold, i)))
        denominator += _denom

    ## Construct the joint probability
    for _, row in plate.iterrows():
        numerator = 1
        for i, well in enumerate(row.values):
            if well == 0:
                numerator = numerator*(np.exp(-x/np.power(fold,i)))
            else:
                numerator = numerator*(1-np.exp(-x/np.power(fold, i)))
        rowterm = rowterm*(numerator/denominator) 
    return(-np.log(rowterm))


def quantifyInputFromSerialDilution(serialDilutionTable, foldDilution=10, initialGuess=1.0):
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
    sol = minimize(PoissonJoint, initialGuess, method="Nelder-Mead",args=(plate, foldDilution))
    return(sol)
    
    
