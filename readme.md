# Contents
1. `XanthoMoClo-plasmid-copy-number.py` contains analysis code to quantify plasmid copy numbers presented in Soltysiak, Ory et. al.
2. `src/PoissonJoint.py` contains the core logic of estimating the number of targets from a (noisy) serial dilution.
# Limiting Serial Dilutions
The _objective_ of this tool is to estimate with high confidence the number of targets (viable cells/genomes/detectable targets) from a limiting serial dilution of this starting material. A serial dilution is said to be _limiting_ if we fully dilute out the starting sample, that is, we expect to see _empty_ wells at some point in the dilution.

Serial dilutions are modeled as a Poisson process. We use the fact that for an expected number of events `x`, the probability of failure to observe an event is given by

Poisson(0 events | x) = e^(-x)

In a serial diution starting from x cells, the Probability of seeing observing any target, i.e. >0 events can thus be written as 

Poisson(>0 events | x) = 1- e^(-x)


If we are performing an `f` fold serial dilution, the process can be represented as 
product_i=1^(i=k) (Poisson(growth observed well | x/f^k ) for `k` serial dilutions.

## Usage

See `./test.py`

