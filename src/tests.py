import numpy as np
from scipy.stats import beta, norm
import scipy.special

##########################################################
#  Generate a varient on Pybus and Harvey's gamma stat.  #
#  normalized for the neutral coalescent.                #
#    t = set of coalescent times, ordered root-to-tip.   #
#        i.e. oldest to youngest	                     #
#    k (NULL calculates the standard stat; k!=NULL       #
#       compares the youngest k events to the rest)      #
##########################################################
def gam(t, k=None):
    n = len(t) + 1  # Number of tips in tree
    t = t * scipy.special.binom(np.arange(2, n+1), 2) # Normalize, so iid under the null
    x = np.sum(t)

    if k is None:  # Calculate the standard test stat.
        w = np.arange(n - 2, -1, -1)  # Weight the oldest events more
        y = 2 * np.sum(t * w) / (n - 2)
        return ((y - x) / x) * np.sqrt(3 * (n - 2))  # Could reverse the sign, s.t. that +ve = pop growth...
    
    # If k is integer, compare the youngest k tips to the remainder
    y = np.sum(t[-k:])
    return norm.ppf(beta.cdf(y / x, a=k, b=n - 1 - k))

