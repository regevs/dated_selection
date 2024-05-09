# Fast simulator for background selection CTMC
# May 3 2024, Bianca De Sanctis

import os
import sys # later will want to read args from the command line 
import math
import numpy as np
from random import random 
from numba import jit
import time # for testing 

@jit(nopython=True, fastmath=True)
def factorial(n):
    if n > 25: # fails over n=25  
        print("I can't factorial n>25") 
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

@jit(nopython=True, fastmath=True)
def random_choice_numba(arr, prob): # thanks modified https://github.com/search?q=numba+random.choice&type=code 
    cs = np.cumsum(prob)
    cs[len(cs)-1] = 1  
    return arr[np.searchsorted(cs, random(), side="right")]

@jit(nopython=True, fastmath=True)
def random_choices_numba(arr, prob, n):
    results = np.empty(n, dtype=arr.dtype) 
    for i in range(n):
        results[i] = random_choice_numba(arr, prob)
    return results

@jit(nopython=True, fastmath=True)
def binomial(n, k): # thanks https://gist.github.com/rougier/ebe734dcc6f4ff450abf 
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b /= t+1
        n -= 1
    return int(b)

@jit(nopython=True, fastmath=True)
def setdiff(set1,set2): # things in set1 that are not in set2 
    return np.array([element for element in set1 if element not in set2])

@jit(nopython=True, fastmath=True)
def eventnametonumber(str): # this is silly; they could just be coded by numbers the whole way through; to be fixed
    # (this was a quick fix because numba doesn't like arrays of strings)
    if str=="popsizechange":
        return 0.0
    if str=="linadd":
        return 1.0
    if str=="coal":
        return 2.0
    if str=="mut":
        return 3.0
    else:
        return -1.0

@jit(nopython=True, fastmath=True)
def generate(Ud=0.02, s=0.01, n=100, N=np.array([10000]), numclasses=0 ,
             nodeages=np.empty(0, dtype=float) , switchtimes=np.empty(0, dtype=float), n_reps=100):
    
    # Initialize
    coalescent_times = np.zeros((n_reps, n - 1), dtype=float) # there are n-1 internal nodes on a phylogeny of n tips 

    # Switchtimes is the times that the pop size changed. I think this in particular not an ideal way to do this
    # This is a lightweight version that only spits out coalescent times and doesn't even bother saving other things 

    # Some sanity checking (more could be added; there's much more in previous code versions)
    if len(nodeages) == 0:
        nodeages = np.array([0] * n,dtype="float")
    else:
        if len(nodeages) != n:
            n = len(nodeages)

    # Choose how many classes you need to cover almost all of the equilibrium distribution
    lotsofks = list(range(25))
    hs = np.array([(Ud / s) ** k * math.exp(-Ud / s) / factorial(k) for k in lotsofks])
    ks = np.where(np.cumsum(hs) < .999)[0]   # 0.999 is an arbitrary decision
    numclasses = len(ks)
    # Define the equilibrium distribution
    hs = hs[0:numclasses]
    h = np.array([h/sum(hs) for h in hs])

    for n_rep in range(n_reps):
        # Sample tip classes according to equilibrium distribution to define your starting state
        tip_classes = random_choices_numba(ks,h,n)
        modern_tip_classes = np.array([tip_classes[i] for i in range(len(nodeages)) if nodeages[i] == 0])
        alpha = np.bincount(modern_tip_classes,minlength=numclasses)

        # Initialize some bits
        # Non-contemporaneous tips and changing pop size is done by keeping track of how many changes have yet to come and their times 
        #   in the variables remainingswitches and tipsnotin
        numlineages = len(modern_tip_classes)
        e = np.eye(numclasses).astype('int64')
        tipsnotin = np.array([i for i, val in enumerate(nodeages) if val > 0.0])   # np.where(nodeages > 0.0)[0]
        t = 0.0
        remainingswitches = switchtimes
        
        currN = 0 # to be incremented if/when we switch population sizes
        # alphalist = np.copy(alpha) # no longer tracking
        counter = 0 # for only coalescent events 
        time_to_next_event = np.inf # for pop size changes or dated tips
        # Loop
        while (numlineages > 1) or (len(tipsnotin) > 0) :
            # Keep going if you still have lineages to coalesce or tips to add
            # possible event types: 0 (popsizechange), 1 (linadd), 2 (coal), 3 (mut) 

            # Calculate rates for this round
            raw_coal_rates = np.array([binomial(i,2) for i in alpha]) / np.multiply(hs,N[currN])
            raw_mut_rates = s*ks*alpha.astype(np.float64)
            lambda_alpha = np.sum(raw_coal_rates) + np.sum(raw_mut_rates) # eq (5.23) from wakeley
            coal_probs = raw_coal_rates / lambda_alpha
            mut_probs = raw_mut_rates / lambda_alpha
            coalprob = np.sum(coal_probs) / (np.sum(coal_probs) + np.sum(mut_probs)) if np.sum(coal_probs) + np.sum(mut_probs) != 0.0 else 0.0

        #     # Update bookkeeping for dated tips and pop size switches
        #     if (len(tipsnotin)>0) or (len(remainingswitches)>0) :
        #         time_to_next_tip_drop_in = np.min(nodeages[nodeages > t] - t) if len(nodeages[nodeages > t]) > 0 else np.inf
        #         time_to_next_pop_size_switch = np.min(remainingswitches[remainingswitches >= t] - t) if remainingswitches[remainingswitches >= t].size > 0 else np.inf
        #         time_to_next_tip_drop_in= np.min(remainingswitches[remainingswitches >= t] - t) if remainingswitches[remainingswitches >= t].size > 0 else np.inf
        #         time_to_next_event =  min(time_to_next_tip_drop_in, time_to_next_pop_size_switch)
        #         next_event = "linadd" if time_to_next_event == time_to_next_tip_drop_in else "popsizechange"

        #     # How long until the next event and what is it?
        #     somethinghappened = np.random.exponential(1 / lambda_alpha) if lambda_alpha > 0 else time_to_next_event
        #     whathappened = next_event if time_to_next_event <= somethinghappened else random_choice_numba(["coal", "mut"], np.array([coalprob, 1 - coalprob]))

        #     # Update CTMC and tracking variables
        #     if whathappened == "popsizechange":
        #         currN += 1  # Increment to next population size
        #         remainingswitches = remainingswitches[1:] 
        #         t += time_to_next_pop_size_switch
        #         time_to_next_event = np.inf

        #     if whathappened == "linadd":
        #         minagetoadd = float(np.min(nodeages[tipsnotin]))
        #         addtips = np.where(nodeages == minagetoadd)[0]
        #         addlineages = tip_classes[addtips]
        #         for l in addlineages:
        #             alpha[l + 1] += 1
        #         tipsnotin = setdiff(tipsnotin, addtips) # previously tipsnotin[np.setdiff1d(tipsnotin, addtips)]
        #         t += time_to_next_tip_drop_in
        #         time_to_next_event = np.inf

        #     if whathappened == "coal":
        #         whichcoal = random_choice_numba(np.arange(0, numclasses), coal_probs / coalprob)
        #         alpha = alpha - e[whichcoal]
        #         t += somethinghappened
        #         coalescent_times[n_rep, counter] = t
        #         counter +=1
            
        #     if whathappened == "mut":
        #         whichmut = random_choice_numba(np.arange(0, numclasses), mut_probs / (1 - coalprob))
        #         if whichmut == 0:
        #             print("serious error")
        #         alpha = alpha - e[whichmut] + e[whichmut-1]
        #         t += somethinghappened
            
        #     numlineages = np.sum(alpha)
        #     # alphalist = np.append(alphalist,alpha.copy())

    # alphalist_matrix = alphalist.reshape((alphalist.size // numclasses, numclasses))
    return coalescent_times
