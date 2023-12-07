import numpy as np
import scipy
import collections

def log_likelihood_dated_with_neutral(x, sequence_length, n_samples, mut_times_cnt, no_mut_times_cnt):
    Un, Ud, s = x
    UdL = Ud/sequence_length
    UnL = Un/sequence_length
    
    mut_times_cnt_keys = np.array(list(mut_times_cnt.keys()))
    mut_times_cnt_values = np.array(list(mut_times_cnt.values()))

    # ### This version works
    # probs = UnL + UdL * np.exp(-s * mut_times_cnt_keys)
    # LL = (np.log(probs) * mut_times_cnt_values).sum()
    # for tm, cnt in no_mut_times_cnt.items():
    #     # TODO: Is this +1 or not
    #     p = (UnL + UdL * np.exp(-s * np.arange(1, tm+1))).sum()
    #     LL += np.log(1 - p) * cnt

    delta = 0.001
    probs = UnL + UdL * np.exp(-s * mut_times_cnt_keys)
    LL = ((np.log(probs * delta) - np.log((1-probs) * delta)) * mut_times_cnt_values).sum()
    for tm, cnt in no_mut_times_cnt.items():
        # TODO: Is this +1 or not
        cnt = sequence_length * n_samples
        p = (UnL + UdL * np.exp(-s * np.arange(1, tm+1))).sum()
        LL += np.log(1 - p) * cnt


    return LL        
    
def composite_likelihood(
    tree_sequence,
    mutation_times_known = True,
    return_counts_only = False,
):
    # We assume no recombination
    assert tree_sequence.num_trees == 1
    T = tree_sequence.first()

    # The times samples come from
    unique_sample_times = set(tree_sequence.nodes_time[list(tree_sequence.samples())])

    mut_times = []
    no_mut_times = []

    # This tree may not have a single root (not all lineages have coalesced),
    # so go over all of them
    for root in T.roots:
        # The time of the root
        root_time = tree_sequence.nodes_time[root]

        # The set of leaves under the root
        root_leaves = set(list(T.leaves(root)))
            
        for site in tree_sequence.sites():
            for mutation in site.mutations:
                mutation_leaves = set(list(T.leaves(mutation.node))) & root_leaves
                
                for sample_node in root_leaves:
                    sample_time = tree_sequence.nodes_time[sample_node]
                    if sample_node in mutation_leaves:
                        mut_times.append(mutation.time - sample_time)
                    else:
                        no_mut_times.append(root_time - sample_time)
                                                                                 
    mut_times = np.array(mut_times)
    no_mut_times = np.array(no_mut_times)

    mut_times_cnt = dict(zip(*np.unique(mut_times, return_counts=True)))
    no_mut_times_cnt = dict(zip(*np.unique(no_mut_times, return_counts=True)))

    for root in T.roots:
        root_leaves = set(list(T.leaves(root)))
        for sample_node in root_leaves:
            sample_time = tree_sequence.nodes_time[sample_node]
            no_mut_times_cnt[root_time - sample_time] += (tree_sequence.sequence_length - tree_sequence.num_sites)

    if return_counts_only:
        return mut_times_cnt, no_mut_times_cnt

    #
    # Find best solution
    #
    bounds = [(1e-10,1-1e-10), (1e-10,1-1e-10), (1e-10,0.5)]
    lower_bounds = np.array([x[0] for x in bounds])
    upper_bounds = np.array([x[1] for x in bounds])

    stepsize = np.diff(np.array(bounds), axis=1).ravel() / 20

    def take_step(x):
        min_step = np.maximum(lower_bounds - x, -stepsize)
        max_step = np.minimum(upper_bounds - x, stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        
        return xnew


    res = scipy.optimize.basinhopping(
        func=lambda *args: -1*log_likelihood_dated_with_neutral(*args),
        x0=(0.5, 0.5, 0.5),    
        take_step=take_step,
        minimizer_kwargs={
            "method": "Nelder-Mead", 
            "bounds": bounds,
            "options": {"maxiter": 10000},
            "args": (tree_sequence.sequence_length, tree_sequence.num_samples, mut_times_cnt, no_mut_times_cnt),
        },
        niter=100,
        disp=False,
    )
        
    return res


def mutation_rate_integral(shifted_start, shifted_end, Un, Ud, s):
    return Un * (shifted_end - shifted_start) - (Ud / s) * (np.exp(-s * shifted_end) - np.exp(-s * shifted_start))

def better_log_likelihood_dated_with_neutral(
    x, 
    sequence_length, 
    branch_interval_counts_starts,
    branch_interval_counts_ends,   
    branch_interval_counts_values, 
    whole_interval_counts_starts,  
    whole_interval_counts_ends,    
    whole_interval_counts_values, 
):
    Un, Ud, s = x
    UdL = Ud/sequence_length
    UnL = Un/sequence_length
    
    LL = 0.0

    ps = mutation_rate_integral(whole_interval_counts_starts, whole_interval_counts_ends, UnL, UdL, s)
    LL += (whole_interval_counts_values * np.log(1 - ps)).sum()

    ps = mutation_rate_integral(branch_interval_counts_starts, branch_interval_counts_ends, UnL, UdL, s)
    LL += (branch_interval_counts_values * (np.log(ps) - np.log(1-ps))).sum()

    return LL

def better_composite_likelihood(
    tree_sequence,
    mutation_times_known = True,
    return_counts_only = False,
):
    # We assume no recombination
    assert tree_sequence.num_trees == 1
    T = tree_sequence.first()

    branch_interval_counts = collections.defaultdict(int)
    whole_interval_counts = collections.defaultdict(int)

    # Set fake interval delta in case mutation time is known
    if mutation_times_known:
        delta = 1e-3

    # This tree may not have a single root (not all lineages have coalesced),
    # so go over all of them
    for root in T.roots:
        # The time of the root
        root_time = tree_sequence.nodes_time[root]

        # Add the intervals per branch
        for node in T.nodes(root=root):
            # Skip the root itself
            if node == root:
                continue

            # For each of the tips under this node..
            for tip in T.samples(node):
                # The tip time
                tip_time = tree_sequence.nodes_time[tip]

                if mutation_times_known:
                    # Get the list of mutations with that node below them            
                    for mutation in np.where(tree_sequence.mutations_node == node)[0]:
                        # Get the time of the mutation
                        mutation_time = tree_sequence.mutations_time[mutation]

                        # Add the interval, relative to the tip time
                        branch_interval_counts[(mutation_time - tip_time, mutation_time - tip_time + delta)] += 1
                
                else:
                    n_mutations_on_branch = np.sum(tree_sequence.mutations_node == node)

                    if n_mutations_on_branch:
                        # Get the branch start and end
                        branch_start_time = tree_sequence.nodes_time[node]
                        branch_end_time = tree_sequence.nodes_time[T.parent(node)]                    

                        branch_interval_counts[(branch_start_time - tip_time, branch_end_time - tip_time)] += n_mutations_on_branch
        
        # Add the intervals for the whole span (tip to root)
        for tip in T.samples(root):
            # The tip time
            tip_time = tree_sequence.nodes_time[tip]
            
            # Add for the number of sites
            whole_interval_counts[(0, root_time - tip_time)] += tree_sequence.sequence_length

    if return_counts_only:
        return branch_interval_counts, whole_interval_counts
    
    #
    # Find best solution
    #
    bounds = [(1e-10,1-1e-10), (1e-10,1-1e-10), (1e-10,0.5)]
    lower_bounds = np.array([x[0] for x in bounds])
    upper_bounds = np.array([x[1] for x in bounds])

    stepsize = np.diff(np.array(bounds), axis=1).ravel() / 20

    def take_step(x):
        min_step = np.maximum(lower_bounds - x, -stepsize)
        max_step = np.minimum(upper_bounds - x, stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        
        return xnew


    branch_interval_counts_starts   = np.array([k[0] for k in branch_interval_counts.keys()])
    branch_interval_counts_ends     = np.array([k[1] for k in branch_interval_counts.keys()])
    branch_interval_counts_values   = np.array([v for v in branch_interval_counts.values()])
    whole_interval_counts_starts   = np.array([k[0] for k in whole_interval_counts.keys()])
    whole_interval_counts_ends     = np.array([k[1] for k in whole_interval_counts.keys()])
    whole_interval_counts_values   = np.array([v for v in whole_interval_counts.values()])

    res = scipy.optimize.basinhopping(
        func=lambda *args: -1*better_log_likelihood_dated_with_neutral(*args),
        x0=(0.5, 0.5, 0.5),    
        take_step=take_step,
        minimizer_kwargs={
            "method": "Nelder-Mead", 
            "bounds": bounds,
            "options": {"maxiter": 10000},
            "args": (
                tree_sequence.sequence_length, 
                branch_interval_counts_starts,
                branch_interval_counts_ends,   
                branch_interval_counts_values, 
                whole_interval_counts_starts,  
                whole_interval_counts_ends,    
                whole_interval_counts_values,  
            ),
        },
        niter=100,
        disp=False,
    )
        

    # res = scipy.optimize.basinhopping(
    #     func=lambda *args: -1*better_log_likelihood_dated_with_neutral(*args),
    #     x0=(0.5, 0.5, 0.5),    
    #     take_step=take_step,
    #     minimizer_kwargs={
    #         "method": "Nelder-Mead", 
    #         "bounds": bounds,
    #         "options": {"maxiter": 10000},
    #         "args": (
    #             tree_sequence.sequence_length, 
    #             branch_interval_counts, 
    #             whole_interval_counts
    #         ),
    #     },
    #     niter=100,
    #     disp=False,
    # )
        
    return res
                    


