import numpy as np
import pandas as pd
import pyslim, tskit
import itertools
import scipy.special
import os, subprocess
import msprime
import sys 

filename = sys.argv[1]


def extract_tree_sequence(tree, sample_times, sample_sizes, Ne=10000, add_neutral=False):
    num_times = len(sample_times)
    if num_times != len(sample_sizes): 
        print("Please give the same length list for sample times and sizes.")   
    
    
    # Subsample tree
    sample = [np.random.choice(
        pyslim.individuals_alive_at(tree,sample_times[i]), 
        size = int(sample_sizes[i]), 
        replace=False) for i in list(range(num_times))]
    
    sample = np.concatenate(sample)
    sample_nodes = [tree.individual(i).nodes[0] for i in sample]
    # Identifying the nodes of the chosen downsample individuals. 
    ts_ds = tree.simplify(samples=sample_nodes) 
    # This writes a new tree sequence with just downsampled individuals.
    
    # Double check everyone has coalesced within the time frame.
    # Recapitate if needed.
    tsf = ts_ds.first()
#     if len(tsf.roots) != 1 : 
#         print("Not everyone has coalesced within your time frame! Recapitating...")
#         ts_ds = pyslim.recapitate(ts_ds,
#                 recombination_rate=0,
#                 ancestral_Ne=Ne, random_seed=5)
#         print("Successfully recapitated")
    
    # Add neutral mutations if needed
    if add_neutral:
        next_id = pyslim.next_slim_mutation_id(ts_ds)
        ts = msprime.sim_mutations(
               ts_haploid,
               rate=1e-3,    # per bp
               model=msprime.SLiMMutationModel(type=3, next_id=next_id), # type=3 is just any mutation type unused in the simulation
               keep=True)
        print(f"We now have {ts_ds.num_mutations} deleterious mutations and {tsn.num_mutations} total mutations.")
    else:
        ts = ts_ds
    
    # Get coalescent times.
    node_times = ts.tables.nodes.time
    coal_times = node_times[sum(sample_sizes):] # the first ones are tips, so a bunch of 0s
    
    # Get mutation information.
    mutation_times = []
    mutation_edges = []
    for m in ts.mutations():
        mutation_times.append(m.metadata["mutation_list"][0]["slim_time"])
        mutation_edges.append(m.edge)
    
    return ts, coal_times, mutation_times, mutation_edges

def get_lineages(ts, max_generations, sample_times, sample_sizes):
    n_lineages = np.zeros(max_generations, dtype=int)
    for sample_time, sample_size in zip(sample_times, sample_sizes):
        n_lineages[sample_time:] += sample_size
    
    parents, n_children_per_parent = np.unique(ts.edges_parent, return_counts=True)
    n_children_array = np.zeros_like(ts.nodes_time, dtype=int)
    n_children_array[parents] = n_children_per_parent
    
    for node_id, n_children, coal_time in zip(np.arange(len(ts.nodes_time)), n_children_array, ts.nodes_time.astype(int)):
        if node_id not in parents:
            continue
        n_lineages[coal_time:] -= (n_children-1)

    return n_lineages
    
def get_mutation_counts(mutation_times, max_generations):
    mutation_count_per_generation = np.histogram(max_generations - np.array(mutation_times), np.arange(max_generations+1))[0]
    return mutation_count_per_generation

def extract_stats_to_file(
    trees,
    sample_times,
    sample_sizes,
    n_tree_samples,
    max_generations,
    n_lineages_csv_filename,
    n_mutations_csv_filename,
):
    all_n_lineages = []
    all_n_mutations = []
    for n_sample in range(n_tree_samples):
        ts, coal_times, mutation_times, mutation_edges = \
            extract_tree_sequence(
                trees, 
                sample_times=sample_times,
                sample_sizes=sample_sizes,
            )

        # Count lineages
        n_lineages = get_lineages(
            ts, max_generations, sample_times, sample_sizes
        )
        all_n_lineages.append(n_lineages)

        # Count mutations
        n_mutations = get_mutation_counts(
            mutation_times, max_generations
        )
        all_n_mutations.append(n_mutations)

    pd.DataFrame(np.array(all_n_lineages))\
        .to_csv(n_lineages_csv_filename, header=False, index=False)

    pd.DataFrame(np.array(all_n_mutations))\
        .to_csv(n_mutations_csv_filename, header=False, index=False)