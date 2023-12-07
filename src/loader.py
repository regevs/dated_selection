import numpy as np
import pyslim
import msprime
import tskit

def load_tree_sequence_from_slim(
    filename,
    n_samples_per_time,
    sample_times = [0],
    neutral_mutation_rate_per_genome = 0.0,
    random_seed = 42,
):
    """
    - Loads a SLiM output file
    - Removed unneeded information left over from the haploid mode of SLiM
    - Take a random subset of individuals at specified times
    - Add neutral mutations
    """
    rng = np.random.default_rng(seed=random_seed)

    # Load the tree sequence
    ts = tskit.load(filename)

    # Throw away all the odd-numbered nodes - these are leftovers from the haploid simulation
    sts = ts.simplify(np.arange(0, len(ts.nodes_time), 2))
                        
    # Take a subset of samples    
    sts = sts.simplify([
        sts.individual(i).nodes[0] \
        for n_samples, sample_time in zip(n_samples_per_time, sample_times) \
        for i in rng.choice(pyslim.individuals_alive_at(sts, sample_time), n_samples, replace=False)
    ])

    mts = msprime.sim_mutations(
        sts,
        rate=neutral_mutation_rate_per_genome/sts.sequence_length,    # per bp
        model=msprime.SLiMMutationModel(type=3, next_id=pyslim.next_slim_mutation_id(sts)), # type=3 is just any mutation type unused in the simulation
        keep=True, 
        discrete_genome=True,
        random_seed=random_seed,
    )

    return mts

    