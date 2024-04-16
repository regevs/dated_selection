import numpy as np
import pyslim
import msprime
import tskit
import tszip

def load_tree_sequence_from_slim(
    filename,
    n_samples_per_time,
    sample_times = [0],
    neutral_mutation_rate_per_genome = 0.0,
    random_seed = 42,
    recapitate = False,
    recapitate_Ne = None,
):
    """
    - Loads a SLiM output file
    - Removed unneeded information left over from the haploid mode of SLiM
    - Take a random subset of individuals at specified times
    - Add neutral mutations
    """
    rng = np.random.default_rng(seed=random_seed)

    # Load the tree sequence
    if filename.endswith(".tsz"):
        sts = tszip.decompress(filename)
    else:
        sts = tskit.load(filename)

    # Recapitate if needed
    if recapitate:
        sts = pyslim.recapitate(
            sts,
            recombination_rate=0,
            ancestral_Ne=recapitate_Ne, 
            random_seed=random_seed,
        )

    # Throw away all the odd-numbered nodes - these are leftovers from the haploid simulation
    sts = sts.simplify(np.arange(0, len(sts.nodes_time), 2))
                        
    # Take a subset of samples    
    sts = sts.simplify([
        sts.individual(i).nodes[0] \
        for n_samples, sample_time in zip(n_samples_per_time, sample_times) \
        for i in rng.choice(pyslim.individuals_alive_at(sts, sample_time), n_samples, replace=False)
    ])

    # Add neutrals if needed
    mts = msprime.sim_mutations(
        sts,
        rate=neutral_mutation_rate_per_genome/sts.sequence_length,    # per bp
        model=msprime.SLiMMutationModel(type=3, next_id=pyslim.next_slim_mutation_id(sts)), # type=3 is just any mutation type unused in the simulation
        keep=True, 
        discrete_genome=True,
        random_seed=random_seed,
    )

    return mts

    