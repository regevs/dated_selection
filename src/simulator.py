from dataclasses import dataclass
import numpy as np
import tqdm

def normalize(a):
    return a / a.sum()

@dataclass
class Simulator(object):
    Ne: int
    s: float
    Un: float
    Ud: float
    n_generations: int
    sampling_scheme: dict = None
    random_seed: ... = None

    def __post_init__(self):
        self.ancestors = []
        self.rng = np.random.default_rng(seed=self.random_seed)
        self.Nes = np.ones(self.n_generations, dtype=int) * self.Ne

        if self.sampling_scheme is None:
            self.sampling_scheme = {}
        self.samples = {}

    def prune_ancestors(self, n_generation):
        """
        Prune the ancestors of n_generation-1 based on n_generation.
        Returns True if any ancestors were removed.
        """
        # If first generation, do nothing
        if n_generation == 0:
            return False

        active_ancestors = self.ancestors[n_generation].values()
        prev_active_ancestors = self.ancestors[n_generation-1].keys()

        if active_ancestors == prev_active_ancestors:
            # Nothing to do
            return False

        # Prune
        n_samples_to_take = self.sampling_scheme.get(n_generation - 1, 0)
        if n_samples_to_take:
            active_ancestors = set(active_ancestors) | set(range(n_samples_to_take))

        self.ancestors[n_generation - 1] = \
            {i:self.ancestors[n_generation - 1][i] for i in active_ancestors}

        return True
        

    def run(self):
        # Initialize first generation
        self.n_mutations_neutral = np.zeros(self.Nes[0], dtype=int)        
        self.n_mutations_selected = np.zeros(self.Nes[0], dtype=int)
        self.all_n_mutations_selected = [self.n_mutations_selected.copy()]
        self.all_new_mutations_selected = [self.n_mutations_selected.copy()]
        self.fitness_scores = np.ones(self.Nes[0], dtype=float)
        self.ancestors = [{} for n_generation in range(self.n_generations)]
        self.ancestors[0] = {i:i for i in range(self.Nes[0])}

        # Do generation by generation
        for n_generation in range(1, self.n_generations):
            # Collect any statistics here

            # Extract samples if needed
            if n_generation in self.sampling_scheme.keys():
                n_samples_to_take = self.sampling_scheme[n_generation]
                self.samples[n_generation] = [self.n_mutations_neutral[:n_samples_to_take], self.n_mutations_selected[:n_samples_to_take]]

            # Draw parents according to selection
            parents = self.rng.choice(
                a = self.Nes[n_generation - 1],
                size = self.Nes[n_generation],
                p = normalize(self.fitness_scores)
            )

            # Record the ancestors
            self.ancestors[n_generation] = \
                {i:parents[i] for i in range(self.Nes[n_generation])}  

            # Prune ancestors
            continue_pruning = True
            n_generation_to_prune = n_generation
            while continue_pruning:
                continue_pruning = self.prune_ancestors(n_generation_to_prune)
                n_generation_to_prune -= 1

            # Copy from ancestors
            self.n_mutations_neutral = self.n_mutations_neutral[parents]
            self.n_mutations_selected = self.n_mutations_selected[parents]

            # Add new mutations
            self.n_mutations_neutral += self.rng.binomial(
                n = 1, 
                p = self.Un,
                size = self.Nes[n_generation]
            )
            new_mutations = self.rng.binomial(
                n = 1, 
                p = self.Ud,
                size = self.Nes[n_generation]
            )
            self.n_mutations_selected += new_mutations

            self.all_new_mutations_selected.append(new_mutations.copy())
            self.all_n_mutations_selected.append(self.n_mutations_selected.copy())

            # Calculate new fitness
            self.fitness_scores = (1 - self.s) ** self.n_mutations_selected
            
    def tmrca(self, time_i, index_i, time_j, index_j):
        # Make sure time_i <= time_j
        if time_i > time_j:
            time_i, index_i, time_j, index_j = time_j, index_j, time_i, index_i

        # Track j up until both are at time_i
        while (time_j > time_i):
            index_j = self.ancestors[time_j][index_j]
            time_j -= 1

        # Track both until they either meet on hit gen 0, in which case return -1
        tmrca = 0
        while time_i:
            if index_i == index_j:
                return tmrca
            index_i = self.ancestors[time_i][index_i]
            index_j = self.ancestors[time_i][index_j]
            time_i -= 1
            tmrca += 1

        if time_i == 0:
            return -1









