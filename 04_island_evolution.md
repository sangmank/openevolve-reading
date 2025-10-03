# Island-Based Evolution: Parallel Worlds of Code

## The Motivation: Beyond Single-Population Limits

We've seen how MAP-Elites maintains diversity within a population by organizing programs into a multi-dimensional grid. But there's a deeper problem that even MAP-Elites can't fully solve: **exploration vs. exploitation**.

Every evolutionary algorithm faces this dilemma:
- **Exploit**: Focus on improving the best solutions found so far
- **Explore**: Search for completely new approaches

Too much exploitation → premature convergence
Too much exploration → wasting time on bad solutions

MAP-Elites helps by maintaining diverse niches, but all programs still share the same gene pool. What if we could run **multiple parallel evolutions** that occasionally share insights?

Enter **island-based evolution**.

## The Biological Inspiration

In nature, island biogeography explains why isolated habitats develop unique species:

- **Galápagos finches**: Each island evolved different beak shapes for local food sources
- **Madagascar lemurs**: Isolation created species found nowhere else
- **Hawaiian honeycreepers**: Spectacular diversity from a single ancestor

Key insight: **Isolation drives diversity**. But occasional migration between islands allows beneficial traits to spread while preserving overall diversity.

OpenEvolve applies this to code evolution:

```
Island 1          Island 2          Island 3
┌─────────┐      ┌─────────┐      ┌─────────┐
│ MAP-    │      │ MAP-    │      │ MAP-    │
│ Elites  │      │ Elites  │      │ Elites  │
│ Grid    │ ←→   │ Grid    │ ←→   │ Grid    │
│         │      │         │      │         │
└─────────┘      └─────────┘      └─────────┘
    ↑                                    ↓
    └────────────────────────────────────┘
         (Ring topology for migration)
```

Each island:
- Runs independent evolution (selection, mutation, evaluation)
- Has its own MAP-Elites grid (separate populations)
- Periodically sends "migrants" to adjacent islands

## The Implementation

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/database.py`

### Island Data Structure

```python
@dataclass
class IslandPopulation:
    """Represents one island in the archipelago."""
    id: int                    # Island identifier (0, 1, 2, ...)
    programs: Set[str]         # Program IDs on this island
    feature_maps: Dict         # Island-specific MAP-Elites grid
    generation: int            # How many generations this island has evolved
    best_program_id: str       # Island champion
    migration_history: List    # Track what/when we migrated

class ProgramDatabase:
    def __init__(self, config):
        # Create archipelago
        self.num_islands = config.num_islands
        self.islands = [
            IslandPopulation(
                id=i,
                programs=set(),
                feature_maps={},
                generation=0,
                best_program_id=None,
                migration_history=[]
            )
            for i in range(self.num_islands)
        ]

        # Migration settings
        self.migration_rate = config.migration_rate        # Fraction to migrate (e.g., 0.05 = 5%)
        self.migration_interval = config.migration_interval  # Generations between migrations
        self.migration_topology = config.migration_topology  # "ring", "full", "star"

        # Shared program storage (all islands' programs)
        self.programs = {}  # {program_id: Program}
```

**Key design**: Programs are stored globally (`self.programs`), but each island maintains its own **index** (`island.programs` is a set of IDs). This allows:
- Memory efficiency (don't duplicate program code)
- Easy migration (copy ID to another island's set)
- Centralized artifact storage

### Island-Aware Sampling

When selecting a parent for mutation, we sample from the **current island only**:

```python
def sample_parent(self, island_id: int) -> Program:
    """Sample parent from specific island."""
    island = self.islands[island_id]

    if len(island.programs) == 0:
        raise ValueError(f"Island {island_id} is empty")

    # Sample uniformly from this island's programs
    program_id = random.choice(list(island.programs))
    return self.programs[program_id]
```

Why island-specific? To maintain **genetic isolation**. Island 0's evolution is independent from Island 1's. They only interact via migration.

### Island-Aware Inspirations

When building prompts, we show the LLM top programs from the **same island**:

```python
def sample_top_programs(self, island_id: int, n: int = 5) -> List[Program]:
    """Sample top N programs from specific island."""
    island = self.islands[island_id]

    # Get programs from this island
    island_programs = [
        self.programs[pid] for pid in island.programs
    ]

    # Sort by fitness
    sorted_programs = sorted(
        island_programs,
        key=lambda p: get_fitness_score(p.metrics, exclude=self.feature_dimensions),
        reverse=True
    )

    return sorted_programs[:n]
```

This creates island-specific "cultures" of programming styles. Island 0 might evolve toward complex simulated annealing, while Island 1 explores genetic algorithms, and Island 2 tries gradient-free optimization.

## The Migration Mechanism

After every `migration_interval` generations, islands exchange top programs:

```python
def should_migrate(self) -> bool:
    """Check if it's time for migration based on generation counts."""
    # Migrate when ANY island has evolved enough generations
    min_generation = min(island.generation for island in self.islands)
    return min_generation > 0 and min_generation % self.migration_interval == 0

def migrate_programs(self):
    """Perform migration between islands."""
    if not self.should_migrate():
        return

    for i, island in enumerate(self.islands):
        # Get all programs from this island
        island_programs = [self.programs[pid] for pid in island.programs]

        if len(island_programs) == 0:
            continue  # Nothing to migrate

        # Sort by fitness
        island_programs.sort(
            key=lambda p: get_fitness_score(p.metrics, exclude=self.feature_dimensions),
            reverse=True
        )

        # Select top X% for migration
        num_to_migrate = max(1, int(len(island_programs) * self.migration_rate))
        migrants = island_programs[:num_to_migrate]

        # Determine target islands based on topology
        if self.migration_topology == "ring":
            # Send to adjacent islands in ring
            target_islands = [
                (i + 1) % self.num_islands,  # Clockwise neighbor
                (i - 1) % self.num_islands   # Counter-clockwise neighbor
            ]
        elif self.migration_topology == "full":
            # Send to all other islands
            target_islands = [j for j in range(self.num_islands) if j != i]
        elif self.migration_topology == "star":
            # Island 0 is hub, others connect to it
            if i == 0:
                target_islands = list(range(1, self.num_islands))
            else:
                target_islands = [0]
        else:
            raise ValueError(f"Unknown topology: {self.migration_topology}")

        # Perform migration
        for migrant in migrants:
            # Check if already migrated (prevent re-migration)
            if migrant.metadata.get("migrant", False):
                continue  # Skip to prevent exponential duplication

            for target_island_id in target_islands:
                # Create migrant copy
                migrant_copy = Program(
                    id=str(uuid.uuid4()),  # New ID for the copy
                    code=migrant.code,
                    parent_id=migrant.id,
                    metrics=migrant.metrics.copy(),
                    metadata={
                        **migrant.metadata,
                        "migrant": True,          # Mark as migrant
                        "source_island": i,       # Track origin
                        "target_island": target_island_id
                    }
                )

                # Add to target island
                self._add_to_island(target_island_id, migrant_copy)

                # Log migration
                island.migration_history.append({
                    "generation": island.generation,
                    "program_id": migrant.id,
                    "target_island": target_island_id,
                    "fitness": get_fitness_score(migrant.metrics, exclude=self.feature_dimensions)
                })
```

### The Anti-Duplication Guard

Notice this critical check:

```python
if migrant.metadata.get("migrant", False):
    continue  # Skip to prevent exponential duplication
```

**Why is this needed?**

Without it, a program could migrate from Island 0 → Island 1 → Island 2 → Island 0 → ... in an infinite loop, exponentially multiplying copies.

With the guard:
1. Program P on Island 0 migrates to Island 1 (marked `migrant=True`)
2. Island 1 tries to migrate P to Island 2... but P is already a migrant, skip!

Result: Each program migrates **at most once**, preventing duplication.

## Migration Topologies

OpenEvolve supports different migration patterns:

### 1. Ring Topology (Default)

```
Island 0 ←→ Island 1 ←→ Island 2 ←→ Island 3
    ↑                                    ↓
    └────────────────────────────────────┘
```

Each island sends to its two neighbors. Forms a **ring**.

**Pros**:
- Gradual information spread (takes time for Island 0's innovation to reach Island 3)
- Strong isolation (maintains distinct island "cultures")

**Cons**:
- Slow global convergence

**Use case**: When you want maximum diversity and long-term exploration.

### 2. Full Topology

```
    ┌─────────┐
    │Island 0 │←──────┐
    └────┬────┘       │
         ↓            │
    ┌─────────┐      │
    │Island 1 │←─────┤
    └────┬────┘      │
         ↓           │
    ┌─────────┐     │
    │Island 2 │─────┘
    └─────────┘
  (all-to-all connections)
```

Every island sends migrants to every other island.

**Pros**:
- Fast information spread
- Quick global convergence

**Cons**:
- Less diversity (islands become similar)

**Use case**: When you've found good solutions and want rapid refinement.

### 3. Star Topology

```
       Island 0 (hub)
        ↙ ↓ ↓ ↘
    Island 1 2 3 4 (spokes)
```

Island 0 is the central hub. Others connect only to it.

**Pros**:
- Controlled information flow
- Central island accumulates best solutions

**Cons**:
- Central island can become bottleneck
- Spoke islands evolve semi-independently

**Use case**: When you want one "elite" population with diverse sub-populations feeding it.

## Configuration

```yaml
database:
  # Island settings
  num_islands: 4                    # Number of islands
  migration_rate: 0.05              # Migrate top 5% of each island
  migration_interval: 50            # Migrate every 50 generations
  migration_topology: "ring"        # Options: "ring", "full", "star"

  # Per-island population size
  population_size: 1000              # Total across all islands
  # (Each island gets population_size / num_islands programs)
```

## Generations vs. Iterations

**Important distinction**:

- **Iteration**: One call to the LLM → one program generated
- **Generation**: One full round of evolution on an island (may include multiple iterations)

Islands track **generations** for migration timing:

```python
def increment_island_generation(self, island_id: int):
    """Called after processing all programs in an island."""
    self.islands[island_id].generation += 1
```

This ensures migration happens on a consistent schedule regardless of parallel processing.

## Example: Evolution Across Islands

Let's trace an experiment with 3 islands over 150 generations:

**Generation 0-49**: Isolated Evolution

```
Island 0: Exploring random search variants
- Best score: 0.40
- Approach: Random sampling with restarts

Island 1: Exploring hill climbing
- Best score: 0.35
- Approach: Greedy local search

Island 2: Exploring simulated annealing
- Best score: 0.55 ← Best overall!
- Approach: Temperature-based acceptance
```

**Generation 50**: First Migration (Ring Topology)

```
Island 2 → Island 0: Migrates simulated annealing (score 0.55)
Island 0 → Island 1: Migrates random search (score 0.40)
Island 1 → Island 2: Migrates hill climbing (score 0.35)
```

**Generation 51-99**: Hybrid Evolution

```
Island 0: Now has both random search AND simulated annealing
- Crossover creates: Random search with temperature-based acceptance
- Best score: 0.62 ← Hybrid is better than either parent!

Island 1: Now has hill climbing AND random search
- Evolves: Multi-start hill climbing
- Best score: 0.50

Island 2: Now has simulated annealing AND hill climbing
- Evolves: Annealing with greedy acceptance criterion
- Best score: 0.58
```

**Generation 100**: Second Migration

```
Island 0 → Island 1: Migrates hybrid (score 0.62) ← Best solution spreads!
Island 1 → Island 2: Migrates multi-start (score 0.50)
Island 2 → Island 0: Migrates modified annealing (score 0.58)
```

**Generation 101-150**: Convergence

```
Island 0: Refining hybrid approach
- Best score: 0.70

Island 1: Adapting hybrid to local style
- Best score: 0.68

Island 2: Combining hybrid with annealing insights
- Best score: 0.72 ← New global best!
```

Notice how:
1. Islands explore different approaches initially (diversity)
2. Migration introduces new ideas (cross-pollination)
3. Islands adapt migrants to their local context (innovation)
4. Best solutions eventually spread (convergence)

Without islands, the system might have converged to simulated annealing (score 0.55) early and gotten stuck. Islands allowed exploration of random search and hill climbing, leading to the hybrid (score 0.72).

## Implementation Detail: Lazy Migration

OpenEvolve uses "lazy" migration based on generation count, not iteration count:

```python
# NOT this (iteration-based):
if iteration % migration_interval == 0:
    migrate()

# But this (generation-based):
if min_generation % migration_interval == 0:
    migrate()
```

**Why?** Because iterations can be processed in parallel. With 10 workers, iteration 100 might complete before iteration 95. But generations are island-specific and sequential within an island.

This ensures migration happens at consistent evolutionary stages, not arbitrary iteration numbers.

## Observing Island Dynamics

OpenEvolve logs island statistics:

```python
def get_island_stats(self) -> Dict:
    """Return statistics for each island."""
    stats = {}
    for island in self.islands:
        programs = [self.programs[pid] for pid in island.programs]
        fitnesses = [
            get_fitness_score(p.metrics, exclude=self.feature_dimensions)
            for p in programs
        ]

        stats[island.id] = {
            "population_size": len(programs),
            "generation": island.generation,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "diversity": self._calculate_island_diversity(island),
            "num_migrations_sent": len(island.migration_history)
        }
    return stats
```

You can track:
- Which island is performing best?
- Are islands diverging (good) or converging (maybe bad)?
- How many migrations have occurred?

## When to Use Islands

**Use islands when**:
- Search space is large and multimodal (many local optima)
- You want to maintain diverse approaches
- You have computational resources for parallelism
- Long-term exploration is more important than fast convergence

**Don't use islands when**:
- Search space is simple (single basin of attraction)
- You need fast convergence to any good solution
- Computational budget is tight
- Problem structure is well-understood

## The Trade-off: Diversity vs. Efficiency

Islands fundamentally trade computational efficiency for solution diversity.

**Single population**:
- All 1000 programs collaborate on one search
- Faster convergence
- Risk of premature convergence

**4 islands with 250 programs each**:
- Four independent searches of 250 programs
- Slower convergence per island
- Much higher chance of finding global optimum

It's the explore/exploit trade-off made explicit: Islands explore, migration exploits.

## Advanced: Adaptive Migration

OpenEvolve's default migration is fixed-schedule (every N generations). Advanced users can implement **adaptive migration**:

```python
def should_migrate_adaptive(self):
    """Migrate when islands have diverged sufficiently."""
    # Calculate diversity between islands
    inter_island_diversity = self._calculate_inter_island_diversity()

    # Migrate if islands are too similar (need fresh genes)
    if inter_island_diversity < self.diversity_threshold:
        return True

    # Or migrate if one island is much better (share the wealth)
    fitness_gap = max_island_fitness - min_island_fitness
    if fitness_gap > self.fitness_gap_threshold:
        return True

    return False
```

This makes migration **data-driven** instead of schedule-driven.

## Next Chapter

We've seen how islands create parallel evolutionary lineages with periodic gene flow. But what happens inside each iteration? How are programs actually evaluated, and how does feedback flow back to guide evolution?

Next, we'll explore the **evaluation pipeline**: cascade testing, timeout handling, artifact collection, and the feedback loop that turns errors into insights.

---

*"In the struggle for survival, the fittest win out at the expense of their rivals because they succeed in adapting themselves best to their environment." — Charles Darwin*

Islands succeed by adapting to **multiple** environments simultaneously, then sharing their adaptations. It's parallel evolution with benefits.
