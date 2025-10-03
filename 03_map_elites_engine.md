# The MAP-Elites Engine: Quality-Diversity in Action

## The Problem with Traditional Evolution

Before we dive into MAP-Elites, let's understand what it solves.

Traditional genetic algorithms maintain a population ranked by fitness:

```
Generation 100:
1. Program A: score 0.95 (elite!)
2. Program B: score 0.94 (almost as good)
3. Program C: score 0.93 (pretty good)
...
100. Program Z: score 0.50 (will be discarded)
```

The next generation is bred from the top programs. Sounds reasonable, right?

**The trap**: Programs A, B, and C might all be variations of the same approach—say, simulated annealing with slightly different parameters. The population **converges** to one region of solution space.

What if there's a completely different approach—genetic algorithms, tabu search, ant colony optimization—that could work better? The system will never find it because it's too far from the current solutions.

This is called **premature convergence**, and it's the death of creativity in optimization.

## Enter MAP-Elites: The Museum of Solutions

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) takes a radically different approach. Instead of asking "What's the best program?", it asks:

**"What's the best program *of each kind*?"**

Imagine a museum with rooms arranged in a grid:

```
                    Diversity →
            Low         Medium         High
        ┌──────────┬──────────┬──────────┐
  Low   │  Room    │  Room    │  Room    │
  ↓     │  (0,0)   │  (0,1)   │  (0,2)   │
        ├──────────┼──────────┼──────────┤
Complex │  Room    │  Room    │  Room    │
  ↓     │  (1,0)   │  (1,1)   │  (1,2)   │
        ├──────────┼──────────┼──────────┤
  High  │  Room    │  Room    │  Room    │
  ↓     │  (2,0)   │  (2,1)   │  (2,2)   │
        └──────────┴──────────┴──────────┘
```

Each room is defined by **behavioral characteristics** (called *features*):
- **Complexity**: How many lines of code?
- **Diversity**: How different from other solutions?

**Rule**: Each room holds exactly **one program**—the highest-scoring program with those characteristics.

When a new program arrives:
1. Measure its features: `complexity = 150 lines`, `diversity = 0.7`
2. Map to grid: `room = (1, 2)` (medium complexity, high diversity)
3. Check room (1,2):
   - **Empty?** Add program. New niche discovered!
   - **Occupied?** Compare scores. Better? Replace. Worse? Discard.

## Why This Is Brilliant

MAP-Elites maintains **both** quality and diversity:

- **Quality**: Each room holds the *best* program of that type
- **Diversity**: Rooms force *different* types of programs to coexist

Even if complex solutions generally score higher, we keep simple solutions if they're the best in their simplicity class. This gives us:

1. **Stepping Stones**: Evolution can build complex solutions by improving simpler ones
2. **Robustness**: If requirements change (e.g., "now we need simple code"), we have options
3. **Exploration**: Diverse parents generate diverse offspring, exploring more of solution space

## OpenEvolve's Implementation

Let's look at the actual code. The MAP-Elites grid is implemented in `database.py`:

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/database.py`

### Data Structures

```python
class ProgramDatabase:
    def __init__(self, config):
        # Core MAP-Elites structure
        self.feature_maps = {}  # {feature_key: program_id}
        self.programs = {}      # {program_id: Program object}

        # Feature configuration
        self.feature_dimensions = config.feature_dimensions  # ["complexity", "diversity"]
        self.feature_bins = config.feature_bins  # Number of bins per dimension

        # Statistics for normalization
        self.feature_stats = defaultdict(lambda: {
            "min": float("inf"),
            "max": float("-inf"),
            "values": []
        })
```

The grid is represented as a **sparse dictionary**: `feature_maps`. Why sparse? Because most cells will be empty (especially in high dimensions). No need to allocate memory for empty rooms.

### Feature Calculation

This is where the magic happens. Given a program, how do we map it to grid coordinates?

```python
def _calculate_feature_coords(self, program: Program) -> List[int]:
    """Map a program to grid coordinates based on its features."""
    coords = []

    for dim in self.feature_dimensions:
        if dim == "complexity":
            # Complexity = code length
            complexity = len(program.code)
            # Map to bin [0, feature_bins)
            bin_idx = self._calculate_complexity_bin(complexity)

        elif dim == "diversity":
            # Diversity = average distance to reference set
            diversity = self._get_cached_diversity(program)
            bin_idx = self._calculate_diversity_bin(diversity)

        elif dim in program.metrics:
            # Use custom metric from evaluation
            score = program.metrics[dim]

            # Update statistics for normalization
            self._update_feature_stats(dim, score)

            # Scale to [0, 1] using percentile-based normalization
            scaled_value = self._scale_feature_value(dim, score)

            # Map to bin
            num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
            bin_idx = int(scaled_value * num_bins)
            bin_idx = max(0, min(num_bins - 1, bin_idx))

        else:
            raise ValueError(f"Unknown feature dimension: {dim}")

        coords.append(bin_idx)

    return coords  # e.g., [2, 5, 1] for 3D grid
```

### Complexity Feature

```python
def _calculate_complexity_bin(self, complexity: int) -> int:
    """Bin programs by code size."""
    if complexity < 50:
        return 0  # Very simple
    elif complexity < 200:
        return 1  # Simple
    elif complexity < 500:
        return 2  # Medium
    elif complexity < 1000:
        return 3  # Complex
    else:
        return 4  # Very complex
```

Simple thresholding. Small programs in low bins, large programs in high bins.

### Diversity Feature

This is more sophisticated. Diversity measures how **different** a program is from others:

```python
def _calculate_diversity(self, program: Program) -> float:
    """Calculate diversity as average distance to reference set."""
    if len(self.diversity_reference_set) == 0:
        return 0.0

    # Calculate edit distance to each reference program
    distances = []
    for ref_program in self.diversity_reference_set:
        distance = self._edit_distance(program.code, ref_program.code)
        distances.append(distance)

    # Diversity = average distance
    return sum(distances) / len(distances)

def _edit_distance(self, code1: str, code2: str) -> float:
    """Levenshtein distance normalized by length."""
    import Levenshtein
    distance = Levenshtein.distance(code1, code2)
    max_len = max(len(code1), len(code2))
    return distance / max_len if max_len > 0 else 0.0
```

**Key insight**: Diversity is calculated against a **reference set** (randomly sampled programs). Why not calculate distance to *all* programs? Performance! With 10,000 programs, that's 10,000 comparisons per new program. Instead, we compare to 100 reference programs—good enough proxy, 100x faster.

### Custom Metrics as Features

You can use **any metric** from evaluation as a feature dimension:

```python
# Configuration
feature_dimensions:
  - complexity
  - diversity
  - memory_usage     # Custom metric!
  - cache_misses     # Another custom metric!
```

When the evaluator returns:

```python
{
    "score": 0.85,  # Fitness (what we optimize)
    "memory_usage": 1024,  # Feature dimension
    "cache_misses": 150    # Feature dimension
}
```

Programs are binned by memory usage and cache misses, creating a grid of *memory-efficient solutions* vs. *cache-friendly solutions*.

This is powerful: You can explore trade-offs explicitly.

### Adding Programs to the Grid

```python
def add(self, program: Program) -> bool:
    """Add program to database, potentially replacing existing program in its cell."""
    # Calculate where this program belongs
    coords = self._calculate_feature_coords(program)
    feature_key = self._feature_coords_to_key(coords)  # e.g., "2_5_1"

    # Check if cell is occupied
    if feature_key in self.feature_maps:
        existing_id = self.feature_maps[feature_key]
        existing_program = self.programs[existing_id]

        # Calculate fitness (excluding feature dimensions!)
        new_fitness = get_fitness_score(
            program.metrics,
            exclude=self.feature_dimensions
        )
        existing_fitness = get_fitness_score(
            existing_program.metrics,
            exclude=self.feature_dimensions
        )

        if new_fitness > existing_fitness:
            # Replace existing program
            del self.programs[existing_id]
            self.feature_maps[feature_key] = program.id
            self.programs[program.id] = program
            return True  # Cell conquered!
        else:
            return False  # Cell defended.
    else:
        # Empty cell, add program
        self.feature_maps[feature_key] = program.id
        self.programs[program.id] = program
        return True  # New niche discovered!
```

**Critical detail**: Fitness is calculated **excluding feature dimensions**. Why?

If `feature_dimensions = ["complexity", "memory_usage"]` and `metrics = {"score": 0.8, "complexity": 100, "memory_usage": 512}`, then:

```python
fitness = get_fitness_score(metrics, exclude=["complexity", "memory_usage"])
# Returns: 0.8 (only the "score" metric)
```

We don't want to optimize *for* complexity—complexity is just a *descriptor* of where the program sits in the grid. We optimize for score *within* each complexity/memory class.

### Sampling Parents

Evolution needs to select parents from the population. MAP-Elites sampling is uniform across occupied cells:

```python
def sample_parent(self) -> Program:
    """Sample a parent uniformly from the population."""
    if len(self.programs) == 0:
        raise ValueError("Cannot sample from empty database")

    # Uniform random selection
    program_id = random.choice(list(self.programs.keys()))
    return self.programs[program_id]
```

Why uniform? To give every niche equal chance to contribute. Elite-biased sampling would focus on high-fitness regions; uniform sampling explores the entire grid.

### Sampling Inspirations

For the prompt, we also need **high-performing inspirations**:

```python
def sample_top_programs(self, n: int = 5) -> List[Program]:
    """Sample from top programs by fitness."""
    # Sort programs by fitness
    sorted_programs = sorted(
        self.programs.values(),
        key=lambda p: get_fitness_score(p.metrics, exclude=self.feature_dimensions),
        reverse=True
    )

    # Return top N
    return sorted_programs[:n]
```

This gives the LLM examples of what "good" looks like.

## Feature Scaling: The Percentile Trick

One challenge: Features have different ranges.
- Complexity: 10 to 10,000 (lines of code)
- Memory: 100KB to 10GB
- Score: 0.0 to 1.0

If we use raw values for binning, everything goes in weird buckets. Solution? **Percentile-based normalization**:

```python
def _scale_feature_value(self, dimension: str, value: float) -> float:
    """Scale feature to [0, 1] using percentile rank."""
    stats = self.feature_stats[dimension]

    # Collect all values seen so far
    values = stats["values"]
    values.append(value)

    # Calculate percentile rank
    rank = sum(1 for v in values if v <= value)
    percentile = rank / len(values)

    return percentile  # In [0, 1]
```

A program with memory usage at the 90th percentile gets mapped to 0.9, regardless of absolute value. This automatically adapts as the population evolves.

## Artifact Storage

Programs can have large artifacts (profiling data, visualizations). Storing them in memory would explode RAM usage. OpenEvolve uses a hybrid approach:

```python
def _store_artifacts(self, program: Program):
    """Store artifacts, using disk for large ones."""
    for key, value in program.artifacts.items():
        # Small artifacts (< 1MB): Store in metadata
        if sys.getsizeof(value) < 1_000_000:
            program.metadata[f"artifact_{key}"] = value
        else:
            # Large artifacts: Save to disk
            artifact_path = self.artifact_dir / f"{program.id}_{key}.json"
            with open(artifact_path, "w") as f:
                json.dump(value, f)
            program.metadata[f"artifact_{key}_path"] = str(artifact_path)
```

When building prompts, we check metadata:

```python
if "artifact_stderr_path" in program.metadata:
    # Load from disk
    with open(program.metadata["artifact_stderr_path"]) as f:
        stderr = json.load(f)
else:
    stderr = program.metadata.get("artifact_stderr", "")
```

## Visualization: What Does the Grid Look Like?

After 1000 iterations, you might have:

```
        Diversity →
        0    1    2    3    4
      ┌────┬────┬────┬────┬────┐
  0   │ ██ │ ██ │ ██ │    │    │  ← Simple programs (all 3 diversity levels populated)
      ├────┼────┼────┼────┼────┤
C 1   │ ██ │ ██ │ ██ │ ██ │    │  ← Medium complexity (4 niches)
o     ├────┼────┼────┼────┼────┤
m 2   │    │ ██ │ ██ │ ██ │ ██ │  ← High complexity (4 niches)
p     ├────┼────┼────┼────┼────┤
l 3   │    │    │ ██ │ ██ │    │  ← Very complex (2 niches)
e     └────┴────┴────┴────┴────┘
x
```

Filled cells (██) contain programs. Empty cells haven't been filled yet (opportunities for exploration!).

Over time, evolution fills more cells (discovering new niches) and improves programs in filled cells (optimizing within niches).

## The Power of Quality-Diversity

Let's see a concrete example from the `function_minimization` task:

**Generation 0**:
```python
# Cell (0, 0): Simple, not diverse
def search():
    return random.uniform(-5, 5), random.uniform(-5, 5)
# Score: 0.1 (terrible, but it's the best simple solution we have)
```

**Generation 100**:
```python
# Cell (0, 0): Simple, not diverse - IMPROVED
def search():
    best = None
    for _ in range(100):
        x, y = random.uniform(-5, 5), random.uniform(-5, 5)
        if best is None or eval(x,y) < eval(*best):
            best = (x, y)
    return best
# Score: 0.3 (better! Still simple.)

# Cell (2, 1): Complex, medium diversity - NEW NICHE
def search():
    temperature = 1.0
    current = random_point()
    for i in range(1000):
        neighbor = mutate(current, step=temperature)
        if accept(neighbor, current, temperature):
            current = neighbor
        temperature *= 0.99
    return current
# Score: 0.8 (much better! Discovered simulated annealing.)
```

**Generation 500**:
```python
# Cell (0, 0): Simple - DEFENDED
# (Still the generation-100 program, because no simpler solution beat it)

# Cell (2, 1): Complex - IMPROVED
def search():
    # Adaptive simulated annealing with restart
    best = None
    for restart in range(5):
        temperature = 1.0
        current = random_point()
        for i in range(200):
            step_size = temperature * (1 + 0.1 * np.random.randn())
            neighbor = mutate(current, step=step_size)
            if accept(neighbor, current, temperature):
                current = neighbor
            temperature *= 0.98
        if best is None or eval(*current) < eval(*best):
            best = current
    return best
# Score: 0.95 (excellent! Evolved sophisticated annealing.)
```

Notice:
- Simple solutions persist (generation 100's random search is still in cell (0,0))
- Complex solutions evolve rapidly (cell (2,1) goes from 0.8 to 0.95)
- Diversity is maintained (we have *both* simple and complex solutions)

If requirements change ("we need a fast solution that runs in < 10ms"), we can grab the simple solution from cell (0,0). If we need the absolute best, we grab from cell (2,1).

## Configuration

Here's how you configure MAP-Elites in `config.yaml`:

```yaml
database:
  # Number of programs in population
  population_size: 1000

  # Feature dimensions for MAP-Elites grid
  feature_dimensions:
    - complexity      # Built-in: code length
    - diversity       # Built-in: distance to reference set

  # Number of bins per dimension
  feature_bins: 5     # Default for all dimensions
  feature_bins_per_dim:
    complexity: 5     # Override for specific dimension
    diversity: 10

  # Diversity calculation
  diversity_reference_size: 100  # Number of programs in reference set

  # Feature scaling
  feature_scaling: "percentile"  # Options: "percentile", "min-max"
```

You can also add custom features:

```yaml
feature_dimensions:
  - complexity
  - diversity
  - memory_efficiency  # Expects program.metrics["memory_efficiency"]
  - parallelism_score  # Expects program.metrics["parallelism_score"]
```

As long as your evaluator returns these metrics, MAP-Elites will use them for binning.

## Limitations and Trade-offs

MAP-Elites is powerful, but not perfect:

**1. Curse of Dimensionality**

With 3 features and 10 bins each, you have 10³ = 1,000 cells. With 5 features, 10⁵ = 100,000 cells. Most will be empty.

Solution? Use sparse representations (dictionaries, not arrays) and accept that high-dimensional grids won't be fully explored.

**2. Feature Engineering**

You need to *define* meaningful features. "Complexity" and "diversity" are general, but domain-specific features require domain knowledge.

For GPU kernels: memory coalescing, register usage, occupancy
For algorithms: time complexity, space complexity, numerical stability

This is both a strength (explicit control over diversity) and a weakness (requires thought).

**3. Fitness Calculation**

You must carefully separate fitness from features. Mixing them leads to weird behaviors (optimizing *for* complexity instead of *within* complexity classes).

## Next Chapter

We've seen how MAP-Elites creates a diverse population. But OpenEvolve adds another layer: **island populations**. Instead of one MAP-Elites grid, it maintains multiple isolated grids that evolve independently.

Why? To prevent premature convergence across the *entire* grid space. Even if one island converges, others might explore different regions.

In the next chapter, we'll explore island-based evolution and migration dynamics.

---

*"The secret to creativity is knowing how to hide your sources." — Albert Einstein*

MAP-Elites doesn't hide them—it catalogs them meticulously, creating a living museum of algorithmic diversity.
