# Architecture Overview: The Evolutionary Engine

## The Big Picture

OpenEvolve's architecture is best understood as a **closed-loop evolutionary system** where code mutates, gets tested, and the results feed back to guide future mutations. Think of it as a software laboratory that runs 24/7, conducting thousands of experiments automatically.

Here's the conceptual flow:

```
┌──────────────────────────────────────────────────────────┐
│                    OpenEvolve Controller                  │
│         "The orchestrator of the entire process"          │
│                                                            │
│  • Manages checkpoints (can pause/resume experiments)     │
│  • Tracks the absolute best program across all time       │
│  • Coordinates parallel workers                           │
│  • Handles graceful shutdown on SIGTERM/SIGINT            │
└───────────────┬──────────────────────────────────────────┘
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│Database │ │Parallel │ │Evaluator │
│(MAP-    │ │Process  │ │(Cascade) │
│Elites)  │ │Control) │ │          │
└─────────┘ └─────────┘ └──────────┘
      │         │         │
      └─────────┼─────────┘
                ▼
        ┌───────────────┐
        │  LLM Ensemble │
        │  (GPT-4,      │
        │   Claude,     │
        │   Gemini...)  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Prompt Builder│
        │ (Context from │
        │  Population)  │
        └───────────────┘
```

Let's explore each component and understand **why** it exists.

## Component 1: The Controller (`controller.py`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/controller.py`

The Controller is the **maestro** conducting the evolutionary symphony. Its job is deceptively simple: run iterations until a target is reached or we run out of time. But the devil is in the details.

### Key Responsibilities

**1. Initialization**

When you start OpenEvolve, the Controller:
- Loads configuration from YAML
- Initializes the database (creating the MAP-Elites grid structure)
- Sets up the LLM ensemble with weighted sampling
- Prepares the evaluator with cascade thresholds
- Configures parallel processing workers

**2. Deterministic Reproducibility**

This is subtle but critical. Notice this pattern:

```python
if self.config.random_seed is not None:
    import random, numpy as np, hashlib

    # Set global random seeds
    random.seed(self.config.random_seed)
    np.random.seed(self.config.random_seed)

    # Create component-specific seeds via hashing
    base_seed = str(self.config.random_seed).encode("utf-8")
    llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

    self.config.llm.random_seed = llm_seed
```

**Why the hash-based derivation?** Because we need:
- **Different** seeds for different components (to avoid correlated randomness)
- **Deterministic** mapping from base seed to component seeds
- **Reproducible** experiments even with stochastic LLM sampling

Run the same experiment with seed 42 today and in six months—you'll get identical results.

**3. Checkpoint Management**

Research-grade experiments might run for days. Power outages happen. Cloud instances get preempted. The Controller handles this:

```python
checkpoint = {
    "iteration": current_iteration,
    "database_state": self.database.serialize(),
    "absolute_best_program": self.absolute_best_program.serialize(),
    "config": self.config.to_dict(),
    "random_state": random.getstate()
}
save_checkpoint(checkpoint_path, checkpoint)
```

You can stop evolution at iteration 5000, restart a week later, and it picks up at iteration 5001 like nothing happened.

**4. Absolute Best Tracking**

Here's a subtle design decision. The database uses MAP-Elites, which means programs can be **evicted** from the population if a better program occupies their grid cell. But we never want to lose the absolute best solution found.

```python
if is_better(new_program, self.absolute_best_program):
    self.absolute_best_program = new_program.copy()
    save_program(new_program, "best_program.py")
```

This is kept **separate** from the database to ensure we can always retrieve the champion.

## Component 2: The Database (`database.py`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/database.py`

This is where the magic of **quality-diversity** happens. The database doesn't just store programs—it organizes them in a multi-dimensional **feature space** to maintain diversity while optimizing quality.

### The MAP-Elites Algorithm

Think of MAP-Elites as creating a **museum of solutions**. Each room in the museum is defined by characteristics (features):

- Room (0,0): Low complexity, low diversity
- Room (0,5): Low complexity, high diversity
- Room (3,2): Medium complexity, medium diversity
- ...and so on

**Rule**: Each room can only hold ONE program—the best one with those characteristics.

When a new program arrives:
1. Calculate its features: `complexity = len(code)`, `diversity = distance_to_reference_set`, etc.
2. Map to grid coordinates: `coords = [complexity_bin, diversity_bin, ...]`
3. Check the room: Does it have a program already?
   - **No**: Add this program (new niche discovered!)
   - **Yes**: Is our program better? If yes, evict the old one. If no, discard ours.

### Why This Works

Traditional genetic algorithms maintain a population by ranking programs: best 1st, second-best 2nd, etc. Problem? They converge to a **narrow region** of the solution space.

MAP-Elites maintains diversity by **forcing** different niches. Even if a complex solution is better overall, we still keep simpler solutions if they're the best in their complexity class.

This gives evolution:
- **Stepping stones**: Can evolve complex solutions via simpler intermediate forms
- **Robustness**: Multiple solutions to choose from if requirements change
- **Exploration**: Diverse parents lead to diverse offspring

### Island-Based Populations

Now add another layer: Instead of one MAP-Elites grid, we have **multiple isolated grids** (islands).

```python
self.islands = [
    IslandPopulation(id=0, programs=set(), ...),
    IslandPopulation(id=1, programs=set(), ...),
    IslandPopulation(id=2, programs=set(), ...)
]
```

Each island evolves **independently** for N generations, then **migrants** move between islands:

```python
def migrate_programs(self):
    for i, island in enumerate(self.islands):
        # Select top programs from island i
        migrants = top_programs[:num_to_migrate]

        # Send to adjacent islands (ring topology)
        target_islands = [(i + 1) % num_islands, (i - 1) % num_islands]

        for migrant in migrants:
            if not migrant.metadata.get("migrant"):  # Prevent re-migration
                migrant_copy = copy_with_metadata(migrant, migrant=True)
                add_to_island(target_island, migrant_copy)
```

**Why islands?** Genetic diversity! If all programs breed together, you get convergence. Isolated islands allow different evolutionary strategies to develop, then the best strategies spread via migration.

This mirrors **island biogeography** in nature—isolated populations evolve unique traits, then occasional migration shares innovations.

## Component 3: The Evaluator (`evaluator.py`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/evaluator.py`

Testing programs is expensive. Running a full benchmark suite on every mutation would be prohibitively slow. The Evaluator solves this with **cascade evaluation**:

```
Stage 1: Quick smoke tests (1 second)
   ↓ (only if passes threshold)
Stage 2: Basic functionality (10 seconds)
   ↓ (only if passes threshold)
Stage 3: Comprehensive evaluation (60 seconds)
```

### The Cascade Pattern

```python
async def _cascade_evaluate(self, program_path):
    # Stage 1: Does it even run?
    stage1_result = await run_with_timeout(stage1_evaluator, timeout=1)

    if stage1_result.metrics["score"] < threshold[0]:
        return stage1_result  # Fail fast!

    # Stage 2: Basic correctness
    stage2_result = await run_with_timeout(stage2_evaluator, timeout=10)
    combined = merge_metrics(stage1_result, stage2_result)

    if combined["score"] < threshold[1]:
        return combined

    # Stage 3: Full evaluation (expensive!)
    stage3_result = await run_with_timeout(stage3_evaluator, timeout=60)
    return merge_all(stage1, stage2, stage3)
```

**Why this matters**: 90% of mutations are broken or worse than parents. We detect this in Stage 1 (1 second) and move on. Only the promising 10% get expensive evaluation.

Result? **10-100x speedup** compared to always running full evaluations.

### The Artifact Side-Channel

Here's a beautiful design choice. Evaluation doesn't just return a score—it returns **rich debugging data**:

```python
return EvaluationResult(
    metrics={
        "score": 0.85,
        "correctness": 1.0,
        "performance": 0.85
    },
    artifacts={
        "stderr": "RuntimeWarning: overflow in exp",
        "failed_test_cases": ["test_edge_case_3"],
        "profiling": {"time_in_function_A": 0.8},
        "llm_code_review": "Variable names are unclear"
    }
)
```

These artifacts get included in the **next generation's prompt**:

```
Your previous attempt had:
- stderr: "RuntimeWarning: overflow in exp"
- failed test case: test_edge_case_3

Please fix these issues in your next mutation.
```

This creates a **learning loop**. The LLM sees not just what failed, but *why* it failed.

## Component 4: Parallel Processing (`process_parallel.py`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/process_parallel.py`

Evolution is embarrassingly parallel—you can evaluate 100 mutations simultaneously. But Python's GIL makes this tricky. OpenEvolve uses **process-based parallelism**:

```python
# Main process
def run_parallel_iterations(num_parallel=10):
    # Create worker pool
    with ProcessPoolExecutor(max_workers=num_parallel) as executor:
        # Submit tasks
        futures = []
        for _ in range(num_parallel):
            parent_id = database.sample_parent()
            inspiration_ids = database.sample_inspirations()

            # Serialize database state (snapshot)
            db_snapshot = database.create_snapshot()

            future = executor.submit(
                _run_iteration_worker,
                iteration=i,
                db_snapshot=db_snapshot,
                parent_id=parent_id,
                inspiration_ids=inspiration_ids
            )
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            child_program = future.result()
            database.add(child_program)
```

### The Worker Pattern

Each worker process:
1. **Receives** a database snapshot (serialized programs)
2. **Reconstructs** the parent and inspiration programs
3. **Builds** a prompt from this context
4. **Calls** the LLM API
5. **Evaluates** the resulting program
6. **Returns** the child program (serialized)

Key insight: Workers don't share state. They get **immutable snapshots**. This avoids:
- Lock contention
- Race conditions
- Pickle errors with complex objects

Trade-off: Database snapshots use memory. But modern machines have RAM to spare, and the parallelism speedup is worth it.

## Component 5: LLM Integration (`llm/`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/llm/`

OpenEvolve treats LLMs as **mutation operators** in the evolutionary algorithm. But not all LLMs are equal—GPT-4 is better at some tasks, Claude at others, Gemini at yet others.

Solution? **Ensemble with weighted sampling**:

```python
class LLMEnsemble:
    def __init__(self, models_config):
        self.models = [
            OpenAILLM(config=cfg) for cfg in models_config
        ]
        self.weights = [m.weight for m in models_config]
        # Normalize: [3, 2, 1] → [0.5, 0.33, 0.17]
        self.weights = normalize(self.weights)

    async def generate(self, prompt):
        # Sample model based on weights
        model = random.choices(self.models, weights=self.weights)[0]
        return await model.generate(prompt)
```

Configuration might look like:

```yaml
llm:
  models:
    - provider: openai
      model: gpt-4-turbo
      weight: 3  # Used 50% of the time
    - provider: anthropic
      model: claude-3-opus
      weight: 2  # Used 33% of the time
    - provider: google
      model: gemini-pro
      weight: 1  # Used 17% of the time
```

This gives **diversity** in mutations. Different models have different inductive biases, leading to different exploration strategies.

## Component 6: Prompt System (`prompt/`)

**Location**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/prompt/`

Prompts are the **DNA of evolution**. They package:
1. The current program (parent)
2. Evolution history (previous attempts and their scores)
3. Top programs (inspiration from high performers)
4. Artifacts (error messages, profiling data, feedback)
5. Task description
6. Instructions for mutation

### Template Stochasticity

To prevent LLMs from getting stuck in local optima, prompts vary randomly:

```python
def build_prompt(self, ...):
    # Randomly select greeting
    greeting = random.choice([
        "Let's enhance this code:",
        "Time to optimize:",
        "Improving the algorithm:"
    ])

    # Randomly select instruction phrasing
    instruction = random.choice([
        "Modify the code to improve performance.",
        "Evolve the implementation for better results.",
        "Enhance the algorithm's effectiveness."
    ])

    return f"{greeting}\n\n{current_code}\n\n{instruction}"
```

Same semantic content, different surface forms. This prevents mode collapse where the LLM returns similar mutations.

### Fitness vs. Features

Critical distinction in prompts:

```python
# WRONG: Include all metrics as "score"
fitness = metrics["performance"] + metrics["diversity"] + metrics["complexity"]

# RIGHT: Separate fitness from features
fitness_metrics = {k: v for k, v in metrics.items()
                   if k not in feature_dimensions}
fitness = sum(fitness_metrics.values())

feature_metrics = {k: v for k, v in metrics.items()
                   if k in feature_dimensions}
```

Why? Features define **where** a program sits in the grid (diversity). Fitness defines **how good** it is. Mixing them confuses the optimization objective.

The LLM is told: "Your program scored 0.85 on performance (good!), and it has complexity 150 (that's just descriptive)." Not: "Your program scored 0.85 + 0.3 = 1.15 (nonsense!)."

## How It All Fits Together

Let's trace one iteration through the entire system:

**1. Controller says**: "Time for iteration 1000"

**2. Database samples**:
- Parent: Program #847 from island 2
- Inspirations: Programs #23, #445, #892 (top performers from island 2)

**3. Worker receives snapshot**:
```python
{
    "parent_id": "847",
    "inspiration_ids": ["23", "445", "892"],
    "programs": {
        "847": {"code": "def search(...)...", "metrics": {...}},
        "23": {"code": "...", "metrics": {...}},
        ...
    }
}
```

**4. Prompt Builder creates context**:
```
You are evolving a search algorithm.

Current best score: 0.92

Your parent program (score: 0.78):
```python
def search():
    # ... parent code ...
```

Evolution history:
- Generation 998: score 0.75 (improved temperature schedule)
- Generation 999: score 0.78 (added adaptive step size)

Top-performing programs for inspiration:
- Program A (score: 0.92): Uses simulated annealing
- Program B (score: 0.88): Uses gradient estimation

Artifacts from previous attempt:
- stderr: "RuntimeWarning: overflow in exp"

Please create an improved version.
```

**5. LLM Ensemble** samples model (say, GPT-4 with 50% probability) and generates:
```python
def search():
    # ... mutated code with overflow fix ...
```

**6. Evaluator** runs cascade:
- Stage 1: ✓ Passes (1 sec)
- Stage 2: ✓ Passes (10 sec)
- Stage 3: Score 0.81 (60 sec)

**7. Worker returns**: Child program with metrics `{"score": 0.81}`

**8. Database adds** child to island 2:
- Features: complexity=165, diversity=0.42
- Grid cell: [5, 4]
- Current occupant of [5, 4]: score 0.75
- **Evict old, insert new** (0.81 > 0.75)

**9. Controller checks**: Is 0.81 > absolute_best (0.92)? No. Continue.

**10. Repeat** for iterations 1001, 1002, ..., until target reached or budget exhausted.

## Architectural Principles

Stepping back, why is the architecture designed this way?

**1. Separation of Concerns**
- Controller: orchestration
- Database: population management
- Evaluator: testing
- LLM: mutation
- Prompt: context packaging

Each component has one job. Easy to test, modify, replace.

**2. Immutable Data Flow**
- Snapshots, not shared state
- Enables parallelism
- Reduces bugs

**3. Fail-Safe Design**
- Checkpointing for recovery
- Timeouts for evaluation
- Graceful degradation (if LLM fails, try next model)

**4. Scientific Reproducibility**
- Deterministic seeding
- Logging of all decisions
- Export traces for analysis

**5. Extensibility**
- Pluggable LLMs (add new model? Just implement interface)
- Custom evaluators (bring your own benchmark)
- Template system (modify prompts without code changes)

## Next Steps

Now that we understand the architecture, we'll dive deeper into the most intricate components:

- **Chapter 3**: The MAP-Elites database and feature calculation
- **Chapter 4**: Island populations and migration dynamics
- **Chapter 5**: Cascade evaluation and artifact feedback
- **Chapter 6**: LLM integration and prompt engineering

Each chapter will include actual code snippets from the implementation to ground the concepts in reality.

---

*The beauty of evolution is that it's simple rules applied recursively lead to profound complexity. OpenEvolve's architecture mirrors this—straightforward components, sophisticated emergent behavior.*
