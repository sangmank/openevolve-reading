# Examples and Applications: Evolution in Action

## From Theory to Practice

We've explored the architecture, the algorithms, the implementation details. Now let's see OpenEvolve in action across diverse domains. Each example reveals different aspects of what evolutionary coding can achieve.

## Example 1: Function Minimization - Discovering Algorithms from Scratch

**Location**: `examples/function_minimization/`

**Challenge**: Minimize the Rastrigin function, a highly multimodal test function:

```python
def rastrigin(x, y):
    A = 10
    return (
        2 * A +
        (x**2 - A * np.cos(2 * np.pi * x)) +
        (y**2 - A * np.cos(2 * np.pi * y))
    )
```

This function has countless local minima. Global minimum at (0, 0) with value 0.

### Initial Program (Generation 0)

```python
# EVOLVE-BLOCK-START
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """Naive random search."""
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = rastrigin(best_x, best_y)

    for _ in range(iterations):
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = rastrigin(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value
# EVOLVE-BLOCK-END
```

**Fitness**: ~0.15 (terrible—random search rarely finds the global minimum)

### Evolution Progression

**Generation 50**: Basic simulated annealing

```python
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """Discovered simulated annealing concept."""
    temperature = 1.0
    current_x = np.random.uniform(bounds[0], bounds[1])
    current_y = np.random.uniform(bounds[0], bounds[1])
    current_value = rastrigin(current_x, current_y)

    for i in range(iterations):
        # Mutation with temperature-scaled step
        step_x = np.random.randn() * temperature
        step_y = np.random.randn() * temperature
        new_x = np.clip(current_x + step_x, bounds[0], bounds[1])
        new_y = np.clip(current_y + step_y, bounds[0], bounds[1])
        new_value = rastrigin(new_x, new_y)

        # Accept if better, or probabilistically if worse
        if new_value < current_value:
            current_x, current_y, current_value = new_x, new_y, new_value
        elif np.random.rand() < np.exp(-(new_value - current_value) / temperature):
            current_x, current_y, current_value = new_x, new_y, new_value

        # Cool down
        temperature *= 0.995

    return current_x, current_y, current_value
```

**Fitness**: ~0.55 (major improvement! Discovered temperature-based exploration)

**How?** The LLM saw:
- Artifact from generation 30: "Stuck in local minimum"
- Top program: Had a parameter that decreased over time
- Evolution history: Programs with "adaptive" behavior scored higher

It connected the dots: Adapt search intensity over time = simulated annealing.

**Generation 150**: Adaptive multi-start annealing

```python
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """Advanced: Multi-start with adaptive temperature and restart."""
    best_overall_x, best_overall_y, best_overall_value = None, None, float('inf')

    # Multiple restarts
    restarts = 5
    iterations_per_restart = iterations // restarts

    for restart in range(restarts):
        # Initialize with previous best if available (warm start)
        if best_overall_x is not None and restart > 0:
            current_x = best_overall_x + np.random.randn() * 0.5
            current_y = best_overall_y + np.random.randn() * 0.5
            current_x = np.clip(current_x, bounds[0], bounds[1])
            current_y = np.clip(current_y, bounds[0], bounds[1])
        else:
            current_x = np.random.uniform(bounds[0], bounds[1])
            current_y = np.random.uniform(bounds[0], bounds[1])

        current_value = rastrigin(current_x, current_y)
        temperature = 2.0  # Higher initial temperature

        stagnation_counter = 0
        best_in_restart_value = current_value

        for i in range(iterations_per_restart):
            # Adaptive step size based on temperature and stagnation
            step_scale = temperature * (1 + 0.1 * np.random.rand())
            step_x = np.random.randn() * step_scale
            step_y = np.random.randn() * step_scale
            new_x = np.clip(current_x + step_x, bounds[0], bounds[1])
            new_y = np.clip(current_y + step_y, bounds[0], bounds[1])
            new_value = rastrigin(new_x, new_y)

            # Acceptance criterion
            if new_value < current_value:
                current_x, current_y, current_value = new_x, new_y, new_value
                stagnation_counter = 0
            else:
                acceptance_prob = np.exp(-(new_value - current_value) / temperature)
                if np.random.rand() < acceptance_prob:
                    current_x, current_y, current_value = new_x, new_y, new_value
                stagnation_counter += 1

            # Track best in this restart
            if current_value < best_in_restart_value:
                best_in_restart_value = current_value

            # Adaptive cooling (faster cooling if stagnating)
            if stagnation_counter > 50:
                temperature *= 0.98
            else:
                temperature *= 0.995

            # Early termination if temperature too low
            if temperature < 0.01:
                break

        # Update global best
        if current_value < best_overall_value:
            best_overall_x, best_overall_y, best_overall_value = current_x, current_y, current_value

    return best_overall_x, best_overall_y, best_overall_value
```

**Fitness**: ~0.92 (excellent! Near-optimal solutions consistently)

**Key Innovations** (discovered autonomously):
1. **Multi-start strategy**: Run multiple restarts to escape local minima
2. **Warm start**: Initialize new searches near previous best
3. **Adaptive cooling**: Cool faster when stuck, slower when improving
4. **Stagnation detection**: Track progress to adjust behavior
5. **Early termination**: Stop if temperature drops too low

None of these were explicitly taught. The LLM discovered them through evolution.

### What This Demonstrates

- **Algorithm discovery**: Started with random search, ended with sophisticated hybrid approach
- **Emergent complexity**: Simple mutations accumulated into complex behaviors
- **Transfer learning**: Concepts from top programs (restarts, adaptation) combined with parent program (annealing)

## Example 2: Circle Packing - State-of-the-Art Mathematical Optimization

**Location**: `examples/circle_packing/`

**Challenge**: Pack N circles into a unit circle, minimizing the outer circle's radius.

This is a well-studied problem in discrete geometry with published benchmarks. For N=26, the best-known solution has radius 2.635.

### Initial Program

```python
def pack_circles(n=26):
    """Random placement."""
    circles = []
    for i in range(n):
        # Random position
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        circles.append((x, y))

    # Calculate required radius
    radius = max(np.sqrt(x**2 + y**2) for x, y in circles) + 1.0/n

    return circles, radius
```

**Result**: radius ≈ 4.5 (terrible)

### Generation 190: Learning Structure

```python
def pack_circles(n=26):
    """Discovered: Use optimization to reduce overlaps."""
    from scipy.optimize import minimize

    # Initial placement: hexagonal grid
    circles = []
    sqrt3 = np.sqrt(3)
    for i in range(int(np.sqrt(n)) + 1):
        for j in range(int(np.sqrt(n)) + 1):
            if len(circles) >= n:
                break
            x = i * 0.5
            y = j * sqrt3 / 2
            circles.append((x, y))

    # Flatten to 1D array for optimization
    x0 = np.array(circles).flatten()

    def objective(positions):
        pos = positions.reshape(-1, 2)

        # Penalty for overlaps
        penalty = 0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 2.0/n:  # Circles overlap
                    penalty += (2.0/n - dist)**2

        # Penalty for being far from center
        radius = max(np.linalg.norm(p) for p in pos) + 1.0/n
        penalty += radius**2

        return penalty

    result = minimize(objective, x0, method='L-BFGS-B')
    final_positions = result.x.reshape(-1, 2)

    radius = max(np.linalg.norm(p) for p in final_positions) + 1.0/n

    return final_positions, radius
```

**Result**: radius ≈ 2.9 (getting closer!)

### Generation 460: State-of-the-Art

```python
def pack_circles(n=26):
    """Sophisticated optimization with multiple strategies."""
    from scipy.optimize import differential_evolution, minimize
    import itertools

    def calculate_radius(positions):
        """Calculate minimum radius to contain all circles."""
        pos = positions.reshape(-1, 2)
        return max(np.linalg.norm(p) for p in pos) + 1.0/n

    def objective(positions):
        """Objective: minimize radius + penalize overlaps."""
        pos = positions.reshape(-1, 2)

        # Hard constraint: no overlaps
        min_dist = float('inf')
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                min_dist = min(min_dist, dist)

        # Heavy penalty for overlaps
        if min_dist < 2.0/n:
            penalty = 1000 * (2.0/n - min_dist)**2
        else:
            penalty = 0

        radius = calculate_radius(positions.flatten())

        return radius + penalty

    # Strategy 1: Differential evolution (global search)
    bounds = [(-3, 3)] * (2 * n)  # x, y for each circle
    result_de = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=1000,
        popsize=30,
        strategy='best1bin'
    )

    # Strategy 2: Refine with gradient-based method
    result_refined = minimize(
        objective,
        result_de.x,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Strategy 3: Try symmetry-based initialization
    # (26 circles can have 13-fold symmetry)
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    symmetric_init = []
    for angle in angles:
        r1 = 1.5
        symmetric_init.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        r2 = 0.7
        symmetric_init.append([r2 * np.cos(angle + np.pi/13), r2 * np.sin(angle + np.pi/13)])
    symmetric_init = np.array(symmetric_init).flatten()

    result_symmetric = minimize(
        objective,
        symmetric_init,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Select best of all strategies
    candidates = [result_de, result_refined, result_symmetric]
    best = min(candidates, key=lambda r: objective(r.x))

    final_positions = best.x.reshape(-1, 2)
    radius = calculate_radius(best.x)

    return final_positions, radius
```

**Result**: radius ≈ **2.634** (matches published benchmark of 2.635!)

### Key Discoveries

1. **Hexagonal initialization**: Better than random
2. **Hybrid optimization**: Global search (DE) + local refinement (L-BFGS-B)
3. **Symmetry exploitation**: 13-fold symmetry for N=26
4. **Multi-strategy ensemble**: Try different approaches, pick best

The system **independently discovered** geometric insights that human researchers documented in papers.

### What This Demonstrates

- **Domain expertise emergence**: No knowledge of circle packing theory, yet discovered geometric principles
- **Hybrid methods**: Combined multiple optimization techniques
- **Competitive results**: Matched state-of-the-art on a benchmark problem

## Example 3: GPU Kernel Optimization - Hardware-Aware Evolution

**Location**: `examples/mlx_metal_kernel_opt/`

**Challenge**: Optimize an attention mechanism kernel for Apple Silicon GPUs (using MLX Metal shaders).

### Initial Kernel

```metal
// Baseline attention implementation
kernel void attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Naive implementation: global memory accesses
    float sum = 0.0;
    for (uint i = 0; i < seq_len; i++) {
        float qk = Q[tid * d_model + i] * K[tid * d_model + i];
        sum += exp(qk);
    }

    float attention_sum = 0.0;
    for (uint i = 0; i < seq_len; i++) {
        float qk = Q[tid * d_model + i] * K[tid * d_model + i];
        float attention_weight = exp(qk) / sum;
        attention_sum += attention_weight * V[tid * d_model + i];
    }

    output[tid] = attention_sum;
}
```

**Performance**: 0.8 GFLOPS (very slow)

### Evolved Kernel (Generation 300)

```metal
// Optimized with tiling, shared memory, and vectorization
kernel void attention_forward_optimized(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Shared memory for tiling (discovered optimization!)
    threadgroup float shared_Q[256];
    threadgroup float shared_K[256];
    threadgroup float shared_V[256];

    // Load into shared memory with coalesced access
    uint global_offset = group_id * 256 + local_id;
    if (global_offset < seq_len * d_model) {
        shared_Q[local_id] = Q[global_offset];
        shared_K[local_id] = K[global_offset];
        shared_V[local_id] = V[global_offset];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute attention using shared memory
    float sum = 0.0;
    float4 qk_vec;  // Vectorization (SIMD)

    // Process 4 elements at a time (discovered SIMD optimization!)
    for (uint i = 0; i < seq_len; i += 4) {
        qk_vec.x = shared_Q[local_id] * shared_K[i];
        qk_vec.y = shared_Q[local_id] * shared_K[i+1];
        qk_vec.z = shared_Q[local_id] * shared_K[i+2];
        qk_vec.w = shared_Q[local_id] * shared_K[i+3];

        // Fast exp approximation for Metal
        float4 exp_vec = fast::exp(qk_vec);
        sum += exp_vec.x + exp_vec.y + exp_vec.z + exp_vec.w;
    }

    // Second pass: compute weighted sum
    float attention_sum = 0.0;
    for (uint i = 0; i < seq_len; i += 4) {
        qk_vec.x = shared_Q[local_id] * shared_K[i];
        qk_vec.y = shared_Q[local_id] * shared_K[i+1];
        qk_vec.z = shared_Q[local_id] * shared_K[i+2];
        qk_vec.w = shared_Q[local_id] * shared_K[i+3];

        float4 exp_vec = fast::exp(qk_vec);
        float4 weights = exp_vec / sum;

        attention_sum += weights.x * shared_V[i];
        attention_sum += weights.y * shared_V[i+1];
        attention_sum += weights.z * shared_V[i+2];
        attention_sum += weights.w * shared_V[i+3];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        output[group_id] = attention_sum;
    }
}
```

**Performance**: 2.24 GFLOPS (**2.8x speedup!**)

### Hardware Optimizations Discovered

1. **Shared memory (threadgroup)**: Reduce global memory bandwidth
2. **Coalesced access**: Consecutive threads access consecutive memory (GPU-friendly)
3. **SIMD vectorization**: Process 4 elements with float4 (use GPU vector units)
4. **Fast math intrinsics**: `fast::exp()` instead of `exp()` (Apple Silicon specific)
5. **Barrier synchronization**: Proper use of `threadgroup_barrier`

### How Were These Discovered?

**Artifacts from evaluator**:

```python
{
    "performance": 0.8,  # GFLOPS
    "profiling": {
        "memory_bandwidth": "35% utilization",  # Low!
        "compute_utilization": "20%",           # GPU mostly idle
        "bottleneck": "global_memory_access"
    },
    "suggestions": [
        "Consider shared memory to reduce global memory traffic",
        "Memory access pattern is not coalesced"
    ]
}
```

The LLM saw:
- Low memory bandwidth utilization
- Explicit suggestion to use shared memory
- Top programs that used `threadgroup` keyword

It learned: Add `threadgroup float shared_memory[...]` → performance improved → continue optimizing.

### What This Demonstrates

- **Hardware-aware optimization**: Discovered GPU-specific patterns (shared memory, vectorization)
- **Artifact-driven learning**: Profiling data guided mutations toward hardware bottlenecks
- **Language-agnostic**: Works on Metal shaders, not just Python

## Example 4: LLM Prompt Optimization - Meta-Evolution

**Location**: `examples/llm_prompt_optimization/`

**Challenge**: Evolve prompts for HotpotQA (multi-hop question answering).

**Initial Prompt**:
```
Answer the question based on the context.

Context: {context}
Question: {question}
Answer:
```

**Accuracy**: 62%

**Evolved Prompt (Generation 80)**:
```
You are an expert researcher tasked with answering complex questions requiring multi-hop reasoning.

Context: {context}

Question: {question}

Instructions:
1. Carefully read the context and identify all relevant facts
2. Break down the question into sub-questions if needed
3. Trace the logical connections between facts
4. Synthesize a concise, evidence-based answer
5. If uncertain, state your confidence level

Reasoning steps:
[Analyze the problem step-by-step]

Final Answer:
```

**Accuracy**: 85% (+23% improvement!)

### The Meta-Loop

This is evolution evolving **itself**:

```
LLM1 (Evolver) → Generates prompt variants
    ↓
Evaluator → Tests prompts on HotpotQA
    ↓
Database → Stores best prompts (MAP-Elites)
    ↓
LLM1 (Evolver) → Mutates best prompts
    ↓
(repeat)
```

The same LLM that evolves code is now evolving its own prompts.

### What This Demonstrates

- **Self-improvement**: AI system improving its own reasoning
- **Meta-learning**: Learning to learn
- **Transferability**: Evolved prompts work on held-out test sets

## Configuration Patterns Across Examples

### Function Minimization

```yaml
evaluator:
  cascade_thresholds: [0.0, 0.3, 0.6]
  cascade_timeouts: [1, 5, 30]

database:
  feature_dimensions: [complexity, diversity]
  population_size: 500
  num_islands: 3

llm:
  models:
    - provider: openai
      model: gpt-4-turbo
      weight: 2
    - provider: anthropic
      model: claude-3-opus
      weight: 1
```

### Circle Packing

```yaml
evaluator:
  cascade_thresholds: [2.0, 1.5, 1.0]  # Radius thresholds
  cascade_timeouts: [5, 30, 120]

database:
  feature_dimensions: [complexity, symmetry_score]  # Custom feature!
  population_size: 200
  num_islands: 5  # More islands for exploration

llm:
  temperature: 0.9  # Higher temperature for creative geometry
```

### GPU Kernel Optimization

```yaml
evaluator:
  cascade_thresholds: [0.5, 1.0, 1.5]  # GFLOPS thresholds
  cascade_timeouts: [2, 10, 60]
  use_llm_feedback: true  # Enable code review

database:
  feature_dimensions: [memory_bandwidth, compute_utilization]  # Hardware metrics!
  population_size: 300

llm:
  models:
    - provider: openai
      model: gpt-4-turbo
      weight: 3  # GPT-4 is best at low-level optimization
```

## Common Success Patterns

Across all examples, successful evolution exhibits:

**1. Rapid early improvement**: First 50-100 generations see major gains (low-hanging fruit)

**2. Plateau and breakthrough**: Stagnation periods followed by sudden jumps (discovering new strategies)

**3. Incremental refinement**: Final 20% of improvement comes from last 80% of generations (polishing)

**4. Island diversity**: Different islands explore different approaches, then converge via migration

**5. Artifact-driven learning**: Error messages and profiling data directly guide mutations

## Next Chapter

We've seen OpenEvolve in action across domains. But how do these applications actually work under the hood? What does the code look like end-to-end?

Next, we'll do a **deep dive into implementation**, walking through critical code paths:
- Full evolution loop
- Database operations
- Prompt building
- Process parallelism

We'll connect the high-level architecture to the actual Python code.

---

*"In theory, theory and practice are the same. In practice, they are not." — Attributed to Yogi Berra*

These examples show the practice: Evolution discovering algorithms, matching benchmarks, optimizing hardware, and even improving itself. The theory works.
