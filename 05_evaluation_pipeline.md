# The Evaluation Pipeline: Testing, Timing, and Learning from Failure

## The Central Challenge

Evolution without evaluation is just random code generation. The evaluator is where **natural selection** happens—where programs prove their worth or fail trying.

But evaluation is expensive:
- Running a full benchmark suite might take minutes
- Most mutations (90%+) are broken or worse than parents
- We're generating thousands of programs per experiment

How do we efficiently test programs while gathering rich feedback for the next generation?

OpenEvolve's answer: **Cascade evaluation** with **artifact collection**.

## The Cascade Pattern

Instead of running one expensive test, run a sequence of increasingly challenging tests:

```
Stage 1: Smoke Test (1 second)
    ↓ (passes threshold?)
Stage 2: Basic Validation (10 seconds)
    ↓ (passes threshold?)
Stage 3: Comprehensive Evaluation (60 seconds)
    ↓
Final Score
```

Programs that fail early tests **don't waste time** on later tests.

Think of it like a university admissions process:
1. **Stage 1**: Does the application have required documents? (reject 50% instantly)
2. **Stage 2**: Does GPA meet minimum threshold? (reject another 30%)
3. **Stage 3**: Full committee review of essays, recommendations, etc. (20% remain)

Only serious candidates get expensive review.

## The Implementation

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/evaluator.py`

### Core Structure

```python
class Evaluator:
    def __init__(self, config, evaluation_file_path):
        self.config = config

        # Timeouts for each stage
        self.timeouts = config.cascade_timeouts or [1, 10, 60]  # seconds

        # Score thresholds to advance
        self.thresholds = config.cascade_thresholds or [0.0, 0.3, 0.6]

        # Load the evaluation function
        self.evaluator_module = self._load_evaluator(evaluation_file_path)

        # Extract stage evaluators
        self.stage_evaluators = [
            getattr(self.evaluator_module, f"evaluate_stage{i+1}", None)
            for i in range(3)
        ]

    async def evaluate(self, program_path: str) -> EvaluationResult:
        """Evaluate a program using cascade strategy."""
        if self.config.use_cascade:
            return await self._cascade_evaluate(program_path)
        else:
            return await self._simple_evaluate(program_path)
```

### Cascade Evaluation

```python
async def _cascade_evaluate(self, program_path: str) -> EvaluationResult:
    """Run cascade evaluation with early stopping."""

    # Stage 1: Quick smoke test
    stage1_result = await self._run_stage(
        stage=1,
        program_path=program_path,
        timeout=self.timeouts[0]
    )

    # Check if passes threshold to continue
    fitness = get_fitness_score(
        stage1_result.metrics,
        exclude=self.config.feature_dimensions
    )

    if fitness < self.thresholds[0]:
        # Failed Stage 1, return immediately
        logger.info(f"Program failed Stage 1 (score {fitness} < {self.thresholds[0]})")
        return stage1_result

    # Stage 2: Basic validation
    stage2_result = await self._run_stage(
        stage=2,
        program_path=program_path,
        timeout=self.timeouts[1]
    )

    # Merge Stage 1 and Stage 2 results
    merged_result = EvaluationResult(
        metrics={**stage1_result.metrics, **stage2_result.metrics},
        artifacts={**stage1_result.artifacts, **stage2_result.artifacts}
    )

    fitness = get_fitness_score(merged_result.metrics, exclude=self.config.feature_dimensions)

    if fitness < self.thresholds[1]:
        logger.info(f"Program failed Stage 2 (score {fitness} < {self.thresholds[1]})")
        return merged_result

    # Stage 3: Comprehensive evaluation
    stage3_result = await self._run_stage(
        stage=3,
        program_path=program_path,
        timeout=self.timeouts[2]
    )

    # Merge all stages
    final_result = EvaluationResult(
        metrics={**merged_result.metrics, **stage3_result.metrics},
        artifacts={**merged_result.artifacts, **stage3_result.artifacts}
    )

    logger.info(f"Program passed all stages (final score: {get_fitness_score(final_result.metrics)})")
    return final_result
```

### Running a Single Stage

```python
async def _run_stage(self, stage: int, program_path: str, timeout: float) -> EvaluationResult:
    """Run a single evaluation stage with timeout protection."""

    evaluator_func = self.stage_evaluators[stage - 1]

    if evaluator_func is None:
        # Stage not defined, skip
        return EvaluationResult(metrics={}, artifacts={})

    try:
        # Run evaluation with timeout
        result = await asyncio.wait_for(
            self._run_evaluator_in_subprocess(evaluator_func, program_path),
            timeout=timeout
        )

        return result

    except asyncio.TimeoutError:
        logger.warning(f"Stage {stage} timed out after {timeout}s")
        return EvaluationResult(
            metrics={"score": 0.0, "timeout": True},
            artifacts={"error": f"Timeout after {timeout}s"}
        )

    except Exception as e:
        logger.error(f"Stage {stage} raised exception: {e}")
        return EvaluationResult(
            metrics={"score": 0.0, "error": True},
            artifacts={"error": str(e), "traceback": traceback.format_exc()}
        )
```

### Subprocess Isolation

Each evaluation runs in a **separate process** to prevent:
- Memory leaks accumulating across evaluations
- Segfaults crashing the main process
- Global state pollution

```python
async def _run_evaluator_in_subprocess(self, evaluator_func, program_path):
    """Run evaluator in isolated subprocess."""

    # Create subprocess
    process = await asyncio.create_subprocess_exec(
        sys.executable,  # Python interpreter
        "-c",
        f"""
import sys
import importlib.util

# Load the program module
spec = importlib.util.spec_from_file_location("program", "{program_path}")
program_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program_module)

# Load evaluator
evaluator_module = importlib.import_module("{self.evaluator_module.__name__}")
evaluator_func = evaluator_module.{evaluator_func.__name__}

# Run evaluation
result = evaluator_func(program_module)

# Serialize result to stdout
import json
print(json.dumps(result))
        """,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Wait for completion
    stdout, stderr = await process.communicate()

    # Parse result
    result_dict = json.loads(stdout.decode())

    # Include stderr as artifact if non-empty
    if stderr:
        result_dict.setdefault("artifacts", {})["stderr"] = stderr.decode()

    return EvaluationResult(**result_dict)
```

## Writing Evaluators

Users define evaluation logic in a Python file with stage functions:

**File**: `examples/function_minimization/evaluator.py`

```python
import numpy as np

# The function we're trying to minimize
def objective_function(x, y):
    """Rastrigin function - highly multimodal."""
    A = 10
    return (
        2 * A +
        (x**2 - A * np.cos(2 * np.pi * x)) +
        (y**2 - A * np.cos(2 * np.pi * y))
    )

def evaluate_stage1(program_module):
    """Stage 1: Does it run without crashing?"""
    try:
        # Extract the search algorithm
        search_func = program_module.search_algorithm

        # Run with small iteration count
        x, y, value = search_func(iterations=10, bounds=(-5, 5))

        # Check output validity
        if not (-5 <= x <= 5 and -5 <= y <= 5):
            return {
                "metrics": {"score": 0.0},
                "artifacts": {"error": "Output out of bounds"}
            }

        # Minimal score for passing syntax check
        return {
            "metrics": {"score": 0.1, "stage1_value": value},
            "artifacts": {}
        }

    except Exception as e:
        return {
            "metrics": {"score": 0.0},
            "artifacts": {"error": str(e)}
        }

def evaluate_stage2(program_module):
    """Stage 2: Does it find reasonable solutions?"""
    search_func = program_module.search_algorithm

    # Run multiple trials
    results = []
    for seed in range(5):
        np.random.seed(seed)
        x, y, value = search_func(iterations=100, bounds=(-5, 5))
        results.append(value)

    # Score based on best result
    best_value = min(results)

    # Rastrigin global minimum is 0.0
    # Scale to [0, 1] where 1 is perfect
    score = max(0.0, 1.0 - best_value / 50.0)

    return {
        "metrics": {
            "score": score,
            "stage2_best": best_value,
            "stage2_mean": np.mean(results)
        },
        "artifacts": {
            "all_results": results
        }
    }

def evaluate_stage3(program_module):
    """Stage 3: Comprehensive evaluation with statistics."""
    search_func = program_module.search_algorithm

    results = []
    for seed in range(20):  # More trials
        np.random.seed(seed)
        x, y, value = search_func(iterations=1000, bounds=(-5, 5))  # More iterations
        results.append(value)

    best_value = min(results)
    mean_value = np.mean(results)
    std_value = np.std(results)

    # Score penalizes both mean and variance
    score = max(0.0, 1.0 - mean_value / 30.0 - std_value / 100.0)

    return {
        "metrics": {
            "score": score,
            "stage3_best": best_value,
            "stage3_mean": mean_value,
            "stage3_std": std_value,
            "convergence_rate": (results[0] - results[-1]) / results[0] if results[0] > 0 else 0
        },
        "artifacts": {
            "full_results": results,
            "histogram": np.histogram(results, bins=10).tolist()
        }
    }
```

### The Progression

Notice how each stage gets more demanding:

**Stage 1**: 10 iterations, 1 trial
- Just checks: Does it run? Does it return valid output?
- Passes ~80% of programs (filters out syntax errors, crashes)

**Stage 2**: 100 iterations, 5 trials
- Checks: Does it find decent solutions?
- Passes ~30% of remaining programs

**Stage 3**: 1000 iterations, 20 trials
- Checks: Is it robust and consistent?
- Passes ~10% of remaining programs

Overall: 100 programs → 80 to Stage 2 → 24 to Stage 3 → 2.4 reach end

**Time saved**:
- Without cascade: 100 programs × 60s = 6000s (100 minutes)
- With cascade: 100×1s + 80×10s + 24×60s = 100 + 800 + 1440 = 2340s (39 minutes)

**Speedup**: 2.5x with this particular funnel shape.

## The Artifact Side-Channel

Evaluation returns not just a score, but **rich debugging data**:

```python
return {
    "metrics": {
        "score": 0.85,           # Overall fitness
        "correctness": 1.0,      # Feature: correctness
        "performance": 0.85,     # Feature: performance
        "memory_usage": 1024     # Feature: memory usage (KB)
    },
    "artifacts": {
        "stderr": "Warning: overflow in exp()",
        "failed_tests": ["test_edge_case_3", "test_large_input"],
        "profiling": {
            "time_in_sort": 0.45,
            "time_in_search": 0.30,
            "time_in_other": 0.25
        },
        "llm_feedback": "Variable names are unclear. Consider renaming 'x' to 'candidate'.",
        "visualization": "<base64-encoded-plot>"
    }
}
```

These artifacts get included in the **next generation's prompt**.

### Example: Error Feedback Loop

**Iteration 100**: LLM generates program

```python
def search():
    x = np.random.randn(1000)
    return np.exp(x).max()  # BUG: exp can overflow!
```

**Evaluation**:
```python
{
    "metrics": {"score": 0.0},
    "artifacts": {
        "stderr": "RuntimeWarning: overflow encountered in exp",
        "error": "Result is inf"
    }
}
```

**Iteration 101**: Prompt includes artifacts

```
Your previous program (iteration 100) had issues:
- stderr: "RuntimeWarning: overflow encountered in exp"
- error: "Result is inf"

This happened in the following code:
```python
return np.exp(x).max()
```

Please fix these issues in your next version.
```

**LLM generates**:
```python
def search():
    x = np.random.randn(1000)
    return np.exp(np.clip(x, -100, 100)).max()  # FIX: Clip to prevent overflow
```

**Evaluation**:
```python
{
    "metrics": {"score": 0.75},
    "artifacts": {}  # No errors!
}
```

The LLM **learned** from the error without explicit instruction on how to fix it.

## Advanced: LLM-Based Feedback

OpenEvolve can optionally use an LLM to provide **code review feedback**:

```python
async def _get_llm_feedback(self, program_code: str, metrics: dict) -> str:
    """Use LLM to analyze code quality."""

    prompt = f"""
You are a code reviewer. Analyze this program:

```python
{program_code}
```

Metrics:
- Score: {metrics['score']}
- Performance: {metrics.get('performance', 'N/A')}

Provide brief feedback on:
1. Code clarity
2. Algorithmic efficiency
3. Potential bugs or edge cases

Keep feedback to 2-3 sentences.
"""

    response = await self.llm.generate(prompt)
    return response
```

This feedback is added to artifacts:

```python
artifacts["llm_feedback"] = await self._get_llm_feedback(program_code, metrics)
```

Next generation sees:
```
LLM Reviewer's feedback on your previous attempt:
"The code is clear but inefficient. The nested loop has O(n²) complexity. Consider using a hash table for O(n) lookup. Edge case: empty input array will crash."
```

This creates a **double-loop**:
1. Outer loop: Evolution (LLM mutates code)
2. Inner loop: Review (LLM critiques code)

Both loops improve code quality from different angles.

## Timeout Handling

Programs can hang (infinite loops, deadlocks). Timeouts prevent this:

```python
try:
    result = await asyncio.wait_for(
        evaluate_program(program_path),
        timeout=60.0
    )
except asyncio.TimeoutError:
    result = EvaluationResult(
        metrics={"score": 0.0, "timeout": True},
        artifacts={"error": "Timeout after 60s"}
    )
```

The `timeout=True` metric can be used as a **feature dimension**:

```yaml
feature_dimensions:
  - complexity
  - timeout  # Creates a grid dimension for timeout behavior
```

This creates separate niches for "fast programs" vs "slow programs". Evolution can then improve within each niche—potentially discovering that some slow programs are actually doing useful work (e.g., thorough search).

## Parallel Evaluation

Multiple programs can be evaluated in parallel using `asyncio`:

```python
async def evaluate_batch(self, program_paths: List[str]) -> List[EvaluationResult]:
    """Evaluate multiple programs concurrently."""

    # Create tasks
    tasks = [
        self.evaluate(path)
        for path in program_paths
    ]

    # Run concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            results[i] = EvaluationResult(
                metrics={"score": 0.0},
                artifacts={"error": str(result)}
            )

    return results
```

With cascade evaluation + parallel execution, you can test 100 programs in ~1-2 minutes instead of ~100 minutes.

## Configuration

```yaml
evaluator:
  # Cascade settings
  use_cascade: true
  cascade_timeouts: [1, 10, 60]        # Seconds per stage
  cascade_thresholds: [0.0, 0.3, 0.6]  # Score needed to advance

  # Parallelism
  max_parallel_evaluations: 10

  # Subprocess settings
  subprocess_timeout: 120               # Max time for subprocess
  subprocess_memory_limit: 2048         # MB

  # Optional LLM feedback
  use_llm_feedback: false
  llm_feedback_model: "gpt-4"
  llm_feedback_frequency: 0.1           # 10% of programs get LLM review
```

## Real-World Example: GPU Kernel Optimization

**Task**: Optimize a CUDA kernel for matrix multiplication.

**Stage 1** (1 second): Compile and run on 128×128 matrices
- Filters out: Syntax errors, compilation errors, wrong output shape

**Stage 2** (10 seconds): Run on 512×512 matrices, measure performance
- Filters out: Slow implementations (< 100 GFLOPS)

**Stage 3** (60 seconds): Run on various sizes, measure occupancy, memory bandwidth
- Selects: Kernels with > 500 GFLOPS, > 80% occupancy, good memory coalescing

**Artifacts collected**:
```python
{
    "nvprof_output": "...",                    # Profiler data
    "occupancy": 0.85,                          # Thread utilization
    "memory_efficiency": 0.78,                  # Memory bandwidth utilization
    "register_usage": 32,                       # Registers per thread
    "shared_memory_usage": 12288,               # Bytes of shared memory
    "bank_conflicts": 15,                       # Shared memory bank conflicts
    "warp_divergence": 0.05,                    # Branch divergence
}
```

These artifacts guide the LLM:
```
Your kernel has:
- Occupancy: 85% (good!)
- Memory efficiency: 78% (could be better)
- Bank conflicts: 15 (consider padding shared memory)
- Warp divergence: 5% (consider restructuring conditionals)

Suggestion: Increase block size from 16×16 to 32×32 to improve occupancy.
```

The LLM uses this **hardware-level feedback** to evolve better kernels—discovering optimizations like:
- Memory coalescing patterns
- Shared memory padding
- Register blocking
- Loop unrolling

None of which were explicitly taught!

## The Evaluation-Evolution Feedback Loop

Let's trace how evaluation drives evolution:

**Iteration 1**: Random program
- Evaluation: Crash (syntax error)
- Artifact: "SyntaxError: invalid syntax on line 5"
- Prompt for iteration 2: "Fix syntax error on line 5"

**Iteration 2**: Fixed syntax
- Evaluation: Wrong output (returns None instead of number)
- Artifact: "AssertionError: expected float, got NoneType"
- Prompt for iteration 3: "Ensure function returns a float"

**Iteration 3**: Returns float
- Evaluation: Correct but slow (score 0.2)
- Artifact: "Profiling: 90% time in nested loop"
- Prompt for iteration 4: "Optimize nested loop (currently 90% of runtime)"

**Iteration 4**: Faster algorithm
- Evaluation: Good performance (score 0.7)
- Artifact: None
- Prompt for iteration 5: "Continue improving (current score: 0.7, best: 0.8)"

**Iteration 5**: Near-optimal
- Evaluation: Excellent (score 0.95)
- Artifact: "LLM review: Code is excellent, minor: variable 'tmp' is unclear"
- Prompt for iteration 6: "Polish code quality (rename 'tmp')"

**Iteration 6**: Polished solution
- Evaluation: Perfect (score 1.0)
- Evolution: Add to MAP-Elites grid, mark as champion

Notice the **guided exploration**: Evaluation provides increasingly sophisticated feedback as programs improve, leading evolution from "broken" → "works" → "fast" → "optimal".

## Next Chapter

We've seen how programs are evaluated. But who creates these programs? The LLM ensemble. How do we integrate multiple LLMs, construct effective prompts, and maintain diversity in the mutation process?

Next, we'll explore the **LLM integration layer**: model ensembles, prompt engineering, and the stochasticity that prevents convergence to local optima.

---

*"What gets measured gets improved." — Peter Drucker*

In OpenEvolve, what gets measured, *how* it gets measured, and *how quickly* it gets measured determines the trajectory of evolution. The cascade is where theory meets reality—and reality teaches the theory what works.
