# LLM Integration: The Mutation Engine

## The Role of LLMs in Evolution

In biological evolution, mutation happens randomly—cosmic rays flip bits in DNA, replication errors introduce changes, radiation damages chromosomes. Random, blind, undirected.

In OpenEvolve, mutation is **intelligent**. The LLM is the mutation operator, but instead of random bit flips, it receives:
- Current program (the genome)
- Fitness scores (natural selection feedback)
- Top performers (the hall of fame)
- Error messages (why previous mutations failed)
- Task description (the environment)

And it generates a **directed mutation**—not random, but informed by context.

This is the key insight that makes evolutionary coding work: **LLMs can mutate code intelligently when given the right context.**

## The Ensemble Strategy

No single LLM is perfect. Each has strengths:
- GPT-4: Excellent at following complex instructions, strong general coding
- Claude: Great at reasoning through edge cases, clear explanations
- Gemini: Strong at mathematical optimization, fast generation

Why choose? Use **all of them** via weighted ensemble:

```python
models:
  - provider: openai
    model: gpt-4-turbo
    weight: 3        # 50% of mutations
  - provider: anthropic
    model: claude-3-opus
    weight: 2        # 33% of mutations
  - provider: google
    model: gemini-pro
    weight: 1        # 17% of mutations
```

Each iteration, sample a model based on weights. Over 1000 iterations:
- ~500 mutations from GPT-4
- ~330 mutations from Claude
- ~170 mutations from Gemini

This creates **diversity in mutation strategies**. GPT-4 might favor elegant refactorings, Claude might introduce defensive checks, Gemini might optimize math operations. The population benefits from all perspectives.

## Implementation

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/llm/ensemble.py`

```python
from typing import List, Optional
from random import Random
import hashlib

class LLMEnsemble:
    def __init__(self, models_config: List[dict], random_seed: Optional[int] = None):
        """Initialize ensemble with multiple LLMs."""

        # Create LLM instances
        self.models = []
        for cfg in models_config:
            if cfg["provider"] == "openai":
                from .openai import OpenAILLM
                self.models.append(OpenAILLM(cfg))
            elif cfg["provider"] == "anthropic":
                from .anthropic import AnthropicLLM
                self.models.append(AnthropicLLM(cfg))
            elif cfg["provider"] == "google":
                from .google import GoogleLLM
                self.models.append(GoogleLLM(cfg))
            else:
                raise ValueError(f"Unknown provider: {cfg['provider']}")

        # Extract and normalize weights
        self.weights = [m.weight for m in models_config]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Deterministic random state for reproducibility
        if random_seed is not None:
            # Create ensemble-specific seed
            seed_bytes = str(random_seed).encode("utf-8") + b"ensemble"
            ensemble_seed = int(hashlib.md5(seed_bytes).hexdigest()[:8], 16)
            self.random_state = Random(ensemble_seed)
        else:
            self.random_state = Random()

    def _sample_model(self) -> "BaseLLM":
        """Sample a model based on weights (deterministic if seeded)."""
        return self.random_state.choices(self.models, weights=self.weights)[0]

    async def generate(self, prompt: dict, **kwargs) -> str:
        """Generate code using a sampled model."""

        # Sample model
        model = self._sample_model()
        logger.debug(f"Sampled model: {model.name}")

        try:
            # Generate with primary model
            response = await model.generate(prompt, **kwargs)
            return response

        except Exception as e:
            logger.warning(f"Model {model.name} failed: {e}")

            # Fallback to next model
            for fallback_model in self.models:
                if fallback_model != model:
                    try:
                        logger.info(f"Falling back to {fallback_model.name}")
                        response = await fallback_model.generate(prompt, **kwargs)
                        return response
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {fallback_model.name} also failed: {fallback_error}")
                        continue

            # All models failed
            raise RuntimeError(f"All models in ensemble failed. Last error: {e}")
```

### Key Features

**1. Weighted Sampling**

```python
self.random_state.choices(self.models, weights=self.weights)
```

Uses Python's `random.choices()` with weights. If weights are `[0.5, 0.33, 0.17]`, models are sampled with those probabilities.

**2. Deterministic Seeding**

```python
ensemble_seed = int(hashlib.md5(seed_bytes).hexdigest()[:8], 16)
self.random_state = Random(ensemble_seed)
```

Creates a **deterministic** random state from the global seed. Same seed = same sequence of model selections.

Why deterministic? **Reproducibility**. Run with seed 42 today and in six months—same models get sampled in same order.

**3. Automatic Fallback**

```python
try:
    response = await model.generate(prompt)
except Exception:
    # Try next model
    for fallback_model in self.models:
        ...
```

If GPT-4 API is down, automatically tries Claude, then Gemini. Evolution doesn't stop due to transient failures.

## The OpenAI-Compatible Interface

Most LLM providers now support OpenAI-compatible APIs:
- OpenAI: Native
- Anthropic: Via `anthropic` SDK (messages API)
- Google: Via `google-generativeai` SDK
- Azure OpenAI: Native
- Local models (Ollama, LM Studio): OpenAI-compatible endpoint

OpenEvolve uses the **OpenAI chat completion format** internally:

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/llm/openai.py`

```python
import openai
from typing import Optional

class OpenAILLM:
    def __init__(self, config: dict):
        self.name = config.get("name", config["model"])
        self.model = config["model"]
        self.weight = config.get("weight", 1.0)

        # API configuration
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)

        # Create client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def generate(self, prompt: dict, **kwargs) -> str:
        """Generate code from prompt.

        Args:
            prompt: Dict with "system" and "user" keys

        Returns:
            Generated code (extracted from response)
        """

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]

        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **kwargs
        )

        # Extract code
        content = response.choices[0].message.content

        # Parse code block if present
        code = self._extract_code_block(content)

        return code

    def _extract_code_block(self, content: str) -> str:
        """Extract code from markdown code blocks."""

        # Look for ```python ... ```
        import re
        match = re.search(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)

        if match:
            return match.group(1)

        # No code block, return as-is
        return content
```

### Code Extraction

LLMs often wrap code in markdown:

````
Here's an improved version:

```python
def search():
    # improved code
    ...
```

I optimized the loop structure.
````

OpenEvolve extracts just the code:

```python
def search():
    # improved code
    ...
```

This prevents explanatory text from being evaluated as code.

## The Prompt Structure

Prompts are the **genome instructions** for mutation. They package all context needed for intelligent evolution.

**File**: `/home/sangmank/projects/synaptic_drift/openevolve/openevolve/prompt/builder.py`

### Components

A complete prompt consists of:

**1. System Message**
```
You are an expert software engineer specializing in algorithm optimization.
Your task is to evolve and improve code through iterative mutations.

Guidelines:
- Preserve the function signature
- Improve performance while maintaining correctness
- Explain your changes briefly
```

**2. Task Description**
```
Task: Minimize the Rastrigin function over the domain [-5, 5] × [-5, 5].

The function is highly multimodal with many local minima.
The global minimum is at (0, 0) with value 0.

You are evolving a search algorithm that finds the minimum.
```

**3. Current Program (Parent)**
```python
# Current program (fitness: 0.65)
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    temperature = 1.0
    current = random_point(bounds)

    for i in range(iterations):
        neighbor = mutate(current, step=temperature)
        if better(neighbor, current, temperature):
            current = neighbor
        temperature *= 0.99

    return current
```

**4. Evolution History**
```
Evolution history:
- Iteration 95 (fitness 0.50): Added simulated annealing
- Iteration 98 (fitness 0.58): Tuned temperature schedule
- Iteration 99 (fitness 0.65): Added adaptive step size
```

**5. Top Programs (Inspiration)**
```python
# Top program #1 (fitness: 0.82) - Different approach
def search_algorithm(...):
    # Uses multi-start strategy
    best = None
    for restart in range(5):
        candidate = local_search()
        if better(candidate, best):
            best = candidate
    return best

# Top program #2 (fitness: 0.75) - Another variation
def search_algorithm(...):
    # Uses adaptive restart
    ...
```

**6. Artifacts (Feedback)**
```
Previous attempt artifacts:
- stderr: "RuntimeWarning: overflow in exp"
- profiling: 80% time spent in evaluate_function()
- failed_test_case: large_input_test
```

**7. Instruction**
```
Create an improved version of the current program.
Focus on:
1. Fixing any errors (see artifacts above)
2. Improving fitness (current: 0.65, best: 0.82)
3. Exploring new approaches (consider ideas from top programs)

Return only the code, no explanations.
```

### Template Building

```python
def build_prompt(
    self,
    current_program: Program,
    metrics: dict,
    evolution_history: List[Program],
    top_programs: List[Program],
    artifacts: dict,
    task_description: str,
    system_message: Optional[str] = None
) -> dict:
    """Build complete prompt for LLM."""

    # Load template
    template = self._load_template("default_template.txt")

    # Apply stochastic variations
    template = self._apply_variations(template)

    # Fill in sections
    system = system_message or self._default_system_message()

    user = template.format(
        task_description=task_description,
        current_code=current_program.code,
        current_fitness=self._format_fitness(metrics),
        evolution_history=self._format_history(evolution_history),
        top_programs=self._format_top_programs(top_programs),
        artifacts=self._format_artifacts(artifacts),
        instruction=self._generate_instruction(metrics, top_programs)
    )

    return {"system": system, "user": user}
```

### Stochastic Variations

To prevent mode collapse (LLM returning similar mutations repeatedly), prompts include **random variations**:

```python
def _apply_variations(self, template: str) -> str:
    """Apply stochastic variations to template."""

    variations = {
        "{{greeting}}": [
            "Let's improve this code:",
            "Time to optimize:",
            "Evolving the algorithm:",
            ""  # Sometimes no greeting
        ],
        "{{instruction_style}}": [
            "Create an improved version.",
            "Generate an enhanced implementation.",
            "Evolve a better solution.",
            "Mutate the current code for higher fitness."
        ],
        "{{encouragement}}": [
            "You're making progress!",
            "Keep pushing for better solutions.",
            "",
            "Current approach is promising."
        ]
    }

    for placeholder, options in variations.items():
        if placeholder in template:
            choice = random.choice(options)
            template = template.replace(placeholder, choice)

    return template
```

Same semantic content, different surface forms. Prevents the LLM from getting stuck in repetitive patterns.

### Fitness vs. Features in Prompts

**Critical**: Prompts must distinguish fitness (what we optimize) from features (diversity dimensions):

```python
def _format_fitness(self, metrics: dict) -> str:
    """Format fitness score (excluding feature dimensions)."""

    fitness_metrics = {
        k: v for k, v in metrics.items()
        if k not in self.feature_dimensions
    }

    fitness = sum(fitness_metrics.values()) / len(fitness_metrics)

    return f"Fitness: {fitness:.2f} (components: {fitness_metrics})"
```

If `feature_dimensions = ["complexity"]` and `metrics = {"score": 0.8, "correctness": 1.0, "complexity": 150}`:

```
Fitness: 0.90 (components: {"score": 0.8, "correctness": 1.0})
Complexity: 150 (for diversity tracking)
```

The LLM is told: "Your fitness is 0.90, and your complexity is 150." Not: "Your score is 0.8+1.0+150=151.8 (nonsense!)."

## Advanced: Evolution History Formatting

How much history should we show? Too little = LLM lacks context. Too much = context window overflow.

OpenEvolve uses a **sliding window** with **compression**:

```python
def _format_history(self, programs: List[Program], max_entries: int = 5) -> str:
    """Format recent evolution history."""

    # Sort by iteration (most recent last)
    programs = sorted(programs, key=lambda p: p.iteration)

    # Take last N
    recent = programs[-max_entries:]

    # Format each entry
    entries = []
    for p in recent:
        fitness = get_fitness_score(p.metrics, exclude=self.feature_dimensions)

        # Summarize changes if available
        changes = p.metadata.get("changes_summary", "Modified algorithm")

        entries.append(f"- Iteration {p.iteration} (fitness {fitness:.2f}): {changes}")

    return "\n".join(entries)
```

Example output:
```
Evolution history:
- Iteration 145 (fitness 0.50): Added simulated annealing
- Iteration 147 (fitness 0.58): Tuned temperature schedule
- Iteration 150 (fitness 0.62): Increased iteration count
- Iteration 152 (fitness 0.65): Added adaptive step size
- Iteration 155 (fitness 0.68): Implemented restart strategy
```

The LLM sees the **trajectory** of improvement, not just the current state.

### Change Detection

How does OpenEvolve know what changed? **Diff analysis**:

```python
def _detect_changes(self, parent_code: str, child_code: str) -> str:
    """Detect high-level changes between parent and child."""

    import difflib

    # Compute diff
    diff = difflib.unified_diff(
        parent_code.splitlines(),
        child_code.splitlines(),
        lineterm=""
    )

    # Analyze diff for semantic changes
    diff_lines = list(diff)

    # Simple heuristics
    added_lines = [l for l in diff_lines if l.startswith("+")]
    removed_lines = [l for l in diff_lines if l.startswith("-")]

    if "def " in " ".join(added_lines):
        return "Added new function"
    elif "for " in " ".join(added_lines) or "while " in " ".join(added_lines):
        return "Modified loop structure"
    elif "import " in " ".join(added_lines):
        return "Added new import/dependency"
    elif len(added_lines) > len(removed_lines) * 2:
        return "Significant expansion of code"
    elif len(removed_lines) > len(added_lines) * 2:
        return "Simplified code"
    else:
        return "Refined implementation"
```

Not perfect, but gives the LLM a high-level sense of what's been tried.

## Configuration

```yaml
llm:
  # Ensemble configuration
  models:
    - provider: openai
      name: GPT-4-Turbo
      model: gpt-4-turbo-preview
      api_key: ${OPENAI_API_KEY}  # Environment variable
      weight: 3
      temperature: 0.7
      max_tokens: 4096

    - provider: anthropic
      name: Claude-3-Opus
      model: claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}
      weight: 2
      temperature: 0.7
      max_tokens: 4096

    - provider: google
      name: Gemini-Pro
      model: gemini-1.5-pro
      api_key: ${GOOGLE_API_KEY}
      weight: 1
      temperature: 0.7
      max_tokens: 4096

  # Prompt configuration
  prompt:
    template_dir: "./prompts"
    template_name: "default_template.txt"
    system_message: "You are an expert algorithm designer."

    # Stochastic variations
    use_variations: true
    variation_seed: null  # null = random, int = deterministic

    # History settings
    max_history_entries: 5
    include_top_programs: 3
    include_artifacts: true

  # Reproducibility
  random_seed: 42
```

## Real-World Example: Prompt Evolution

One fascinating capability: **Prompts can evolve prompts**.

**Example**: `examples/llm_prompt_optimization/`

The system optimizes prompts for a question-answering task (HotpotQA):

**Initial Prompt** (Generation 0):
```
Answer the question: {question}
Use the context: {context}
```

**Evolved Prompt** (Generation 50):
```
You are a meticulous researcher. Analyze the following context carefully:
{context}

Question: {question}

Instructions:
1. Identify key facts relevant to the question
2. Reason through the logical connections
3. Provide a concise, evidence-based answer

Answer:
```

**Result**: +23% accuracy improvement on HotpotQA benchmark.

**How?** The meta-level evolution:
1. LLM generates variant prompts
2. Evaluator tests each prompt on QA dataset
3. Better prompts survive (MAP-Elites)
4. LLM mutates the best prompts
5. Repeat

The LLM learns to write better prompts for *itself*.

## The Mutation-Selection Loop

Let's trace one full cycle:

**Step 1: Sample Parent**
```python
parent = database.sample_parent(island_id=0)
# Code: def search(): ...
# Fitness: 0.65
```

**Step 2: Sample Inspirations**
```python
top_programs = database.sample_top_programs(island_id=0, n=3)
# Programs with fitness: [0.82, 0.75, 0.68]
```

**Step 3: Build Prompt**
```python
prompt = prompt_builder.build_prompt(
    current_program=parent,
    metrics=parent.metrics,
    evolution_history=database.get_recent_programs(parent_id=parent.id, n=5),
    top_programs=top_programs,
    artifacts=parent.artifacts,
    task_description=config.task_description
)
```

**Step 4: Sample Model**
```python
model = llm_ensemble._sample_model()
# Sampled: GPT-4-Turbo (50% probability)
```

**Step 5: Generate Mutation**
```python
child_code = await llm_ensemble.generate(prompt)
# Generated:
# def search():
#     # Hybrid of parent + inspiration #1
#     temperature = 1.0
#     candidates = []
#     for restart in range(5):
#         current = random_point()
#         for i in range(200):
#             neighbor = mutate(current, step=temperature)
#             if accept(neighbor, current, temperature):
#                 current = neighbor
#             temperature *= 0.98
#         candidates.append(current)
#     return min(candidates, key=evaluate)
```

**Step 6: Evaluate**
```python
result = await evaluator.evaluate(child_code)
# Fitness: 0.78 (improvement!)
```

**Step 7: Add to Database**
```python
child_program = Program(
    code=child_code,
    parent_id=parent.id,
    metrics=result.metrics,
    artifacts=result.artifacts
)
database.add(child_program, island_id=0)
# Result: Replaces existing program in cell [2, 5] (complexity=2, diversity=5)
```

**Step 8: Repeat**

The mutation (child) becomes a potential parent for future iterations. If it's fit, it will be sampled more often (via top_programs). If not, it might still occupy a unique niche in MAP-Elites.

## Challenges and Solutions

### Challenge 1: Context Window Limits

Modern LLMs have context windows of 8K-128K tokens. But evolution history + top programs + artifacts can exceed this.

**Solution**: Prioritize information
1. Current program (always include)
2. Task description (always include)
3. Recent artifacts (last 3 iterations)
4. Evolution history (last 5 entries, summarized)
5. Top programs (top 3, code only, no explanations)

Estimated tokens: ~4K-6K, well within limits.

### Challenge 2: Mode Collapse

LLMs can get stuck generating similar mutations.

**Solution**: Stochastic variations + model ensemble + temperature tuning

Different models explore different mutation strategies. Variation in prompt phrasing prevents repetition.

### Challenge 3: Code Extraction Reliability

LLMs sometimes return explanations instead of code.

**Solution**: Clear instructions + regex extraction + validation

Prompt explicitly: "Return ONLY the code, no explanations."
Regex extracts from ```python blocks.
Validation: Does it have the required function signature?

### Challenge 4: API Costs

Thousands of LLM calls = $$$

**Solution**: Cascade evaluation + caching + smaller models for refinement

Cascade reduces calls (fail fast on broken code).
Cache responses for identical prompts (rare but happens).
Use GPT-4 for discovery, GPT-3.5 for refinement (cheaper).

## Next Chapter

We've seen how LLMs mutate code intelligently. But how does this all come together in practice? What do real-world applications look like?

Next, we'll explore **example use cases**:
- Function minimization (discovering algorithms)
- Circle packing (mathematical optimization)
- GPU kernel optimization (hardware-aware code)
- LLM prompt optimization (meta-evolution)

We'll see the architecture in action, from initial programs to evolved solutions.

---

*"The whole is greater than the sum of its parts." — Aristotle*

An LLM alone can write code. Evolution alone can optimize. But LLM + Evolution + Feedback creates an autonomous researcher that discovers solutions humans might never find.
