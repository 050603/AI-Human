# AI-Human Classroom Core Validation

Notebook-level prototype for classroom transcript evaluation with uncertainty routing.

The prototype focuses on the text-only validation loop:

1. Load structured classroom transcript JSON.
2. Slice transcript into fixed windows with overlap.
3. Evaluate each slice multiple times with a configurable LLM provider.
4. Cluster generated rationales with embeddings.
5. Compute score entropy and semantic entropy.
6. Route low-uncertainty slices to automatic acceptance and high-uncertainty slices to human review.

The code is designed for offline intranet testing first, while keeping provider interfaces compatible with later API-based models.

## Quick start

```bash
python scripts/run_core_validation.py \
  --transcript data/sample/lesson_001.json \
  --config configs/local_mock.yaml \
  --output outputs/core_validation_result.json
```

## Runtime modes

- `local`: uses local providers. The included `mock_local` LLM and `hashing` embedder make the pipeline runnable without downloading models.
- `api`: interface-compatible mode for later OpenAI-compatible chat and embedding APIs.

For production local deployment, replace the mock LLM with a vLLM/SGLang/Transformers provider behind the same interface.

## Output

Each slice output contains:

- slice id and time range
- Monte Carlo samples
- score distribution
- majority score
- semantic clusters
- score entropy
- semantic entropy
- routing decision: `auto_accept` or `human_review`
