# Bug Report: tl-evaluate Auto-Discovery Failure

**Date:** 2025-10-29
**Reporter:** Nicolas Lazaro
**Severity:** High (blocks evaluation workflow without explicit seed specification)

---

## Summary

The `tl-evaluate` CLI fails when auto-discovering seeds for evaluation due to a type mismatch. The code expects `get_checkpoint_rankings()` to return a dictionary but it returns a list, causing an `AttributeError` when attempting to call `.keys()`.

---

## Error Message

```
AttributeError: 'list' object has no attribute 'keys'
```

**Full Stack Trace:**

```
Traceback (most recent call last):
  File "/workspace/pub-usa/.venv/bin/tl-evaluate", line 10, in <module>
    sys.exit(main())
  File "/workspace/pub-usa/.venv/lib/python3.12/site-packages/click/core.py", line 1462, in __call__
    return self.main(*args, **kwargs)
  File "/workspace/pub-usa/.venv/lib/python3.12/site-packages/click/core.py", line 1383, in main
    rv = self.invoke(ctx)
  File "/workspace/pub-usa/.venv/lib/python3.12/site-packages/click/core.py", line 1246, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/workspace/pub-usa/.venv/lib/python3.12/site-packages/click/core.py", line 814, in invoke
    return callback(*args, **kwargs)
  File ".../transfer_learning_publication/evaluation_cli/__main__.py", line 117, in main
    orchestrator.evaluate_models(models_to_evaluate, seeds_list)
  File ".../transfer_learning_publication/evaluation_cli/orchestrator.py", line 96, in evaluate_models
    seeds_to_eval = self._get_seeds_for_model(model_name, seeds_list)
  File ".../transfer_learning_publication/evaluation_cli/orchestrator.py", line 129, in _get_seeds_for_model
    seeds_to_eval = list(rankings.keys())
AttributeError: 'list' object has no attribute 'keys'
```

---

## Steps to Reproduce

1. Train models with `tl-train` (creates checkpoints with seed 42)
2. Create evaluation config:
   ```yaml
   # configs/evaluation/fold_1_eval.yaml
   data:
     base_path: ../CARAVAN_TRANSFORMED_NO_LOG_eval
     gauge_ids_file: configs/basin_ids_files/test_fold_1.txt
     pipeline_path: ../CARAVAN_TRANSFORMED_NO_LOG_eval/ts_pipeline.joblib
   ```
3. Run evaluation WITHOUT explicit seeds:
   ```bash
   uv run tl-evaluate experiments/small_models_275k_params_12fold.yaml \
     --eval-config configs/evaluation/fold_1_eval.yaml \
     --models ealstm_275k_fold_1,mamba_275k_fold_1
   ```
4. Observe error at auto-discovery step

---

## Root Cause

**Location:** `src/transfer_learning_publication/evaluation_cli/orchestrator.py:129`

**Problematic Code:**

```python
def _get_seeds_for_model(self, model_name: str, seeds_list: list[int] | None) -> list[int]:
    if seeds_list:
        # Use explicitly provided seeds
        return seeds_list
    else:
        # Auto-discover seeds from checkpoints
        rankings = self.discovery.get_checkpoint_rankings(model_name, stage=self.checkpoint_source)
        if not rankings:
            return []

        seeds_to_eval = list(rankings.keys())  # ❌ BUG: rankings is a list, not a dict
        logger.info(f"Auto-discovered seeds for {model_name}: {seeds_to_eval}")
        return seeds_to_eval
```

**Expected Return Type (from API):**

Looking at `checkpoint_utils/discovery.py:325-327`:

```python
def get_checkpoint_rankings(
    self, model_name: str, seeds: list[int] | None = None, stage: str = "training", timestamp: str | None = None
) -> list[Checkpoint]:  # ← Returns a LIST of Checkpoint objects
    """
    Get all checkpoints ranked by performance.

    Returns:
        List of checkpoints sorted by val_loss (best first), with rank and percentile
    """
```

**Analysis:**

The `get_checkpoint_rankings()` method returns `list[Checkpoint]`, but the evaluation orchestrator expects it to return a dictionary with seeds as keys. This is either:

1. A recent API change in `checkpoint_utils` that wasn't propagated to `evaluation_cli`, OR
2. A misunderstanding of the intended API during development

---

## Workaround

Explicitly specify seeds using the `--seeds` flag:

```bash
uv run tl-evaluate experiments/small_models_275k_params_12fold.yaml \
  --eval-config configs/evaluation/fold_1_eval.yaml \
  --models ealstm_275k_fold_1,mamba_275k_fold_1 \
  --seeds 42
```

This bypasses the auto-discovery code path entirely.

---

## Proposed Fix

**Option 1: Extract seeds from list of Checkpoint objects**

```python
def _get_seeds_for_model(self, model_name: str, seeds_list: list[int] | None) -> list[int]:
    if seeds_list:
        return seeds_list
    else:
        # Auto-discover seeds from checkpoints
        rankings = self.discovery.get_checkpoint_rankings(model_name, stage=self.checkpoint_source)
        if not rankings:
            return []

        # Extract unique seeds from Checkpoint objects
        seeds_to_eval = sorted(set(ckpt.seed for ckpt in rankings))
        logger.info(f"Auto-discovered seeds for {model_name}: {seeds_to_eval}")
        return seeds_to_eval
```

**Option 2: Use a different discovery method**

Check if there's a method that returns seed-keyed data:

```python
# Look for alternative methods in CheckpointDiscovery
# e.g., get_seeds_for_model(), list_runs(), etc.
```

---

## Impact

**User Impact:**
- **High:** Evaluation workflow is broken without explicit seed specification
- Users must manually discover and specify seeds for every model
- Documentation suggests auto-discovery should work (CLI guide line 681)

**Affected Commands:**
- `tl-evaluate` without `--seeds` flag
- Any workflow relying on automatic seed discovery

**Versions Affected:**
- Current version in `/Users/nicolaslazaro/Desktop/work/transfer-learning-publication`
- Likely affects recent commits (check git log for `checkpoint_utils/discovery.py`)

---

## Related Files

**Primary:**
- `src/transfer_learning_publication/evaluation_cli/orchestrator.py` (line 129)
- `src/transfer_learning_publication/checkpoint_utils/discovery.py` (lines 325-365)

**Documentation:**
- `docs/cli_guide.md` (line 681 - mentions auto-discovery feature)

---

## Testing Recommendations

1. **Unit test for auto-discovery:**
   ```python
   def test_auto_discover_seeds():
       """Test that _get_seeds_for_model works without explicit seeds."""
       orchestrator = EvaluationOrchestrator(...)
       seeds = orchestrator._get_seeds_for_model("test_model", seeds_list=None)
       assert isinstance(seeds, list)
       assert all(isinstance(s, int) for s in seeds)
   ```

2. **Integration test:**
   - Create mock checkpoints with multiple seeds
   - Run `tl-evaluate` without `--seeds` flag
   - Verify correct seeds are discovered and used

3. **Regression test:**
   - Test both explicit and auto-discovery paths
   - Ensure backward compatibility

---

## Next Steps

1. ✅ Document bug (this file)
2. ⬜ Submit issue to GitHub repository
3. ⬜ Implement and test proposed fix
4. ⬜ Update CLI documentation if API changed intentionally
5. ⬜ Add regression tests for auto-discovery

---

## Additional Notes

- This bug was discovered during 12-fold cross-validation evaluation setup
- All 12 folds have consistent checkpoints (seed=42)
- Workaround is sufficient for current needs but should be fixed upstream
