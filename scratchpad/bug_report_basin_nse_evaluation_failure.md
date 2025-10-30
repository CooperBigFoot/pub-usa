# Bug Report: Basin NSE Loss Fails During Model Evaluation

**Date:** 2025-10-30
**Reporter:** Nicolas Lazaro
**Severity:** High (blocks evaluation workflow for models trained with Basin NSE loss)
**Component:** `tl-evaluate` CLI, `BasinNSELoss` class

---

## Summary

Models trained with `basin_nse` loss function cannot be evaluated using `tl-evaluate` because the loss function requires basin identifiers that were present during training, but evaluation is performed on unseen test basins. This causes a `KeyError` during the test step when PyTorch Lightning attempts to log the test loss.

---

## Error Message

```
KeyError: "Basin 'camels_01139000' not found in group_identifiers. This should not happen - please check data consistency."
```

**Full Stack Trace:**

```python
Traceback (most recent call last):
  File "/workspace/pub-usa/.venv/bin/tl-evaluate", line 10, in <module>
    sys.exit(main())
  ...
  File ".../transfer_learning_publication/models/base/base_lit_model.py", line 282, in test_step
    loss = self._compute_loss(y_hat, batch.y.unsqueeze(-1), batch)
  File ".../transfer_learning_publication/models/base/base_lit_model.py", line 212, in _compute_loss
    return self.criterion(y_pred, y_true, batch)
  File ".../transfer_learning_publication/models/losses.py", line 224, in forward
    raise KeyError(
        f"Basin '{basin_id}' not found in group_identifiers. "
        f"This should not happen - please check data consistency."
    )
KeyError: "Basin 'camels_01139000' not found in group_identifiers. This should not happen - please check data consistency."
```

---

## Steps to Reproduce

1. Train a model using Basin NSE loss:

   ```yaml
   # configs/models/ealstm_basin_nse.yaml
   model:
     type: ealstm
     overrides:
       loss_fn: "basin_nse"
       loss_fn_kwargs:
         basin_stats_file: "configs/basin_stats/basin_stats_eval.json"
         epsilon: 0.1

   data:
     base_path: ../CARAVAN_TRANSFORMED
     gauge_ids_file: configs/basin_ids_files/train_fold_1.txt  # 565 training basins
   ```

2. Train the model successfully:

   ```bash
   tl-train experiments/my_experiment.yaml
   ```

3. Attempt to evaluate on test basins:

   ```bash
   tl-evaluate experiments/my_experiment.yaml \
     --eval-config configs/evaluation/fold_1_eval.yaml
   ```

   Where the evaluation config specifies different basins:

   ```yaml
   # configs/evaluation/fold_1_eval.yaml
   data:
     base_path: ../CARAVAN_TRANSFORMED
     gauge_ids_file: configs/basin_ids_files/test_fold_1.txt  # 52 test basins (unseen)
   ```

4. **Error occurs**: The test basins are not in the Basin NSE loss's internal mapping, causing a KeyError

---

## Root Cause

The `BasinNSELoss` class is initialized during model creation with a fixed set of `group_identifiers` (basin IDs) from the training data configuration. This creates an internal mapping `_group_to_idx` that only contains training basins:

**From `losses.py:186-187`:**

```python
# Build group_id → group_idx mapping for batch processing
self._group_to_idx = {gid: idx for idx, gid in enumerate(group_identifiers)}
```

During evaluation with `tl-evaluate`:

1. The checkpoint is loaded with the loss function frozen to training basins
2. A new datamodule is created with **test basins** (intentionally different for proper evaluation)
3. PyTorch Lightning's `test_step()` tries to compute and log the test loss
4. The Basin NSE loss receives a batch with test basin IDs not in `_group_to_idx`
5. **KeyError raised** at `losses.py:224`

This is a **train/test split design conflict**:

- Basin NSE loss is designed for training with multi-basin normalization
- Evaluation is designed to test on unseen basins
- The test loss logging step combines these incompatible requirements

**Key insight:** The test loss is purely a **reporting metric** - it does not influence model performance or evaluation results. The actual predictions are extracted separately and metrics (NSE, KGE, etc.) are computed post-hoc from the parquet outputs.

---

## Impact

- **All models trained with `basin_nse` loss cannot be evaluated** using `tl-evaluate`
- Affects cross-validation workflows (12-fold experiments in this project)
- Requires manual workarounds or skipping evaluation entirely

**Models affected:**

- `ealstm_275k_params_fold_*_all_features.yaml` (24 models)
- `mamba_275k_params_fold_*_all_features.yaml` (24 models)
- Any other models using Basin NSE loss

---

## Proposed Solution

**Use MSE as the evaluation metric for all models, regardless of training loss function.**

### Rationale

1. **Test loss is reporting-only**: The test loss logged during evaluation does not affect:
   - Model predictions
   - Evaluation metrics (NSE, KGE, RMSE, etc.)
   - Checkpoint selection
   - Any downstream analysis

2. **MSE is universally comparable**: Using MSE for evaluation provides:
   - Consistent metric across all models (Basin NSE, Power Loss, standard MSE)
   - Interpretable scale (squared error)
   - No dependency on basin-specific statistics

3. **Separates training objective from evaluation reporting**:
   - Training loss (Basin NSE) optimizes for equal basin weighting
   - Evaluation loss (MSE) provides standard error metric for logging
   - Actual performance metrics computed separately from parquet outputs

### Implementation

The solution involves **swapping the loss function to MSE during evaluation** while preserving the training loss in the checkpoint.

#### Option 1: Modify `BaseLitModel.test_step()` (Minimal Change)

Override the loss function temporarily during test mode:

```python
# In base_lit_model.py

def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
    """Execute test step and collect outputs.

    Note: For evaluation, we always use MSE loss for reporting regardless
    of training loss function. This ensures compatibility with basin_nse
    and other training-specific losses while providing a consistent metric.
    """
    self._validate_batch(batch)
    x_normalized = self._apply_rev_in_normalization(batch.X)
    y_hat = self(x_normalized, batch.static, batch.future)
    y_hat = self._apply_rev_in_denormalization(y_hat)

    # Use MSE for test loss reporting (training loss preserved in checkpoint)
    test_loss = nn.functional.mse_loss(y_hat, batch.y.unsqueeze(-1))
    self.log("test_loss", test_loss, batch_size=batch.batch_size)

    # Rest of the method unchanged...
```

**Pros:**

- Minimal code change (1-2 lines)
- No impact on training behavior
- Backward compatible
- Clear separation of training vs evaluation metrics

**Cons:**

- Hardcodes MSE assumption (but this is acceptable for reporting)

#### Option 2: Evaluation Mode Flag in Model Config (More Flexible)

Add an evaluation-specific loss function option:

```python
# In base_lit_model.py

def __init__(self, config: ModelConfigBase):
    super().__init__()
    self.config = config

    # Training loss function
    self.criterion = self._create_loss_function(
        config.loss_fn,
        config.loss_fn_kwargs
    )

    # Evaluation loss function (defaults to MSE if not specified)
    eval_loss_fn = getattr(config, 'eval_loss_fn', 'mse')
    eval_loss_kwargs = getattr(config, 'eval_loss_fn_kwargs', {})
    self.eval_criterion = self._create_loss_function(
        eval_loss_fn,
        eval_loss_kwargs
    )

def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
    # ... [prediction code] ...

    # Use evaluation-specific loss for reporting
    test_loss = self._compute_loss_with_criterion(
        y_hat,
        batch.y.unsqueeze(-1),
        batch,
        criterion=self.eval_criterion  # Use eval_criterion instead of self.criterion
    )
    self.log("test_loss", test_loss, batch_size=batch.batch_size)
    # ...
```

**Pros:**

- Explicit configuration
- Allows custom evaluation losses if needed
- More flexible for future use cases

**Cons:**

- More complex implementation
- Requires config schema changes
- Possibly over-engineered for current needs

#### Option 3: Modify BasinNSELoss to Handle Unknown Basins (Not Recommended)

Add fallback behavior when encountering unknown basins:

```python
# In losses.py BasinNSELoss.forward()

for i, basin_id in enumerate(batch.group_identifiers):
    if basin_id not in self._group_to_idx:
        # Fallback: use mean normalization factor across all training basins
        logger.warning(
            f"Basin '{basin_id}' not in training set. "
            f"Using mean normalization factor for evaluation."
        )
        norm_factor = self.norm_factors.mean()
    else:
        group_idx = self._group_to_idx[basin_id]
        norm_factor = self.norm_factors[group_idx]
    # ...
```

**Pros:**

- Preserves Basin NSE loss during evaluation
- No changes to test_step

**Cons:**

- Semantically questionable (normalizing by training basin stats)
- Adds complexity to loss function
- Still doesn't solve the fundamental train/test split issue
- The metric becomes less interpretable

---

## Recommendation

**Implement Option 1** - Use MSE for test loss reporting in all evaluation scenarios.

**Justification:**

1. Simplest implementation
2. Test loss is purely cosmetic (doesn't affect model behavior or analysis)
3. MSE is the de facto standard for regression reporting
4. Maintains clean separation between training objectives and evaluation reporting
5. No configuration changes needed
6. Fully backward compatible

**Additional suggestion:** Add a log message during evaluation to clarify which loss is being used:

```python
logger.info(
    f"Using MSE for test loss reporting "
    f"(model trained with {self.config.loss_fn})"
)
```

---

## Example Fix

Here's a complete implementation of Option 1:

```python
# In src/transfer_learning_publication/models/base/base_lit_model.py

def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
    """Execute test step and collect outputs.

    Args:
        batch: Batch of data
        batch_idx: Index of current batch

    Returns:
        Dictionary with predictions and metadata

    Note:
        Test loss is always computed using MSE for consistent reporting
        across all models, regardless of training loss function. This
        ensures compatibility with training-specific losses like basin_nse.
    """
    self._validate_batch(batch)
    x_normalized = self._apply_rev_in_normalization(batch.X)
    y_hat = self(x_normalized, batch.static, batch.future)
    y_hat = self._apply_rev_in_denormalization(y_hat)

    # Always use MSE for test loss reporting
    # This provides a consistent metric across all models and avoids
    # issues with training-specific losses that require basin context
    test_loss = nn.functional.mse_loss(y_hat, batch.y.unsqueeze(-1))
    self.log("test_loss", test_loss, batch_size=batch.batch_size)

    # Store predictions for later retrieval
    predictions = self._create_predictions_dict(y_hat, batch)
    self._forecast_output_buffer.append(predictions)

    return predictions
```

Add import at top of file:

```python
import torch.nn as nn  # Add if not already present
```

---

## Testing

After implementing the fix, verify:

1. **Basin NSE models can be evaluated:**

   ```bash
   tl-evaluate experiments/small_models_275k_params_12fold_all_features.yaml \
     --eval-config configs/evaluation/fold_1_eval_all_features.yaml \
     --models ealstm_275k_fold_1_all_features
   ```

2. **Test loss is logged:**
   - Check that `test_loss` appears in the output
   - Verify it's a reasonable MSE value (not NaN or inf)

3. **Predictions are still correct:**
   - Load the parquet output
   - Verify predictions match expected ranges
   - Compute NSE/KGE metrics manually to ensure evaluation works

4. **Backward compatibility:**
   - Test with MSE-trained models
   - Test with Power loss models
   - Verify all produce evaluation outputs

---

## Alternative: Skip Test Loss Logging Entirely

If the test loss metric is not needed at all, a simpler approach would be to **remove test loss logging** from `test_step()`:

```python
def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
    """Execute test step and collect outputs."""
    self._validate_batch(batch)
    x_normalized = self._apply_rev_in_normalization(batch.X)
    y_hat = self(x_normalized, batch.static, batch.future)
    y_hat = self._apply_rev_in_denormalization(y_hat)

    # No loss computation/logging during test
    # Metrics are computed post-hoc from parquet outputs

    predictions = self._create_predictions_dict(y_hat, batch)
    self._forecast_output_buffer.append(predictions)
    return predictions
```

This is even simpler but loses the convenience of having test loss in logs.

---

## Files Affected

- `src/transfer_learning_publication/models/base/base_lit_model.py` - Modify `test_step()`
- `docs/cli_guide.md` - Document evaluation loss behavior (optional)
- `tests/models/test_base_lit_model.py` - Add test for evaluation with basin_nse models

---

## References

- Basin NSE Loss implementation: `src/transfer_learning_publication/models/losses.py:66-251`
- Test step implementation: `src/transfer_learning_publication/models/base/base_lit_model.py:282`
- Model evaluator: `src/transfer_learning_publication/models/model_evaluator.py:210`
- Related bug report: `scratchpad/bug_report_tl_evaluate_auto_discovery.md` (different issue, now fixed)
