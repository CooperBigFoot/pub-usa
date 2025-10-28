# CAMELS Basin Filtering and Cross-Validation Setup

**Date**: 2025-10-28
**Author**: Nicolas Lazaro

## Overview

Created basin ID lists for CAMELS region with area filtering and 12-fold cross-validation splits for model training experiments.

---

## 1. Basin Filtering by Area

### Objective
Filter CAMELS basins to include only those with watershed area < 2000 km² (following Newman et al. 2017 size threshold).

### Script
`scripts/filter_camels_by_area.py`

### Method
1. Load CAMELS static attributes from cleaned CARAVAN data
2. Filter using `area` column where area < 2000 km²
3. Extract and sort gauge IDs

### Results
- **Input**: 671 CAMELS basins
- **Output**: 617 basins (54 removed)
- **File**: `/Users/nicolaslazaro/Desktop/work/pub-usa/configs/basin_ids_files/camels_area_lt_2000.txt`

---

## 2. Newman et al. (2017) Area Discrepancy Analysis (Abandoned)

### Objective
Attempted to replicate Newman et al. (2017) filtering which removed basins with >10% discrepancy between different area calculation methods.

### Script
`scripts/filter_camels_newman.py`

### Method
1. Load CAMELS shapefile geometries
2. Calculate area from shapefile polygons (reprojected to ESRI:102003 - USA Albers Equal Area Conic)
3. Compare with `area` attribute from static metadata
4. Filter basins with >10% discrepancy OR area >= 2000 km²

### Results
- **Area discrepancy**: Min 0.00%, Max 0.31%, Mean 0.12%
- **Basins removed by discrepancy**: 0 (all <10%)
- **Basins removed by size**: 54 (area >= 2000 km²)
- **Final count**: 617 basins

### Why Not 531 Basins?
Newman et al. (2017) reported 531 basins after filtering. Our analysis yielded 617 because:
- The cleaned CARAVAN dataset has consistent area measurements between shapefile geometry and static attributes
- Original Newman et al. likely compared against different metadata sources (e.g., USGS GAGES-II) that had actual discrepancies
- **Decision**: Use the simpler `camels_area_lt_2000.txt` (617 basins) instead

---

## 3. 12-Fold Cross-Validation Splits

### Objective
Create 12-fold cross-validation setup for model training experiments.

### Script
`scripts/create_12_folds.py`

### Method
1. Load 617 basin IDs from `camels_area_lt_2000.txt`
2. Randomly shuffle with seed=42 (reproducible)
3. Split into 12 equal folds
4. For each fold k:
   - Test: basins in fold k
   - Train: basins in all other 11 folds

### Results
- **Total basins**: 617
- **Folds 1-5**: 52 test, 565 train
- **Folds 6-12**: 51 test, 566 train
- **Files created**: 24 (12 train + 12 test)
- **Location**: `/Users/nicolaslazaro/Desktop/work/pub-usa/configs/basin_ids_files/`

### File Naming
```
train_fold_1.txt, test_fold_1.txt
train_fold_2.txt, test_fold_2.txt
...
train_fold_12.txt, test_fold_12.txt
```

### Verification
✓ No overlap between train/test within each fold
✓ All 617 basins appear exactly once as test across all folds
✓ Train + test = 617 for each fold

---

## Data Sources

### Input Data
- **Path**: `/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_eval/train/`
- **Cleaning script**: `scripts/clean_caravan_with_splits.py`
- **Region**: CAMELS (Continental United States)
- **Total gauges**: 671

### Static Attributes
- Source: Hive-partitioned parquet files
- Key column: `area` (watershed area in km²)

### Shapefiles
- File: `camels_basin_shapes.shp`
- Original CRS: EPSG:4326 (WGS 84)
- Projected CRS for area calculation: ESRI:102003 (USA Albers Equal Area Conic)

---

## Final Dataset Summary

| Metric | Value |
|--------|-------|
| Total CAMELS basins | 671 |
| Basins with area < 2000 km² | 617 |
| Basins removed | 54 |
| Cross-validation folds | 12 |
| Random seed | 42 |

---

## Usage

### Training a model on fold 1
```bash
# Train on 565 basins from train_fold_1.txt
# Test on 52 basins from test_fold_1.txt
```

### Running 12-fold cross-validation
```bash
for k in {1..12}; do
    echo "Training fold $k..."
    # Use train_fold_${k}.txt for training
    # Use test_fold_${k}.txt for evaluation
done
```
