# 📦 Sharing Dataset & Model Files

## What to Share

Your friend needs these files to run predictions:

### Required (for training):
- `data/fe/features.parquet` (~5-10 MB) - Final training dataset

### Optional (to skip training):
- `models/ranker_lgb.txt` (~1-2 MB) - Trained LightGBM model
- `models/ranker_meta.joblib` (~few KB) - Model metadata

---

## Sharing Methods

### Method 1: Cloud Storage (Easiest)
1. Upload `data/fe/features.parquet` to:
   - Google Drive
   - Dropbox
   - OneDrive
   - WeTransfer
2. Share the link with your friend
3. They download and place in `data/fe/`

### Method 2: Rebuild from Raw Data (Most Reproducible)
Your friend can rebuild everything from scratch:

```bash
# 1. Fetch raw data (takes ~30-60 min, but ensures consistency)
python -m backend.ml.fetch_data --years 2018-2025

# 2. Build features
python -m backend.ml.build_features

# 3. Train model
python -m backend.ml.trainer --tune
```

### Method 3: Git LFS (For Version Control)
If you want to track these files in git:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/fe/*.parquet"
git lfs track "models/*.txt"
git lfs track "models/*.joblib"

# Add and commit
git add .gitattributes
git add data/fe/features.parquet
git add models/*.txt models/*.joblib
git commit -m "Add training dataset and model files"
```

---

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| `features.parquet` | ~0.1 MB | Training dataset (173 races, 3,978 rows) |
| `ranker_lgb.txt` | ~0.03 MB | Trained model |
| `ranker_meta.joblib` | ~few KB | Model config |

**Total: ~0.13 MB** (super easy to share - can even email!)

---

## Quick Setup for Your Friend

After receiving the files:

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd f1-project

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Place shared files
# - Put features.parquet in data/fe/
# - Put model files in models/ (if shared)

# 4. Verify it works
python -m backend.ml.predict --year 2024
```

---

## Recommendation

**Best approach:** Share `features.parquet` only, let them train the model themselves.

**Why?**
- ✅ Ensures they have the exact same training setup
- ✅ They can tune hyperparameters if needed
- ✅ Smaller file to share (~5-10 MB vs ~7-12 MB)
- ✅ More reproducible

