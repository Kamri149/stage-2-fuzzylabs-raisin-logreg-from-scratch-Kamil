# Raisin Logistic Regression (from scratch)

Implements and trains logistic regression using only built-in Python libraries + NumPy.
Includes a minimal JSON HTTP inference server using only `http.server` (built-in) + NumPy.

## 1) Dataset

Place the Raisin CSV in the repo root as:

- `Raisin_Dataset.csv`

Expected columns:
- Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter
- Class (binary: e.g. Kecimen / Besni)

Note: if your CSV has a UTF-8 BOM on the first column (e.g. "\ufeffArea"), the loader handles it via `encoding="utf-8-sig"`.

## 2) Local setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e .