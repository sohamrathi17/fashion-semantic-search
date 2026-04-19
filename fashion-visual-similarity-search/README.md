# 👗 Fashion Visual Similarity Search with CLIP + FAISS

A semantic visual search engine for fashion items. Given a **text query** (*"blue denim jacket"*) or a **query image**, it retrieves the most visually similar clothing items from 10,000 DeepFashion2 images — powered by OpenAI's CLIP model and a FAISS vector index.

---

## 📌 Problem Statement

Traditional keyword search fails at visual nuance — it cannot understand that *"floral midi dress"* and *"flower-patterned skirt dress"* refer to similar items. This project builds an **embedding-based retrieval system** that understands the semantic meaning of fashion, both from text and images.

---

## 🗂️ Project Structure

```
fashion-visual-similarity-search/
├── semantic-fashion-search-2.ipynb     # Full Kaggle notebook
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🔄 Pipeline Overview

```
DeepFashion2 Dataset (10,000 images)
        ↓
  Crop clothing items via bounding boxes
        ↓
  CLIP Image Encoder → 512-dim embeddings
        ↓
  L2 Normalization
        ↓
  FAISS Index (IndexFlatIP / cosine similarity)
        ↓
  ┌──────────────────────────────┐
  │  Query: Text or Image        │
  │  → CLIP Encoder → embedding  │
  └──────────────────────────────┘
        ↓
  Top-K nearest neighbours retrieved
        ↓
  Display matched fashion items
```

---

## 🔑 Key Components

| Component | Role |
|---|---|
| **CLIP** (`openai/clip-vit-base-patch32`) | Encodes both images and text into a shared 512-dim embedding space |
| **FAISS** (`IndexFlatIP`) | High-speed nearest-neighbour search over embeddings |
| **DeepFashion2** | Large-scale fashion dataset with 200K+ images and bounding box annotations |

---

## 🔍 Search Modes

### Text-to-Image Search
```python
query_text = "red floral dress"
results = search_by_text(query_text, top_k=6)
```

### Image-to-Image Search
```python
query_image_path = "path/to/your/image.jpg"
results = search_by_image(query_image_path, top_k=6)
```

---

## 💾 Saved Artifacts

After running the notebook, the following are saved:

| File | Description |
|---|---|
| `artifacts_fashion/faiss.index` | FAISS vector index |
| `artifacts_fashion/embeddings.npy` | Raw 512-dim embedding matrix |
| `artifacts_fashion/metadata.json` | Image paths + category labels |
| `fashion_thumbs.zip` | Cropped thumbnails at 224×224 px |

---

## 🚀 How to Run (Kaggle — Recommended)

> This notebook is designed for **Kaggle** with a **GPU accelerator**.

1. Go to [kaggle.com](https://www.kaggle.com) → **Create Notebook**
2. Add the dataset: search for `thusharanair/deepfashion2-original-with-dataframes`
3. In **Settings → Accelerator**, select **GPU T4**
4. Upload and run `semantic-fashion-search-2.ipynb` cell by cell

---

## 🖥️ How to Run Locally (CPU)

```bash
git clone https://github.com/YOUR_USERNAME/fashion-visual-similarity-search.git
cd fashion-visual-similarity-search
pip install -r requirements.txt
jupyter notebook semantic-fashion-search-2.ipynb
```

> ⚠️ Running on CPU is significantly slower for embedding generation. Reduce the dataset size (e.g., `.head(500)`) for quick local testing.

---

## 📦 Requirements

```
numpy
pandas
Pillow
matplotlib
torch
transformers
faiss-cpu
tqdm
```

---

## 📝 Key Design Decisions

- **Bounding box cropping** removes background clutter before embedding, improving retrieval quality.
- **L2-normalized embeddings** with `IndexFlatIP` is equivalent to cosine similarity search.
- **CLIP's shared embedding space** allows text and image queries to be compared directly — no separate training needed.
- The index covers **10,000 images**. To scale up, remove the `.head(10000)` sampling limit.

---

## 👤 Author

> Add your name and links here (GitHub, LinkedIn, Kaggle).
