# Fashion Visual Similarity Search with CLIP and FAISS

This project builds a semantic visual search engine for fashion items. Given a query image, it retrieves the most visually similar clothing items from a subset of DeepFashion2 images using CLIP embeddings and a FAISS vector index.

---

## Problem Statement

Traditional keyword-based search fails to capture visual similarity and semantic meaning in fashion. For example, "floral midi dress" and "flower-patterned dress" may refer to visually similar items but are treated as different queries. This project uses embedding-based retrieval to capture semantic similarity across both text and images.

---

## Project Structure

fashion-visual-similarity-search/
    semantic-fashion-search-2.ipynb
    requirements.txt
    README.md

- semantic-fashion-search-2.ipynb contains the full pipeline including preprocessing, embedding generation, indexing, and retrieval
- requirements.txt lists all required Python dependencies

---

## Pipeline Overview

The pipeline begins with a subset of the DeepFashion2 dataset consisting of approximately 10,000 images.

Clothing items are cropped using bounding box annotations to remove background noise.

The cropped images are passed through a CLIP image encoder to generate 512-dimensional embeddings.

These embeddings are normalized and stored in a FAISS index using cosine similarity.

At query time, either a text query or an image is encoded using CLIP and compared against the stored embeddings.

The system retrieves the top-K nearest neighbours and displays visually similar fashion items.

---

## Key Components

Component         Role
CLIP              Encodes images and text into a shared 512-dimensional embedding space
FAISS             Performs efficient nearest-neighbour search over embeddings
DeepFashion2      Provides fashion images with bounding box annotations

---

## Search Modes

Text-to-Image Search

query_text = "red floral dress"
results = search_by_text(query_text, top_k=6)

Image-to-Image Search

query_image_path = "path/to/your/image.jpg"
results = search_by_image(query_image_path, top_k=6)

---

## Saved Artifacts

File                              Description
artifacts_fashion/faiss.index     FAISS vector index
artifacts_fashion/embeddings.npy  Raw embedding matrix
artifacts_fashion/metadata.json   Image paths and labels
fashion_thumbs.zip                Cropped thumbnails

---

## How to Run (Kaggle Recommended)

This notebook is designed to run on Kaggle with GPU support.

1. Create a new notebook on Kaggle
2. Add the dataset "thusharanair/deepfashion2-original-with-dataframes"
3. Enable GPU (T4) in settings
4. Upload and run semantic-fashion-search-2.ipynb

---

## How to Run Locally

git clone https://github.com/YOUR_USERNAME/fashion-visual-similarity-search.git
cd fashion-visual-similarity-search
pip install -r requirements.txt
jupyter notebook semantic-fashion-search-2.ipynb

Running on CPU will be slower for embedding generation. For quick testing, reduce dataset size.

---

## Requirements

numpy
pandas
Pillow
matplotlib
torch
transformers
faiss-cpu
tqdm

---

## Key Design Decisions

- Bounding box cropping is used to remove background and improve embedding quality
- L2-normalized embeddings with IndexFlatIP enable cosine similarity search
- CLIP provides a shared embedding space for both text and images, eliminating the need for additional training
- The current implementation uses approximately 10,000 images and can be scaled further

---

## Author

Soham Rathi
sohamrathi.com
