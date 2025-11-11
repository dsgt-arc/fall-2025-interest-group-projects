# Visualizing and Clustering Fashion MNIST Embeddings

This notebook demonstrates a pipeline for leveraging **pretrained deep learning models** to extract meaningful numerical representations (embeddings) from images. We then use these embeddings to visualize data structure, perform clustering, and enable similarity search **without any additional training**.

---

## Setting up the Virtual Environment

To stay within your home directory quota, we'll create the virtual environment in the fast, temporary file system located at **`$TMPDIR`**.

Make sure to add this to your `~/.bashrc` file:
```bash
# Set a temporary directory
export TMPDIR=${TMPDIR:-/tmp}
```

### Option A. Manual Virtual Environment Setup (Recommended for Jupyter Users)

This approach creates a reusable `.venv` in `$TMPDIR` that you can activate manually and connect to your Jupyter notebook.

#### 1. **Create and Activate the Environment in `$TMPDIR`:**

```bash
# Create the virtual environment directly in $TMPDIR
uv venv $TMPDIR/.venv

# Activate it
source $TMPDIR/.venv/bin/activate
```

#### 2. **Navigate to the `user/` directory and link dependencies:**

Navigate to your user directory and create a **symbolic link** named `.venv` that points back to the environment in `$TMPDIR`.

Assuming your current repo user directory is:
`~/dsgt-arc/fall-2025-interest-group-projects/user/<your-user-name>`

```bash
# Navigate to your project directory
cd ~/dsgt-arc/fall-2025-interest-group-projects/user/<your-user-name>

# Create a symbolic link named .venv in your user folder, 
# pointing to the actual environment in $TMPDIR
ln -s $TMPDIR/.venv $(pwd)/.venv

# Install project dependencies defined in pyproject.toml
cd ~/dsgt-arc/fall-2025-interest-group-projects/project/02-embeddings
uv pip install -e .
```

Once done, open the Jupyter notebook and select the newly linked **`.venv`** kernel.

---

### Option B. Using `UV_PROJECT_ENVIRONMENT` (Recommended for Batch or Sbatch Runs)

You can configure the environment path once via an environment variable, which is useful for non-interactive jobs.

```bash
# Set the environment variable to the $TMPDIR location
export UV_PROJECT_ENVIRONMENT=$TMPDIR/.venv

# Install all dependencies, including optional [dev] extras
uv sync --all-extras

# Navigate to project directory and link it for easier access (optional)
# This step assumes you are in your project directory
cd /storage/home/hcoda1/9/mgustineli3/dsgt-arc/fall-2025-interest-group-projects/user/mgustineli/project/01-llm-lora/
ln -s $UV_PROJECT_ENVIRONMENT $(pwd)/.venv
```

## Key Concepts Covered

### 1. Embeddings
* **Deep models** (specifically, a **DINO** variant) are used to transform raw images into high-dimensional numerical vectors.
* These vectors, or **embeddings**, encode both **visual and semantic similarity**; images that look alike or belong to the same category will have embeddings that are close in vector space.

### 2. Clustering in Embedding Space
* We show that similar images (e.g., all "Sneakers" or all "Dresses" from Fashion MNIST) naturally form **clusters** because the pretrained model has already learned general, useful visual features.

### 3. Dimensionality Reduction
* The high-dimensional embeddings are projected into a 2D space for visualization using **UMAP (Uniform Manifold Approximation and Projection)**.
* The quality of the cluster separation is quantitatively assessed using the **Silhouette Score**.

### 4. Similarity Search
* We use a **Nearest Neighbor** search approach on the embeddings to find images most similar to a query image. This highlights the utility of embeddings for tasks like **recommendation systems** or **content-based image retrieval**.

---

## Possible Extensions

The following are suggested next steps for deeper exploration:

1.  **Model Comparison:** Evaluate different DINO model architectures (e.g., ViT-S, ViT-B) to compare their embedding quality.
2.  **UMAP Tuning:** Experiment with UMAP parameters (`n_neighbors`, `min_dist`) to observe the effect on visualization and cluster structure.
3.  **Alternative Methods:** Compare the results of UMAP against other dimensionality reduction techniques like **PCA** or **t-SNE**.
4.  **Classification:** Implement a **K-Nearest Neighbor (K-NN)** classifier directly on the embeddings to perform classification.
5.  **New Datasets:** Apply this pipeline to a different image dataset (e.g., CIFAR-10) or a custom collection of images.
