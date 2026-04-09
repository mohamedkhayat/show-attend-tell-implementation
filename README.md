# Show, Attend and Tell: Implementation

An implementation of the "Show, Attend and Tell" image captioning paper. This project is optimized for university research, utilizing a **src-layout** for package integrity and **Docker** for environment reproducibility.

## 📂 Project Structure

```text
└── paper_implementation_uni/
    ├── src/
    │   ├── dataset/         # Dataset classes, Vocabulary, and Factories
    │   ├── utils/           # Data splitting and preparation scripts
    │   └── main.py          # Development entry point
    ├── data/                # Local data storage (Git ignored)
    ├── Dockerfile           # GPU-ready environment
    ├── pyproject.toml       # Project metadata and editable install config
    └── .devcontainer/       # VS Code containerized development config
```

---

## 🛠 Docker Setup & Usage

This project is fully containerized. It is installed as an **editable package** inside the container, meaning code changes in your local `src/` folder are reflected immediately inside the running container.

### 1. Build the Image
```bash
docker build -t show-attend-tell .
```

### 2. Run the Container
```bash
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -w /workspace \
  show-attend-tell \
  /bin/bash
```

#### Flag Breakdown:
* `--gpus all`: Enables NVIDIA GPU support.
* `-v $(pwd):/workspace`: Mounts your current project folder to `/workspace` inside the container.
* `-w /workspace`: Sets the working directory so Python can resolve the `src` package.
* `--rm`: Automatically removes the container on exit.

---

## 🚀 Quick Start

### Data Preparation
1. **Download**: Obtain the [Flickr8k Dataset from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. **Structure**: Create a `data` folder in the project root and extract the files so they match this exact structure:

```text
data/flicker8k/
├── captions.txt
└── Images/
    ├── 1000268201_693b08cb0e.jpg
    ├── 1001773457_577c3a7d70.jpg
    └── ...
```

*The utility scripts will automatically generate the required `.json` split files in `data/flicker8k/` on the first run.*

### Running the Implementation
To verify the pipeline (inside the container):

```bash
python -m src.main
```

---

## 📊 Features
- **GPU Optimized**: Based on `pytorch/pytorch:2.8.0-cuda12.8`.
- **Automated Pipeline**: Includes `ensure_flicker_splits` logic to handle data prep.
- **Developer Friendly**: Pre-configured `.devcontainer` for VS Code users.
- **Modular Design**: Separated concerns for `Vocabulary`, `Transforms`, and `Datasets`.

---

## ⚖️ License
Distributed under the **MIT License**. See `LICENSE` for more information.