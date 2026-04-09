# show-attend-tell-implementation
---

## 🛠 Docker Setup & Usage

This project is containerized to ensure consistent environments for image captioning experiments. It uses a **src-layout** and is installed as an editable package inside the container.

### 1. Build the Image
Before running for the first time (or after changing `pyproject.toml`/`requirements.txt`), build the Docker image:

```bash
docker build -t show-attend-tell .
```

### 2. Run the Container
Use the following command to start the container. This setup mounts your local directory into the container so that code changes are reflected instantly without rebuilding.

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
* `--gpus all`: Enables NVIDIA GPU support (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
* `-v $(pwd):/workspace`: Mounts your current project folder to `/workspace` inside the container.
* `-w /workspace`: Sets the working directory so Python can find the `src` package.
* `--rm`: Automatically removes the container when you exit.

### 3. Running Scripts
The project uses Python's module syntax. To run the main entry point:

```bash
# Inside the container (or as a trailing command to 'docker run')
python -m src.main
```

