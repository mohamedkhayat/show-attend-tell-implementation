FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	ca-certificates \
	build-essential \
	curl \
	ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ -s /workspace/requirements.txt ]; then pip install -r /workspace/requirements.txt; fi

COPY src/ /workspace/src/
COPY config/ /workspace/config/
COPY data/ /workspace/data/

EXPOSE 8888

CMD ["python", "src/main.py"]
