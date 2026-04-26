FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	ca-certificates \
	build-essential \
	openssh-client \
	curl \
	ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install -r requirements.txt \
	&& pip install -e .

ENV TORCH_HOME=/workspace/.cache/torch

RUN python -c "from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.DEFAULT)"

EXPOSE 8888

CMD ["python", "-m", "src.main"]