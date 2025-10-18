# Dockerfile (example: Python 3.8, CUDA 11.7 image)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# set noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-venv python3-pip git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# install pip for python3.8
RUN python3.8 -m ensurepip --upgrade && python3.8 -m pip install --upgrade pip

# copy app
WORKDIR /app
COPY . /app

# install python deps (adjust to match your requirements exactly)
RUN python3.8 -m pip install --upgrade pip setuptools wheel
RUN python3.8 -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

EXPOSE 8080
CMD ["python3.8", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
