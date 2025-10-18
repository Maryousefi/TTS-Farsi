# Dockerfile
# Base image with CUDA 11.7 + cuDNN8
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and dependencies
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-venv python3-pip git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3.8 -m ensurepip --upgrade && python3.8 -m pip install --upgrade pip setuptools wheel

# Copy project files
WORKDIR /app
COPY . /app

# Install requirements
RUN python3.8 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

# Expose port
EXPOSE 8080

# Run the FastAPI app
CMD ["python3.8", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
