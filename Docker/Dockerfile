FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

LABEL maintainers="ksj6659@gmail.com"
LABEL title="swuniv-ai-competition-final"
LABEL version="0.1"
LABEL description="SWUniv AI Competition Environment"

RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    pip install numpy pandas matplotlib tqdm jamo sklearn opencv-python arabic_reshaper python-bidi wikipedia fire lmdb Pillow==9.0.0 jupyter

WORKDIR /workspace