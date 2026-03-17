FROM --platform=linux/amd64 python:3.11-slim

# System deps required by OpenCV and pycolmap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    pycolmap \
    Pillow \
    numpy \
    opencv-python-headless

# Copy source code (datasets are mounted as a volume)
COPY pipeline.py colmap2nerf.py ./
COPY libs/ ./libs/

CMD ["python", "pipeline.py"]
