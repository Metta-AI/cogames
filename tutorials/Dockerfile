# CoGames Tutorial Environment
# Dockerfile for running tutorial notebooks with all dependencies

FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    jupyterlab \
    torch --index-url https://download.pytorch.org/whl/cpu \
    imageio \
    imageio-ffmpeg \
    matplotlib \
    numpy \
    scipy \
    pandas \
    seaborn \
    gymnasium \
    pufferlib-core \
    pydantic>=2.11.5 \
    pyyaml>=6.0.2 \
    einops>=0.8.0 \
    rich>=13.7.0

# Copy cogames source
COPY . /workspace/cogames
WORKDIR /workspace/cogames

# Install cogames and mettagrid
RUN pip install -e .

# Copy tutorial files
COPY tutorials /workspace/tutorials

# Expose JupyterLab port
EXPOSE 8888

# Set working directory to tutorials
WORKDIR /workspace/tutorials

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

