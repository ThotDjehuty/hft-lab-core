# hfthot-lab-core: Jupyter demo container
# Builds the PyO3 extension and exposes Jupyter Lab on port 8888
FROM python:3.11-slim

# Install Rust toolchain
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin, Jupyter, and all notebook dependencies
RUN pip install --no-cache-dir maturin jupyterlab numpy scipy matplotlib pandas seaborn

WORKDIR /opt/hfthot-lab-core

# Copy source
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/
COPY python/ ./python/
COPY examples/ ./examples/
COPY README.md ./

# Build wheel and install into system Python
RUN maturin build --release --features python --out /tmp/wheels
RUN pip install /tmp/wheels/*.whl

# Work from notebooks dir so Path.cwd().parents[1] == /opt/hfthot-lab-core/
WORKDIR /opt/hfthot-lab-core/examples/notebooks

EXPOSE 8888

# JUPYTER_TOKEN must be set in docker-compose env or runtime environment.
# Leaving it empty disables token auth — never do that in production.
# Shell form is required so ${JUPYTER_TOKEN} is expanded at container start.
CMD jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.password="" \
    --ServerApp.allow_origin="*"
