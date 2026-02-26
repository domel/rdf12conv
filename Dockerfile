FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/rdf12conv

# Copy only files required to build/install the CLI tool.
COPY pyproject.toml README.md LICENSE rdf_converter.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install .

# Default working directory for mounted input/output files.
WORKDIR /work

ENTRYPOINT ["rdf12conv"]
CMD ["--help"]
