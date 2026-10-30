FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="team@semantic.works"

# The template's ONBUILD hook copies requirements.txt (and build.sh, if present)
# from the build context to /app and runs build.sh followed by
# `uv pip install -r requirements.txt` into the project venv. build.sh installs
# the CPU-only torch wheel first, so sentence-transformers reuses it instead of
# pulling the CUDA-bundled torch with its multi-GB nvidia libraries.

# Bake the cross-encoder reranker into the image (pinned revision) so containers
# start without downloading anything at runtime. Only the safetensors weights,
# tokenizer and config files are kept; the duplicate pytorch/tf weights are skipped.
RUN /usr/src/app/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', allow_patterns=['*.json', '*.safetensors', 'sentencepiece.bpe.model', 'tokenizer*'])"

ENV RERANK_MODEL_PATH=/root/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
