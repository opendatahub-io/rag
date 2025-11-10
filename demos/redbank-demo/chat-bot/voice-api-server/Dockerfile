# OpenShift-friendly base (runs as non-root, supports random UID)
FROM registry.access.redhat.com/ubi9/python-312:latest

ENV LLAMASTACK_URL="http://localhost:8321"
ENV INFERENCE_MODEL="vllm-inference/qwen2-5"
ENV VECTOR_STORE_NAME="redbank-kb-vector-store"
ENV WHISPER_MODEL="whisper-large-v3-turbo-quantized"
ENV WHISPER_URL="http://localhost:8080/v1"

USER 0

# Create app dir with group 0 write so random UID can write
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# OpenShift runs containers with a random UID; ensure group 0 has write
RUN chgrp -R 0 /opt/app && chmod -R g+rwX /opt/app

# Drop to non-root
USER 1001

EXPOSE 8080

ENV HOST=0.0.0.0 PORT=8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
