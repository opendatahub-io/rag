FROM registry.access.redhat.com/ubi9/python-312:latest

ENV KOKORO_URL=http://kokoro:8000
# For Llama Stack and voice functionality
ENV VOICE_API_BASE_URL=http://voice:8000

WORKDIR /opt/app

# Switch to root to install dependencies and set permissions
USER 0

COPY . /opt/app

# Create README.md if it doesn't exist
RUN touch README.md

# Install the package and fix permissions
RUN pip install -e . && \
    chown -R 1001:0 /opt/app && \
    chmod -R g=u /opt/app

# Switch back to non-root user
USER 1001

EXPOSE 3000
CMD ["python", "main.py"]
