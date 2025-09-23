FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_server/ ./mcp_server/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Waiting for PostgreSQL to be ready..."\n\
until pg_isready -h postgres -p 5432 -U postgres; do\n\
  echo "PostgreSQL is unavailable - sleeping"\n\
  sleep 1\n\
done\n\
echo "PostgreSQL is up - executing command"\n\
echo "Initializing database..."\n\
cd /app && bash scripts/create_db_and_insert_data.sh\n\
echo "Starting MCP server..."\n\
cd /app/mcp_server && python mcp_server.py' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
