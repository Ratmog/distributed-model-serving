FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY serving serving
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home app
COPY scripts scripts
USER app
EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
