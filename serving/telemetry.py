import os

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("serving_requests_total", "Inference outcomes", ["outcome", "backend"])
LATENCY = Histogram(
    "serving_latency_seconds",
    "End-to-end inference time",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
TOKENS = Counter("serving_output_tokens_total", "Backend-reported output tokens", ["backend"])
INFLIGHT = Gauge("serving_inflight", "In-flight inference requests")
JOBS = Counter("serving_jobs_total", "Worker outcomes", ["outcome"])
QUEUE = Gauge("serving_queue_active", "Queued plus leased jobs")


def configure(service):
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)


tracer = trace.get_tracer("distributed-model-serving")
