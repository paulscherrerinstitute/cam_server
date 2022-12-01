import logging
import os
import socket

from cam_server import __VERSION__, config

# OpenTelemetry utilities
# conda install -c conda-forge opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-elasticsearch opentelemetry-instrumentation-wsgi opentelemetry-instrumentation-requests opentelemetry-instrumentation opentelemetry-instrumentation-logging opentelemetry-exporter-otlp-proto-grpc

otel_resources = {}
def get_otel_resource(service_name=config.TELEMETRY_SERVICE):
    global otel_resources
    if not config.TELEMETRY_ENABLED:
        return None

    if not service_name in otel_resources.keys():
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_INSTANCE_ID, SERVICE_VERSION, \
            PROCESS_PID
        otel_resources[service_name] = Resource(attributes={SERVICE_NAME: config.TELEMETRY_SERVICE, \
                                                            SERVICE_INSTANCE_ID: socket.gethostname(), \
                                                            SERVICE_VERSION: __VERSION__, \
                                                            PROCESS_PID: os.getpid()})
    return otel_resources[service_name];


def otel_auto_instrument(app):
    if not config.TELEMETRY_ENABLED:
        return

    from opentelemetry.instrumentation.wsgi import OpenTelemetryMiddleware
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    resource = get_otel_resource()
    if config.TELEMETRY_COLLECTOR is None:
        tracer_exporter = ConsoleSpanExporter()
        metric_exporter = ConsoleMetricExporter()
    else:
        tracer_exporter = OTLPSpanExporter(endpoint=config.TELEMETRY_COLLECTOR)
        metric_exporter = OTLPMetricExporter(endpoint=config.TELEMETRY_COLLECTOR)

    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(tracer_exporter)
    tracer_provider.add_span_processor(processor)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    return OpenTelemetryMiddleware(app, tracer_provider=tracer_provider, meter_provider=meter_provider)


otel_log_handler = None
def otel_setup_logs():
    global otel_log_handler

    if not config.TELEMETRY_ENABLED:
        return
    from opentelemetry.sdk import _logs
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LogEmitterProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogProcessor, ConsoleLogExporter
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    resource = get_otel_resource()
    if otel_log_handler in logging.getLogger().handlers:
        return

    if config.TELEMETRY_COLLECTOR is None:
        log_exporter = ConsoleLogExporter()
    else:
        log_exporter = OTLPLogExporter(endpoint=config.TELEMETRY_COLLECTOR, insecure=True)
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging._checkLevel(config.TELEMETRY_LOG_LEVEL))
    logging.getLogger().handlers.clear()
    if len(logging.getLogger().handlers) == 1:
        stream_handler =logging.getLogger().handlers[0]
    else:
        stream_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stream_handler)
    stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger_emitter_provider = LogEmitterProvider(resource=resource)
    _logs.set_log_emitter_provider(logger_emitter_provider)
    log_processor = BatchLogProcessor(log_exporter)
    logger_emitter_provider.add_log_processor(log_processor)
    log_emitter = logger_emitter_provider.get_log_emitter(__name__, __VERSION__)
    otel_log_handler = LoggingHandler(level=logging.INFO, log_emitter=log_emitter)
    if config.TELEMETRY_LOG_SPAN_ONLY:
        class LogsInSpansFilter(logging.Filter):
            def filter(self, record):
                return record.otelSpanID != "0"
        otel_log_handler.addFilter(LogsInSpansFilter())
    otel_log_handler.setFormatter(logging.Formatter(config.TELEMETRY_LOG_FORMAT))
    logging.getLogger().addHandler(otel_log_handler)

otel_tracers = {}
def otel_get_tracer(name=config.TELEMETRY_SERVICE):
    global otel_tracer

    if not config.TELEMETRY_ENABLED:
        return

    if name in otel_tracers.keys():
        return otel_tracers[name]
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    if len(otel_tracers) == 0:
        resource = get_otel_resource ()
        if config.TELEMETRY_COLLECTOR is None:
            tracer_exporter = ConsoleSpanExporter()
        else:
            tracer_exporter = OTLPSpanExporter(endpoint=config.TELEMETRY_COLLECTOR)

        tracer_provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(tracer_exporter)
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(name)
    otel_tracers[name] = tracer
    return tracer


otel_meters={}
def otel_get_meter(name=config.TELEMETRY_SERVICE):
    global otel_meters

    if not config.TELEMETRY_ENABLED:
        return

    if name in otel_meters.keys():
        return otel_meters[name]
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    if len(otel_tracers) == 0:
        resource = get_otel_resource()
        if config.TELEMETRY_COLLECTOR is None:
            metric_exporter = ConsoleMetricExporter()
        else:
            metric_exporter = OTLPMetricExporter(endpoint=config.TELEMETRY_COLLECTOR)
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter(name)
    otel_meters[name] = meter
    return meter
