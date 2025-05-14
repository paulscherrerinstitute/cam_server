def get_host_port_from_stream_address(stream_address):
    if stream_address.startswith("ipc"):
        return stream_address.split("//")[1], -1
    source_host, source_port = stream_address.rsplit(":", maxsplit=1)
    if "//" in source_host:
        source_host = source_host.split("//")[1]
    return source_host, int(source_port)

