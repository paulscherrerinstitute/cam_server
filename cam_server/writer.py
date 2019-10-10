import argparse
import logging
from cam_server.utils import get_host_port_from_stream_address
from bsread.handlers.compact import Value
from bsread.data.helpers import get_channel_specs

import h5py
import numpy
import socket
import datetime
import os
import getpass
import json

_logger = logging.getLogger(__name__)

DATA_GROUP = "/data/"
HEADER_GROUP = "/header/"
ATTRIBUTE_GROUP = "/general/"

LAYOUT_DEFAULT = "DEFAULT"
LAYOUT_FLAT = "FLAT"

UNDEFINED_NUMBER_OF_RECORDS= -1

LOCALTIME_DEFAULT = True

from bsread import source, SUB, PULL



class Writer(object):
    def __init__(self, output_file="/dev/null",
                       number_of_records = UNDEFINED_NUMBER_OF_RECORDS,
                       layout = LAYOUT_DEFAULT,
                       save_local_timestamps = True,
                       attributes={}):
        self.stream = None
        self.output_file = output_file
        self.attributes = attributes or {}
        if isinstance( self.attributes, str):
            self.attributes = json.loads(self.attributes)
        self.number_of_records = number_of_records
        self.layout = layout
        self.save_local_timestamps = save_local_timestamps

        if self.layout.upper() == LAYOUT_FLAT:
            self.value_dataset_name_format = DATA_GROUP + "%s"
            self.timestamp_dataset_name_format = DATA_GROUP + "%s_timestamp"
        else:
            self.value_dataset_name_format = DATA_GROUP + "%s/value"
            self.timestamp_dataset_name_format = DATA_GROUP + "%s/timestamp"

        self.attributes["created"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
        self.attributes["user"] = getpass.getuser()
        self.attributes["pid"] = os.getpid()
        self.attributes["host"] = socket.gethostname()

        _logger.info("Opening output_file=%s with attributes %s",  self.output_file, self.attributes)

        self.current_record = 0
        self.file = h5py.File(output_file, 'w')
        self._create_attributes_datasets()
        self.scalar_datasets, self.array_datasets = {}, {}
        self.serializers = {}

    def _create_scalar_dataset(self, name, dtype):
        ret =  self.file.create_dataset(name=name,
                                 shape=((self.number_of_records,) if (self.number_of_records > 0) else (0,)),
                                 maxshape= ((self.number_of_records,) if (self.number_of_records > 0) else (None,)),
                                 dtype=dtype)
        self.scalar_datasets[name] = ret
        return ret

    def _create_array_dataset(self, name, dtype, size):
        ret =  self.file.create_dataset(name=name,
                                 shape=(tuple([(self.number_of_records if (self.number_of_records > 0) else 0), ] + list(size))),
                                 maxshape=(tuple([(self.number_of_records if (self.number_of_records > 0) else None), ] + list(size))),
                                 dtype=dtype)
        self.array_datasets[name] = ret
        return ret

    def _append_scalar_dataset(self, name, value):
        dataset = self.scalar_datasets[name]
        if self.number_of_records < 0:
            dataset.resize(size=self.current_record+1, axis=0)
        if name in self.serializers.keys():
            (serializer, dtype) = self.serializers[name]
        if isinstance(value, str):
            value = numpy.string_(value)
        dataset[self.current_record] = value

    def _append_array_dataset(self, name, value):
        dataset = self.array_datasets[name]
        if self.number_of_records < 0:
            dataset.resize(size=self.current_record+1, axis=0)
        if self.serializers.get(name):
            (serializer, dtype) = self.serializers[name]
            value = serializer(value, dtype)
        dataset[self.current_record] = value

    def create_header_datasets(self):
        self._create_scalar_dataset(HEADER_GROUP + "pulse_id", "uint64")
        self._create_array_dataset(HEADER_GROUP + "global_timestamp", "uint64", (2,))


    def append_header(self, pulse_id, global_timestamp, global_timestamp_offset):
        self._append_scalar_dataset(HEADER_GROUP + "pulse_id", pulse_id)
        self._append_array_dataset(HEADER_GROUP + "global_timestamp", [global_timestamp, global_timestamp_offset])

    def create_channel_datasets(self, data):
        for name in data.keys():
            val = data[name]
            timestamp, timestamp_offset, value = val.timestamp, val.timestamp_offset, val.value
            if isinstance(value, numpy.ndarray):
                self._create_array_dataset(self.value_dataset_name_format % name, value.dtype, value.shape)
            else:
                if hasattr(value, 'dtype'):
                    dtype = value.dtype
                else:
                    if isinstance(value, str):
                        dtype = "S1000"
                    else:
                        dtype, _, serializer, _ = get_channel_specs(value, extended=True)
                        self.serializers[self.value_dataset_name_format % name] = (serializer, dtype)
                self._create_scalar_dataset(self.value_dataset_name_format % name, dtype)
            if self.save_local_timestamps:
                self._create_array_dataset(self.timestamp_dataset_name_format % name, "uint64", (2,))

    def append_channel_data(self, data):
        for name in data.keys():
            val = data[name]
            timestamp, timestamp_offset, value = val.timestamp, val.timestamp_offset, val.value
            value_dataset_name = self.value_dataset_name_format % name
            timestamp_dataset_name = self.timestamp_dataset_name_format % name
            if value_dataset_name in self.scalar_datasets:
                self._append_scalar_dataset(value_dataset_name, value)
            else:
                self._append_array_dataset(value_dataset_name, value)
            if self.save_local_timestamps:
                self._append_array_dataset(timestamp_dataset_name, [timestamp, timestamp_offset])

    def _create_attributes_datasets(self):
        for key in self.attributes.keys():
            self.file.create_dataset(ATTRIBUTE_GROUP + key,data=self.attributes.get(key))

    def write_metadata(self, metadata):
        for name, value in metadata.items():

            if name not in self.cache:
                self.cache[name] = []

            self.cache[name].append(value)


    def close(self):
        if self.number_of_records >=0:
            if self.current_record != self.number_of_records:
                _logger.debug("Image dataset number of records set to=%s" % self.current_record)
                for dataset in list(self.scalar_datasets.values()) + list(self.array_datasets.values()):
                    dataset.resize(size=self.current_record, axis=0)
        self.file.close()
        self.scalar_datasets, self.array_datasets = {}, {}
        _logger.info("Writing completed.")

    def add_record(self, pulse_id, data, format_changed, global_timestamp, global_timestamp_offset):
        if (self.number_of_records >= 0) and (self.current_record >= self.number_of_records):
            raise Exception("HDF5 Writer reached the total number of records")
        if self.current_record == 0:
            self.create_header_datasets()
            self.create_channel_datasets(data)
        else:
            if format_changed:
                raise Exception("Data format changed")
        self.append_header(pulse_id, global_timestamp, global_timestamp_offset)
        self.append_channel_data(data)
        self.current_record += 1

    def start(self, stream, stream_mode=SUB):
        self.stream = stream
        try:
            stream_host, stream_port = get_host_port_from_stream_address(stream)
            with source(host=stream_host, port=stream_port, mode=stream_mode) as stream:
                while True:
                    if (self.number_of_records>=0) and (self.current_record >= self.number_of_records):
                        break
                    rec = stream.receive()

                    pulse_id, data, format_changed, global_timestamp, global_timestamp_offset = \
                        rec.data.pulse_id, rec.data.data, rec.data.format_changed, rec.data.global_timestamp, \
                        rec.data.global_timestamp_offset
                    self.add_record(pulse_id, data, format_changed, global_timestamp, global_timestamp_offset)

        finally:
            self.close()

class WriterSender(object):
    def __init__(self, output_file="/dev/null", number_of_records=UNDEFINED_NUMBER_OF_RECORDS,
                       layout = LAYOUT_DEFAULT, save_local_timestamps = LOCALTIME_DEFAULT, attributes={}):
        self.writer = Writer(output_file, number_of_records, layout, save_local_timestamps, attributes)
        self.stream=None

    def open(self, no_client_action=None, no_client_timeout=None):
        pass

    def send(self, data, timestamp, pulse_id):
        bsdata = {}
        for key in data.keys():
            bsdata[key] = Value(data[key],timestamp[0], timestamp[1])
        self.writer.add_record(pulse_id, bsdata, False, timestamp[0], timestamp[1])

    def close(self):
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Stream writer')
    parser.add_argument('-s', '--stream', default="tcp://localhost:5555", help="Stream to connect to")
    parser.add_argument('-t', '--type', default="SUB", help="Stream type")
    parser.add_argument('-f', '--filename', default='/dev/null', help="Output file")
    parser.add_argument('-r', '--records', default=UNDEFINED_NUMBER_OF_RECORDS, help="Number of records to write")
    parser.add_argument('-l', '--layout', default='DEFAULT', choices=[LAYOUT_DEFAULT, LAYOUT_FLAT], help="File layout")
    parser.add_argument('-e', '--localtime', default=str(LOCALTIME_DEFAULT), choices=['True', 'False'], help="Write channels local timestamps")
    parser.add_argument('-a', '--attributes', default="{}", help="User attribute dictionary to be written to file")
    parser.add_argument("--log_level", default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()
    logging.basicConfig(level=arguments.log_level)
    writer = Writer(arguments.filename, int(arguments.records), arguments.layout, arguments.localtime.lower() != "false", arguments.attributes)
    writer.start(arguments.stream, PULL if arguments.type == "PULL" else SUB)

if __name__ == "__main__":
    main()