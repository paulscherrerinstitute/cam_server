import json
import logging
import numpy
import traceback
from logging import getLogger

from bsread.data.helpers import get_channel_reader, get_value_reader
from bsread.handlers.compact import Message, Value
from bsread.data.serialization import channel_type_deserializer_mapping, \
    compression_provider_mapping, channel_type_scalar_serializer_mapping

_logger = getLogger(__name__)


def get_value_reader(channel_type, compression, shape=None, endianness="", value_name=None):
    """
    Get the correct value reader for the specific channel type and compression.
    :param channel_type: Channel type.
    :param compression: Compression on the channel.
    :param shape: Shape of the data.
    :param endianness: Encoding of the channel: < (small endian) or > (big endian)
    :param value_name: Name of the value to decode. For logging.
    :return: Object capable of reading the data, when get_value() is called on it.
    """
    # If the type is unknown, NoneProvider should be used.
    if channel_type not in channel_type_deserializer_mapping:
        _logger.warning("Channel type '%s' not found in mapping." % channel_type)
        # If the channel is not supported, always return None.
        return lambda x: None

    # If the compression is unknown, NoneProvider should be used.
    if compression not in compression_provider_mapping:
        _logger.warning("Channel compression '%s' not supported." % compression)
        # If the channel compression is not supported, always return None.
        return lambda x: None

    decompressor = compression_provider_mapping[compression].unpack_data
    dtype, serializer = channel_type_deserializer_mapping[channel_type]
    # Expand the dtype with the correct endianess.
    dtype = endianness + dtype

    def value_reader(raw_data):
        try:
            # Decompress and deserialize the received value.
            if raw_data:
                numpy_array = decompressor(raw_data, dtype, shape)
                return serializer(numpy_array)
            else:
                return None

        except Exception as e:
            # We do not want to throw exceptions in case we cannot decode a channel.
            _logger.warning("Unable to decode value_name '%s' - returning None. Exception: %s",
                            value_name, traceback.format_exc())

            _logger.info("Decoding failed value name '%s' with dtype='%s', shape='%s' "
                         "compression='%s', raw_data_length='%s' and raw_data='%s'. Exception: %s",
                         value_name, channel_type, shape, compression, len(raw_data), raw_data, e)
            return None

    return value_reader



def get_channel_reader(channel):
    """
    Construct a value reader for the provided channel.
    :param channel: Channel to construct the value reader for.
    :return: Value reader.
    """
    # If no channel type is specified, float64 is assumed.
    channel_type = channel['type'].lower() if 'type' in channel else None
    if channel_type is None:
        _logger.warning("'type' channel field not found. Parse as 64-bit floating-point number float64 (default).")
        channel_type = "float64"

    name = channel['name']
    compression = channel['compression'] if "compression" in channel else None
    shape = channel['shape'] if "shape" in channel else None
    endianness = channel['encoding']

    value_reader = get_value_reader(channel_type, compression, shape, endianness, name)
    return value_reader


class Handler:
    def __init__(self, data_change_callback=None):
        # Used for detecting if the data header has changed - we need to reconstruct the channel definitions.
        self.data_header_hash = None
        self.channels_definitions = None
        self.data_change_callback = data_change_callback

    def set_data_change_callback(self, callback):
        self.data_change_callback = callback

    def receive(self, receiver):
        # Receive main header
        header = receiver.next(as_json=True)
        changed = False

        # We cannot process an empty Header.
        if not header:
            return None

        message = Message()
        message.pulse_id = header['pulse_id']
        message.hash = header['hash']

        if 'global_timestamp' in header:
            if 'sec' in header['global_timestamp']:
                message.global_timestamp = header['global_timestamp']['sec']
            elif 'epoch' in header['global_timestamp']:
                message.global_timestamp = header['global_timestamp']['epoch']
            else:
                raise RuntimeError("Invalid timestamp format in BSDATA header message {}".format(message))

            message.global_timestamp_offset = header['global_timestamp']['ns']

        # Receiver data header, check if header has changed - and in this case recreate the channel definitions.
        if receiver.has_more() and (self.data_header_hash != header['hash']):
            changed = self.data_header_hash is not None

            # Set the current header hash as the new hash.
            self.data_header_hash = header['hash']

            # Read the data header.
            data_header_bytes = receiver.next()
            data_header = json.loads(get_value_reader("string", header.get('dh_compression'),
                                                      value_name="data_header")(data_header_bytes))

            # If a message with ho channel information is received,
            # ignore it and return from function with no data.
            if not data_header['channels']:
                logging.warning("Received message without channels.")
                while receiver.has_more():
                    # Drain rest of the messages - if entering this code there is actually something wrong
                    receiver.next()

                return message

            # TODO: Why do we need to pre-process the message? Source change?
            for channel in data_header['channels']:
                # Define endianness of data
                # > - big endian
                # < - little endian (default)
                channel["encoding"] = '>' if channel.get("encoding") == "big" else '<'

            # Construct the channel definitions.
            self.channels_definitions = [(channel["name"], channel["encoding"], get_channel_reader(channel))
                                         for channel in data_header['channels']]

            # Signal that the format has changed.
            message.format_changed = True
        else:
            # Skip second header - we already have the receive functions setup.
            receiver.next()

        # Receiving data
        counter = 0

        # Todo add some more error checking
        while receiver.has_more():
            channel_name, channel_endianness, channel_reader = self.channels_definitions[counter]

            raw_data = receiver.next()
            channel_value = Value()

            if raw_data:
                channel_value.value = channel_reader(raw_data)

                if receiver.has_more():

                    raw_timestamp = receiver.next()

                    if raw_timestamp:
                        timestamp_array = numpy.frombuffer(raw_timestamp, dtype=channel_endianness + 'u8')
                        channel_value.timestamp = timestamp_array[0]  # Second past epoch
                        channel_value.timestamp_offset = timestamp_array[1]  # Nanoseconds offset
            else:
                # Consume empty timestamp message
                if receiver.has_more():
                    receiver.next()  # Read empty timestamp message

            message.data[channel_name] = channel_value
            counter += 1

        if changed:
            if self.data_change_callback is not None:
                self.data_change_callback(data_header['channels'])

        return message

