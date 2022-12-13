def process(data, pulse_id, timestamp, params):
    ret = dict()
    prefix = params["prefix"]
    for c in data.keys():
        ret[prefix+c] = data[c]
    return ret