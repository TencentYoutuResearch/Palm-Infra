#!/usr/bin/env python3
"""Read-only logical weight-payload audit of a v1 package's v3 decode graph.

Counts dense MATMUL payloads and routed/shared MOE projections per one-token
graph execution. This is not measured DRAM traffic: caching, rereads, other
ops and activations are deliberately not converted into a bandwidth claim.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def unpack(self, fmt):
        size = struct.calcsize('<' + fmt)
        if size > len(self.data) - self.offset:
            raise ValueError('truncated graph')
        result = struct.unpack_from('<' + fmt, self.data, self.offset)
        self.offset += size
        return result

    def string(self):
        size, = self.unpack('I')
        if size > len(self.data) - self.offset:
            raise ValueError('truncated graph string')
        text = self.data[self.offset:self.offset + size].decode()
        self.offset += size
        return text

    def vector(self, fmt):
        count, = self.unpack('I')
        if count > (len(self.data) - self.offset) // struct.calcsize('<' + fmt):
            raise ValueError('truncated graph vector')
        return list(self.unpack(str(count) + fmt))


def graph_nodes(data):
    reader = Reader(data)
    magic, version, count = reader.unpack('III')
    if (magic, version) != (0x4D4C4C47, 3):
        raise ValueError('expected graph v3')
    metadata_count, = reader.unpack('I')
    for _ in range(metadata_count):
        reader.string(); reader.string()
    reader.vector('i'); reader.vector('i')
    nodes = []
    for index in range(count):
        node_id, op = reader.unpack('iI')
        if node_id != index:
            raise ValueError('node ids are not serialized indices')
        inputs = reader.vector('i')
        if any(dependency < 0 or dependency >= index for dependency in inputs):
            raise ValueError('graph input is not a preceding node')
        shape = reader.unpack('4q')
        dim_expr = [reader.unpack('b3xi') for _ in range(4)]
        if any(kind not in range(6) for kind, _ in dim_expr):
            raise ValueError('invalid dynamic dimension kind')
        precision, = reader.unpack('I')
        integers = reader.vector('i')
        reader.vector('f')
        strings_count, = reader.unpack('I')
        strings = [reader.string() for _ in range(strings_count)]
        nodes.append(dict(id=index, op=op, inputs=inputs, shape=shape, dim_expr=dim_expr,
                          precision=precision, integers=integers, strings=strings))
    if reader.offset != len(data):
        raise ValueError('unexpected trailing graph data')
    return nodes


def audit(path):
    with path.open('rb') as source:
        size = source.seek(0, 2)
        source.seek(0)
        header = struct.unpack('<II15Q', source.read(128))
        if header[:2] != (0x4D4C4F4D, 1):
            raise ValueError('expected package v1')
        if header[12] > size or header[13] > size - header[12]:
            raise ValueError('invalid weights section')

        def section(offset, length):
            if offset > size or length > size - offset or length > 256 * 1024 * 1024:
                raise ValueError('invalid metadata/graph section')
            source.seek(offset)
            data = source.read(length)
            if len(data) != length:
                raise ValueError('truncated section')
            return data

        metadata = json.loads(section(header[2], header[3]))
        nodes = graph_nodes(section(header[10], header[11]))
        weights = {}
        for name, (offset, length) in metadata['weights'].items():
            if offset > header[13] or length > header[13] - offset or length < 88:
                raise ValueError('invalid weight span: ' + name)
            source.seek(header[12] + offset)
            weight = struct.unpack('<4I4Q4Q2I', source.read(88))
            if weight[0] != 0x50414D58 or weight[2] not in (1, 2, 3, 4):
                raise ValueError('invalid weight header: ' + name)
            data_offset, data_size, scales_offset, scales_size = weight[8:12]
            if any(begin > length or count > length - begin for begin, count in
                   ((data_offset, data_size), (scales_offset, scales_size))):
                raise ValueError('invalid payload: ' + name)
            payload = data_size + scales_size
            logical = payload
            if weight[3] == 3 and weight[1] & 4:
                if weight[5] % 32 or scales_size:
                    raise ValueError('unsupported BG32 contract: ' + name)
                logical = math.prod(weight[4:8]) // 32 * 20
                if logical > payload:
                    raise ValueError('BG32 logical payload exceeds storage: ' + name)
            weights[name] = dict(bytes=logical, stored_bytes=payload, precision=weight[3],
                                 shape=weight[4:8])

    def constant(index):
        node = nodes[index]
        if node['op'] == 1 and node['strings'] and node['strings'][0] in weights:
            return node['strings'][0]
        if node['op'] in (60, 61) and len(node['inputs']) == 1:
            return constant(node['inputs'][0])
        raise ValueError(f'cannot resolve projection weight node {index}')

    expert_sizes = {}
    for layer in metadata['moe_expert_storage']['layers']:
        for projection in ('gate_up', 'down'):
            entry = layer[projection]
            expert_sizes[entry['weight']] = entry['expert_data_bytes'] + entry['expert_scales_bytes']
    groups = defaultdict(lambda: dict(calls=0, bytes=0, associated_storage_bytes=0))
    details = []
    counted_names = set()

    def add(node, index, category, multiplier=1, expert=False):
        name = constant(index)
        payload = expert_sizes[name] if expert else weights[name]['bytes']
        total = multiplier * payload
        groups[category]['calls'] += 1
        groups[category]['bytes'] += total
        groups[category]['associated_storage_bytes'] += total if expert else multiplier * weights[name]['stored_bytes']
        counted_names.add(name)
        details.append(dict(node=node['id'], category=category, weight=name,
                            shape=weights[name]['shape'], precision=weights[name]['precision'],
                            multiplier=multiplier, bytes=total))

    for node in nodes:
        if node['op'] == 10:
            name = constant(node['inputs'][1])
            category = 'lm_head' if name == './lm_head.weights' else 'dense_matmul'
            add(node, node['inputs'][1], category)
        elif node['op'] == 120:
            if len(node['inputs']) != 8 or len(node['integers']) < 3:
                raise ValueError('unsupported MOE contract')
            top_k = node['integers'][2]
            for index in (1, 4, 5, 6, 7):
                add(node, node['inputs'][index], 'moe_router_shared')
            for index in (2, 3):
                add(node, node['inputs'][index], 'moe_routed', top_k, True)
    # Engine::sample_logits invokes the shared head outside the decode graph.
    head = './lm_head.weights'
    if head not in counted_names:
        if head not in weights:
            raise ValueError('package lacks externally invoked lm_head')
        info = weights[head]
        groups['lm_head'] = dict(calls=1, bytes=info['bytes'], associated_storage_bytes=info['stored_bytes'])
        counted_names.add(head)
        details.append(dict(node='external_sampler', category='lm_head', weight=head,
                            shape=info['shape'], precision=info['precision'], multiplier=1, bytes=info['bytes']))
    uncounted = Counter()
    for node in nodes:
        if node['op'] not in (1, 10, 120, 60, 61):
            for index in node['inputs']:
                dependency = nodes[index]
                if dependency['op'] == 1 and dependency['strings'] and dependency['strings'][0] in weights:
                    uncounted[str(node['op'])] += 1
    return dict(package=str(path), scope='logical projection payload per one-token decode; NOT measured DRAM bytes',
                decode_nodes=len(nodes), graph_op_counts=dict(Counter(str(n['op']) for n in nodes)),
                categories=dict(groups), logical_projection_bytes=sum(g['bytes'] for g in groups.values()),
                unique_projection_weights=len(counted_names), other_direct_constant_consumers=dict(uncounted),
                details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    result = audit(args.package)
    if not args.details:
        del result['details']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
