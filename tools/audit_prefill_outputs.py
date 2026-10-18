#!/usr/bin/env python3
"""Read only a package's prefill graph; estimate declared output storage.

This is not measured memory, nor a safe fit guarantee. Excludes weights,
caches, scratch, fusion and pool retention; counts possible materialized views.
Conservative logical peak simulates the experimental planner. Pool figures
simulate exact-size best-fit reuse for the declared materialization scenario,
not actual device execution, a memory upper bound or a safe-fit guarantee.
"""
import argparse
import bisect
from collections import defaultdict
import math
import struct
from audit_decode_weight_traffic import Reader, graph_nodes


def simulate_output_pool(sizes, releases, free=None):
    """Simulate CUDA best-fit allocation with no actual storage allocation.

    Released buffers retain their original allocation size, not the size of
    the latest tensor borrowing them. End-of-call cleanup returns all retained
    outputs. Passing the returned free list models a subsequent changed shape.
    """
    free = sorted(free or [])
    if any(size <= 0 for size in free) or any(size < 0 for size in sizes):
        raise ValueError('invalid simulated allocation size')
    reserved = sum(free)
    active = peak = 0
    allocations = {}
    for index, size in enumerate(sizes):
        if size:
            position = bisect.bisect_left(free, size)
            if position < len(free):
                allocation = free.pop(position)
            else:
                allocation = size
                reserved += allocation
            allocations[index] = allocation
            active += allocation
            peak = max(peak, active)
        for released in releases.get(index, []):
            # Zero-size nodes represent non-owning views/constants.
            if sizes[released] == 0:
                continue
            if released not in allocations:
                raise ValueError('invalid simulated release')
            allocation = allocations.pop(released)
            active -= allocation
            bisect.insort(free, allocation)
    for allocation in allocations.values():
        bisect.insort(free, allocation)
    if sum(free) != reserved:
        raise ValueError('invalid simulated pool accounting')
    return reserved, peak, free


def gr_inject_consumers(nodes, outputs):
    """Direct graph-edge eligibility only; not a runtime fusion safety proof."""
    consumers = defaultdict(list)
    for node in nodes:
        for dependency in node['inputs']:
            consumers[dependency].append(node)
    records = []
    for node in nodes:
        if node['op'] != 134:
            continue
        users = consumers[node['id']]
        records.append(dict(id=node['id'], consumers=[user['id'] for user in users],
                            ops=[user['op'] for user in users],
                            rms=[user['id'] for user in users if user['op'] == 25],
                            graph_output=node['id'] in outputs,
                            removable=(len(users) == 1 and users[0]['op'] == 25 and
                                       node['id'] not in outputs)))
    return records


def stateful_protected_ids(node, recurrent_contracts=False, cuda_contracts=False):
    stateful = {22, 23, 24, 50, 51, 110, 111, 112, 135, 137, 140, 160, 161, 162}
    if node['op'] not in stateful:
        return []
    if cuda_contracts and node.get('precision') == 0:
        # Mirror CudaBackend::independent_stateful_mutable_inputs with decode
        # segments disabled. No implication of capture/shape/fusion safety.
        if node['op'] == 24 and len(node['inputs']) == 6:
            return []  # QK_RMS_NORM_ROPE writes only its independent output.
        if node['op'] == 50 and len(node['inputs']) in (6, 7):
            return [node['inputs'][4], node['inputs'][5]]  # Persistent K/V.
    if (recurrent_contracts or cuda_contracts) and node.get('precision') == 0:
        mutable = {140: (3, 2), 111: (8, 7)}.get(node['op'])
        if mutable and len(node['inputs']) == mutable[0]:
            return [node['inputs'][mutable[1]]]
    return [node['id'], *node['inputs']]


def dense_projection_shapes(nodes):
    """Declared constant-weight shapes, not runtime dispatch/fusion counts."""
    counts = defaultdict(int)
    unresolved = 0
    for node in nodes:
        if node['op'] == 10:
            category, indices = 'matmul', (1,)
        elif node['op'] == 120:
            category, indices = 'moe_router_shared', (1, 4, 5, 6, 7)
        else:
            continue
        for index in indices:
            if index >= len(node['inputs']):
                unresolved += 1
                continue
            weight = nodes[node['inputs'][index]]
            shape = weight.get('shape', ())
            dynamic = weight.get('dim_expr', ())
            if (weight['op'] != 1 or len(shape) != 4 or shape[0] <= 0 or shape[1] <= 0 or
                    shape[2:] != (1, 1) or len(dynamic) != 4 or any(kind for kind, _ in dynamic)):
                unresolved += 1
                continue
            counts[(category, shape[0], shape[1])] += 1
    return counts, unresolved


def leading_split_strides(shape, strides, target, element_size):
    """Exact address-preserving 2D row split; not a general reshape algorithm."""
    if (len(shape) != 4 or len(strides) != 4 or len(target) != 4 or
            element_size <= 0 or any(d <= 0 for d in (*shape, *target)) or
            any(s <= 0 for s in strides) or strides[0] != element_size or
            shape[2:] != (1, 1) or target[3] != 1 or
            target[0] * target[1] != shape[0] or target[2] != shape[1]):
        return None
    return (element_size, element_size * target[0], strides[1],
            strides[1] * target[2])


def declared_layout_copies(nodes, sequence):
    """Mirror unfused dense-output/view layout rules, NOT runtime event counts.

    Non-layout producers are assumed dense. Unknown precisions stay unknown;
    do not infer storage layouts for packed constants. The proposed split is
    reported only: it is NOT propagated into subsequent simulated layouts.
    """
    if sequence <= 0:
        raise ValueError('nonpositive sequence')
    layouts, records = [], []
    consumers = defaultdict(list)
    for node in nodes:
        for dependency in node['inputs']:
            consumers[dependency].append(node['id'])
    for node in nodes:
        shape = []
        for static, (kind, coeff) in zip(node['shape'], node['dim_expr']):
            if kind == 5 and (coeff <= 0 or sequence % coeff):
                raise ValueError('non-divisible dynamic dimension')
            shape.append(static if kind == 0 else sequence if kind == 1 else
                         coeff * sequence if kind == 2 else coeff + sequence if kind == 3
                         else 1 if kind == 4 else sequence // coeff)
        shape = tuple(shape)
        if len(shape) != 4 or any(d <= 0 for d in shape):
            raise ValueError('invalid resolved shape')
        es = {0: 4, 1: 2, 2: 1}.get(node['precision'])
        dense = None if es is None else (es, es * shape[0],
                                        es * shape[0] * shape[1],
                                        es * shape[0] * shape[1] * shape[2])
        stride = dense
        op = node['op']
        if op in (60, 61, 63, 65):
            if len(node['inputs']) != 1:
                raise ValueError('layout node requires one input')
            parent = node['inputs'][0]
            source_shape, source_stride = layouts[parent]
            if (source_stride is None or es is None or
                    nodes[parent]['precision'] != node['precision']):
                stride = None
            elif op == 61:
                axes = node['integers'][:4]
                if sorted(axes) != [0, 1, 2, 3]:
                    raise ValueError('invalid permutation')
                result = [0] * 4
                for d, axis in enumerate(axes):
                    if shape[axis] != source_shape[d]:
                        raise ValueError('permutation shape mismatch')
                    result[axis] = source_stride[d]
                stride = tuple(result)
            elif op == 63:
                stride = source_stride
            else:
                if math.prod(shape) != math.prod(source_shape):
                    raise ValueError('reshape element count mismatch')
                expected, contiguous = es, source_stride[0] == es
                for d in range(4):
                    if source_shape[d] != 1 and source_stride[d] != expected:
                        contiguous = False
                    expected *= source_shape[d]
                if not contiguous:
                    split = (leading_split_strides(source_shape, source_stride, shape, es)
                             if op == 60 else None)
                    records.append(dict(id=node['id'], op=op, parent=parent,
                                        parent_op=nodes[parent]['op'], shape=shape,
                                        bytes=math.prod(shape) * es,
                                        consumers=consumers[node['id']],
                                        leading_split_candidate=split is not None))
        layouts.append((shape, stride))
    return records


def audit(path, sequences, inspect_gr=False, recurrent_contracts=False, shapes_only=False,
          cuda_contracts=False, layout_only=False):
    with open(path, 'rb') as source:
        size = source.seek(0, 2)
        source.seek(0)
        raw = source.read(128)
        if len(raw) != 128:
            raise ValueError('truncated package header')
        header = struct.unpack('<II15Q', raw)
        if header[:2] != (0x4D4C4F4D, 1):
            raise ValueError('expected package v1')
        offset, length = header[8:10]
        if offset > size or length > size - offset or length > 256 * 1024**2:
            raise ValueError('invalid prefill section')
        source.seek(offset)
        data = source.read(length)
        nodes = graph_nodes(data)
    if layout_only:
        for sequence in sequences:
            records = declared_layout_copies(nodes, sequence)
            candidates = [r for r in records if r['leading_split_candidate']]
            print(f'declared_layout seq={sequence} copies={len(records)} '
                  f'bytes={sum(r["bytes"] for r in records)} '
                  f'leading_split_candidates={len(candidates)} '
                  f'candidate_bytes={sum(r["bytes"] for r in candidates)} '
                  'scope=unfused_dense_producer_assumption_not_runtime_or_fusion_safety')
            for record in records:
                print('layout_copy ' + ' '.join(f'{k}={v}' for k, v in record.items()))
        return
    if shapes_only:
        counts, unresolved = dense_projection_shapes(nodes)
        print(f'declared_projection_shapes prefill_nodes={len(nodes)} unresolved={unresolved} '
              'scope=static_graph_not_runtime_dispatch_or_fusion_counts')
        for (category, n, k), count in sorted(counts.items()):
            print(f'projection_shape category={category} n={n} k={k} nodes={count}')
        return
    reader = Reader(data)
    reader.unpack('III')
    metadata_count, = reader.unpack('I')
    for _ in range(metadata_count):
        reader.string(); reader.string()
    reader.vector('i')
    outputs = reader.vector('i')
    if inspect_gr:
        records = gr_inject_consumers(nodes, outputs)
        print(f'gr_inject_nodes={len(records)} '
              f'with_rms_consumer={sum(bool(record["rms"]) for record in records)} '
              f'single_rms_nonoutput={sum(record["removable"] for record in records)}')
        for record in records:
            print('gr_inject_edge ' + ' '.join(f'{key}={value}' for key, value in record.items()))
    roots, last = [], [-1] * len(nodes)
    protected = set(outputs)
    for node in nodes:
        index = node['id']
        roots.append(roots[node['inputs'][0]] if node['op'] in (60, 61, 63, 65)
                     and node['inputs'] else index)
        for dependency in node['inputs']:
            last[dependency] = max(last[dependency], index)
            last[roots[dependency]] = max(last[roots[dependency]], index)
        protected.update(stateful_protected_ids(node, recurrent_contracts, cuda_contracts))
    protected.update(roots[index] for index in tuple(protected))
    # Potential views can materialize: retain every intermediate parent, not
    # merely the earliest assumed owner. Mirror the experimental executor.
    for node in reversed(nodes):
        if node['op'] in (60, 61, 63, 65) and node['inputs']:
            index, parent = node['id'], node['inputs'][0]
            last[parent] = max(last[parent], last[index])
            if index in protected:
                protected.add(parent)
    warm_free = []
    for seq in sequences:
        totals = defaultdict(int)
        sizes = [0] * len(nodes)
        for node in nodes:
            if node['op'] in (0, 1, 61, 63):
                continue  # INPUT/CONSTANT and always-borrowed PERMUTE/SLICE
            shape = []
            for static, (kind, coeff) in zip(node['shape'], node['dim_expr']):
                if kind == 5 and (coeff <= 0 or seq % coeff):
                    raise ValueError('non-divisible dynamic dimension')
                shape.append(static if kind == 0 else seq if kind == 1 else
                             coeff * seq if kind == 2 else coeff + seq if kind == 3
                             else 1 if kind == 4 else seq // coeff)
            if any(dim <= 0 for dim in shape):
                raise ValueError('nonpositive resolved dimension')
            if node['precision'] not in (0, 1, 2):
                raise ValueError('unsupported output precision')
            sizes[node['id']] = math.prod(shape) * (4, 2, 1)[node['precision']]
            totals[node['op']] += sizes[node['id']]
        releases = defaultdict(list)
        for index in range(len(nodes)):
            if index not in protected:
                releases[max(index, last[index])].append(index)
        active = peak = 0
        for index, count in enumerate(sizes):
            active += count
            peak = max(peak, active)
            for released in releases[index]:
                active -= sizes[released]
        if not 0 <= active <= peak <= sum(sizes):
            raise ValueError('invalid simulated liveness accounting')
        cold_reserved, cold_active, _ = simulate_output_pool(sizes, releases)
        warm_reserved, warm_active, warm_free = simulate_output_pool(sizes, releases, warm_free)
        protected_totals = defaultdict(int)
        for index in protected:
            protected_totals[nodes[index]['op']] += sizes[index]
        print(f'seq={seq} declared_outputs_bytes={sum(totals.values())} '
              f'declared_outputs_GiB={sum(totals.values()) / 1024**3:.6f}')
        scenario = 'cuda_last_use_contract' if cuda_contracts else 'recurrent_contract' if recurrent_contracts else 'conservative'
        print(f'  {scenario}_logical_peak_GiB={peak / 1024**3:.6f} '
              f'protected_outputs_GiB={sum(sizes[i] for i in protected) / 1024**3:.6f}')
        print(f'  scenario_cold_pool_reserved_GiB={cold_reserved / 1024**3:.6f} '
              f'scenario_cold_pool_active_peak_GiB={cold_active / 1024**3:.6f}')
        print(f'  scenario_sequential_pool_reserved_GiB={warm_reserved / 1024**3:.6f} '
              f'scenario_sequential_pool_active_peak_GiB={warm_active / 1024**3:.6f}')
        for op, count in sorted(protected_totals.items(), key=lambda x: -x[1])[:5]:
            print(f'  protected_op={op} bytes={count} GiB={count / 1024**3:.6f}')
        for op, count in sorted(totals.items(), key=lambda x: -x[1])[:8]:
            print(f'  op={op} bytes={count} GiB={count / 1024**3:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('--seq', type=int, nargs='+', default=[256, 1024, 2048, 4096])
    parser.add_argument('--gr-inject-consumers', action='store_true',
                        help='inspect direct consumers before considering GR/RMS fusion')
    parser.add_argument('--dense-shapes-only', action='store_true',
                        help='count declared static projection shapes without loading weight payloads; NOT runtime dispatch counts')
    parser.add_argument('--layout-copies-only', action='store_true',
                        help='inspect declared view materializations assuming dense producers; NOT runtime counts or safe alias/lifetime proof')
    parser.add_argument('--recurrent-last-use-contracts', action='store_true',
                        help='hypothetical SHORTCONV/GDN_PREFILL lifetime contracts; NOT measured memory or a fit guarantee')
    parser.add_argument('--cuda-last-use-contracts', action='store_true',
                        help='mirror current CUDA FP32 QK_ROPE/SDPA/SHORTCONV/GDN_PREFILL last-use contracts with decode segments disabled; NOT fusion, measured memory, a cap override or safe-fit proof')
    args = parser.parse_args()
    if any(seq <= 0 for seq in args.seq):
        parser.error('sequence lengths must be positive')
    audit(args.package, args.seq, args.gr_inject_consumers, args.recurrent_last_use_contracts,
          args.dense_shapes_only, args.cuda_last_use_contracts, args.layout_copies_only)
