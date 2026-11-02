"""Inference-only Triton kernels: scatter append, closed-block means, paged SDPA.

Attention reads the physical page table directly. No dense KV reconstruction
or query-head KV replication occurs on the attention execution path.
"""
import triton
import triton.language as tl


@triton.jit
def _append_kernel(K, V, KC, VC, TABLE, IDS, POS,
                   HK: tl.constexpr, D: tl.constexpr, DP: tl.constexpr,
                   PAGE: tl.constexpr, WIDTH: tl.constexpr):
    token = tl.program_id(0) // HK
    head = tl.program_id(0) % HK
    d = tl.arange(0, DP)
    sid = tl.load(IDS + token)
    pos = tl.load(POS + token)
    physical = tl.load(TABLE + sid.to(tl.int64) * WIDTH + pos // PAGE)
    dst = ((physical.to(tl.int64) * PAGE + pos % PAGE) * HK + head) * D + d
    src = (token.to(tl.int64) * HK + head) * D + d
    tl.store(KC + dst, tl.load(K + src, mask=d < D, other=0), mask=d < D)
    tl.store(VC + dst, tl.load(V + src, mask=d < D, other=0), mask=d < D)


@triton.jit
def _closed_means_kernel(KC, TABLE, MEANS, IDS, CHUNKS,
                         HK: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                         WIDTH: tl.constexpr, C: tl.constexpr, CP: tl.constexpr,
                         MAX_CHUNKS: tl.constexpr):
    feature_tiles = tl.cdiv(D, 32)
    item = tl.program_id(0) // (HK * feature_tiles)
    head = (tl.program_id(0) // feature_tiles) % HK
    feature_tile = tl.program_id(0) % feature_tiles
    sid, chunk = tl.load(IDS + item), tl.load(CHUNKS + item)
    token = chunk * C + tl.arange(0, CP)
    d = feature_tile * 32 + tl.arange(0, 32)
    valid_t = tl.arange(0, CP) < C
    physical = tl.load(TABLE + sid.to(tl.int64) * WIDTH + token // PAGE, mask=valid_t, other=0)
    address = ((physical.to(tl.int64) * PAGE + token % PAGE)[:, None] * HK + head) * D + d[None, :]
    keys = tl.load(KC + address, mask=valid_t[:, None] & (d[None, :] < D), other=0).to(tl.float32)
    means = tl.sum(keys, axis=0) / C
    dst = ((sid.to(tl.int64) * MAX_CHUNKS + chunk) * HK + head) * D + d
    tl.store(MEANS + dst, means, mask=d < D)


@triton.jit
def _route_kernel(Q, MEANS, QSID, QPOS, SCORES,
                  NQ: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                  D: tl.constexpr, DP: tl.constexpr, C: tl.constexpr,
                  MAX_CHUNKS: tl.constexpr, NC: tl.constexpr, BC: tl.constexpr):
    head = tl.program_id(1) % HQ
    query = tl.program_id(1) // HQ
    sid, pos = tl.load(QSID + query), tl.load(QPOS + query)
    chunk = tl.program_id(0) * BC + tl.arange(0, BC)
    legal = (chunk < NC) & ((chunk + 1) * C <= pos)
    d = tl.arange(0, DP)
    kvhead = head // (HQ // HK)
    q = tl.load(Q + (query.to(tl.int64) * HQ + head) * D + d, mask=d < D, other=0).to(tl.float32)
    address = ((sid.to(tl.int64) * MAX_CHUNKS + chunk[:, None]) * HK + kvhead) * D + d[None, :]
    mean = tl.load(MEANS + address, mask=legal[:, None] & (d[None, :] < D), other=0).to(tl.float32)
    score = tl.sum(q[None, :] * mean, axis=1)
    tl.store(SCORES + (query.to(tl.int64) * HQ + head) * NC + chunk,
             tl.where(legal, score, -float("inf")), mask=chunk < NC)


@triton.jit
def _attention_kernel(Q, KC, VC, TABLE, LENGTHS, QSID, QPOS, SELECTED, OUT, LSE,
                      HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                      DP: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr,
                      C: tl.constexpr, SLOTS: tl.constexpr, DENSE: tl.constexpr,
                      BT: tl.constexpr):
    query, head = tl.program_id(0) // HQ, tl.program_id(0) % HQ
    sid, pos = tl.load(QSID + query), tl.load(QPOS + query)
    length = tl.load(LENGTHS + sid)
    kvhead = head // (HQ // HK)
    d = tl.arange(0, DP)
    q = tl.load(Q + (query.to(tl.int64) * HQ + head) * D + d, mask=d < D, other=0).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    z = tl.full((), 0, tl.float32)
    acc = tl.full((DP,), 0, tl.float32)
    if DENSE:
        iterations = tl.cdiv(pos + 1, BT)
    else:
        iterations = SLOTS * tl.cdiv(C, BT)
    for iteration in range(iterations):
        if DENSE:
            token = iteration * BT + tl.arange(0, BT)
            valid = (token <= pos) & (token < length)
        else:
            slot = iteration // tl.cdiv(C, BT)
            tile = iteration % tl.cdiv(C, BT)
            chunk = tl.load(SELECTED + (query.to(tl.int64) * HQ + head) * SLOTS + slot)
            within = tile * BT + tl.arange(0, BT)
            token = chunk * C + within
            valid = (chunk >= 0) & (within < C) & (token <= pos) & (token < length)
        physical = tl.load(TABLE + sid.to(tl.int64) * WIDTH + token // PAGE, mask=valid, other=0)
        address = ((physical.to(tl.int64) * PAGE + token % PAGE)[:, None] * HK + kvhead) * D + d[None, :]
        keys = tl.load(KC + address, mask=valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * (D ** -0.5)
        scores = tl.where(valid, scores, -float("inf"))
        has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
        next_m = tl.maximum(m, tl.max(scores, axis=0))
        safe_m = tl.where(has_valid, next_m, 0.0)
        alpha = tl.where(has_valid, tl.exp(m - safe_m), 1.0)
        weights = tl.where(valid, tl.exp(scores - safe_m), 0.0)
        values = tl.load(VC + address, mask=valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        acc = acc * alpha + tl.sum(weights[:, None] * values, axis=0)
        z = z * alpha + tl.sum(weights, axis=0)
        m = tl.where(has_valid, next_m, m)
    tl.store(OUT + (query.to(tl.int64) * HQ + head) * D + d, acc / z, mask=d < D)
    tl.store(LSE + query.to(tl.int64) * HQ + head, m + tl.log(z))


def append_tokens(k, v, kc, vc, table, ids, positions):
    n, hk, d = k.shape
    _append_kernel[(n * hk,)](k, v, kc, vc, table, ids, positions,
                            hk, d, triton.next_power_of_2(d), kc.shape[1], table.shape[1])


def update_closed_means(kc, table, means, ids, chunks, chunk_size):
    if ids.numel():
        hk, d = kc.shape[2:]
        _closed_means_kernel[(ids.numel() * hk * triton.cdiv(d, 32),)](
            kc, table, means, ids, chunks, hk, d, kc.shape[1], table.shape[1],
            chunk_size, triton.next_power_of_2(chunk_size), means.shape[1],
            num_warps=8 if chunk_size >= 512 else 4)


def route_scores(q, means, query_ids, positions, chunk_size, scores):
    nq, hq, d = q.shape
    count = scores.shape[-1]
    _route_kernel[(triton.cdiv(count,8),nq*hq)](
        q, means, query_ids, positions, scores,
        nq, hq, means.shape[2], d, triton.next_power_of_2(d), chunk_size, means.shape[1],count,8)


def paged_attention(q, kc, vc, table, lengths, ids, positions, selected, out, lse,
                    chunk_size, dense):
    nq, hq, d = q.shape
    _attention_kernel[(nq * hq,)](
        q, kc, vc, table, lengths, ids, positions, selected, out, lse,
        hq, kc.shape[2], d, triton.next_power_of_2(d), kc.shape[1], table.shape[1],
        chunk_size, 1 if dense else selected.shape[-1], dense, 32, num_warps=4)
