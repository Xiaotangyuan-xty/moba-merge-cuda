"""Independent CPU FP64 attention for explicit cached token positions.

Accepts logical per-sequence K/V and optionally actual selected MoBA chunks.
It knows nothing about GPU page layouts, scatter kernels, or online softmax.
"""
import torch


def decode_reference(q, keys, values, sequence_ids, positions, *, chunk_size, selected_blocks=None):
    qc = q.detach().cpu().double()
    kc = [k.detach().cpu().double() for k in keys]
    vc = [v.detach().cpu().double() for v in values]
    route = None if selected_blocks is None else selected_blocks.detach().cpu().long()
    nq, hq, d = qc.shape
    hk = kc[sequence_ids[0]].shape[1]
    groups = hq // hk
    outputs, lses = [], []
    for row, (sid, pos) in enumerate(zip(sequence_ids, positions)):
        hs, ls = [], []
        for head in range(hq):
            if route is None:
                tokens = list(range(pos + 1))
            else:
                chunks = [int(x) for x in route[row, head].tolist() if x >= 0]
                current = pos // chunk_size
                if not chunks or chunks[0] != current or len(set(chunks)) != len(chunks):
                    raise ValueError("route must include current chunk exactly once")
                if any(c >= current for c in chunks[1:]):
                    raise ValueError("route sees a future/nonhistorical chunk")
                tokens = sorted(t for c in chunks
                                for t in range(c * chunk_size, min((c+1)*chunk_size, pos+1)))
            idx = torch.tensor(tokens, dtype=torch.long)
            logits = (kc[sid][idx, head // groups] * qc[row, head]).sum(-1) * d**-0.5
            ls.append(torch.logsumexp(logits, 0))
            hs.append(torch.softmax(logits, 0) @ vc[sid][idx, head // groups])
        outputs.append(torch.stack(hs))
        lses.append(torch.stack(ls))
    return torch.stack(outputs), torch.stack(lses)
