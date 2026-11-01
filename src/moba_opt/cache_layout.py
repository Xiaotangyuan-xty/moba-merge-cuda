"""CPU page allocator. Planning capacity failures never mutates live state."""
from dataclasses import dataclass
import numbers


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class AppendPlan:
    version: int
    tables: tuple
    lengths: tuple
    free_pages: tuple
    token_sequences: tuple
    token_positions: tuple
    completed_chunks: tuple


class PageAllocator:
    """Fixed batch slots; logical pages may map to arbitrary physical pages."""

    def __init__(self, batch_size, max_seq_len, page_size, num_pages, chunk_size):
        self.batch_size = positive_int(batch_size, "batch_size")
        self.max_seq_len = positive_int(max_seq_len, "max_seq_len")
        self.page_size = positive_int(page_size, "page_size")
        self.num_pages = positive_int(num_pages, "num_pages")
        self.chunk_size = positive_int(chunk_size, "chunk_size")
        if self.max_seq_len >= 2**30 or self.num_pages * self.page_size >= 2**31:
            raise ValueError("cache exceeds int32 indexing capacity")
        self.table_width = (self.max_seq_len + self.page_size - 1) // self.page_size
        self.tables = tuple((-1,) * self.table_width for _ in range(self.batch_size))
        self.lengths = (0,) * self.batch_size
        self.free_pages = tuple(range(self.num_pages - 1, -1, -1))
        self.version = 0

    def validate_ids(self, ids, *, unique=False):
        result = []
        for sid in ids:
            if isinstance(sid, bool) or not isinstance(sid, numbers.Integral):
                raise ValueError("sequence IDs must be integers")
            sid = int(sid)
            if not 0 <= sid < self.batch_size:
                raise ValueError("sequence ID outside cache batch slots")
            result.append(sid)
        if not result or (unique and len(set(result)) != len(result)):
            raise ValueError("nonempty unique sequence IDs required for append/reset")
        return tuple(result)

    def plan_append(self, sequence_ids, counts):
        ids = self.validate_ids(sequence_ids, unique=True)
        if len(ids) != len(counts):
            raise ValueError("one token count per sequence required")
        counts = [positive_int(n, "append count") for n in counts]
        required = 0
        for sid, n in zip(ids, counts):
            new_len = self.lengths[sid] + n
            if new_len > self.max_seq_len:
                raise ValueError("append exceeds max_seq_len")
            required += ((new_len + self.page_size - 1) // self.page_size
                         - (self.lengths[sid] + self.page_size - 1) // self.page_size)
        if required > len(self.free_pages):
            raise ValueError("KV page pool exhausted; cache was not modified")
        tables = [list(row) for row in self.tables]
        lengths = list(self.lengths)
        free = list(self.free_pages)
        token_ids, positions, completed = [], [], []
        for sid, n in zip(ids, counts):
            old, new = lengths[sid], lengths[sid] + n
            old_pages = (old + self.page_size - 1) // self.page_size
            new_pages = (new + self.page_size - 1) // self.page_size
            for logical in range(old_pages, new_pages):
                tables[sid][logical] = free.pop()
            token_ids.extend([sid] * n)
            positions.extend(range(old, new))
            completed.extend((sid, c) for c in range(old // self.chunk_size, new // self.chunk_size))
            lengths[sid] = new
        return AppendPlan(self.version, tuple(map(tuple, tables)), tuple(lengths), tuple(free),
                          tuple(token_ids), tuple(positions), tuple(completed))

    def commit(self, plan):
        if plan.version != self.version:
            raise RuntimeError("stale append plan")
        self.tables, self.lengths, self.free_pages = plan.tables, plan.lengths, plan.free_pages
        self.version += 1

    def reset(self, sequence_ids):
        ids = self.validate_ids(sequence_ids, unique=True)
        tables = [list(row) for row in self.tables]
        lengths, free = list(self.lengths), list(self.free_pages)
        for sid in ids:
            free.extend(p for p in tables[sid] if p >= 0)
            tables[sid] = [-1] * self.table_width
            lengths[sid] = 0
        self.tables, self.lengths, self.free_pages = tuple(map(tuple, tables)), tuple(lengths), tuple(free)
        self.version += 1
