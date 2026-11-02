"""moba_opt: MoBA sparse-attention GPU optimization workspace (week 1: scaffolding)."""

from . import baseline_adapter  # noqa: F401  (registers "official")
from . import mixed_attention  # noqa: F401  (registers "opt_merge_atomic")
from . import grouped_attention  # noqa: F401  (native GQA/MQA and repeat reference)
from .kv_cache import ContiguousKVCache, PagedKVCache  # noqa: F401
from .prepared_decode import PreparedDecode  # noqa: F401
