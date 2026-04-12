from __future__ import annotations

import base64
import pickle
import zlib
from typing import Any


def encode_transport_payload(payload: Any) -> str:
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(serialized, level=3)
    return base64.b64encode(compressed).decode("ascii")


def decode_transport_payload(payload_b64: str) -> Any:
    compressed = base64.b64decode(str(payload_b64).encode("ascii"))
    serialized = zlib.decompress(compressed)
    return pickle.loads(serialized)
