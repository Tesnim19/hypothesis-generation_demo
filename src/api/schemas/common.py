from __future__ import annotations

"""
API models use `src.utils.serialize_datetime_fields` before validating dynamic
responses (FlexibleDict / FlexibleList / concrete models with dict/list fields)
so wire JSON matches historical datetime encoding (ISO strings).
"""

from typing import Any

from pydantic import RootModel


class FlexibleDict(RootModel[dict[str, Any]]):
    """JSON object with arbitrary keys (Mongo documents, merged API payloads)."""


class FlexibleList(RootModel[list[Any]]):
    """JSON array with arbitrary elements."""
