"""Share link schemas.

Shared-snapshot models double as the whitelist: undeclared fields are dropped
when a snapshot is built, so internal ids never reach the public payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints, computed_field


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ShareLinkCreateRequest(BaseModel):
    hypothesis_id: str
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] | None = None
    expires_in_days: int | None = Field(
        30, ge=1, le=365, description="Days until the link expires; null never expires."
    )


class ShareLinkResponse(BaseModel):
    id: str
    token: str
    hypothesis_id: str
    label: str | None = None
    created_at: UtcDatetime
    expires_at: UtcDatetime | None = None
    revoked_at: UtcDatetime | None = None
    view_count: int = 0
    last_viewed_at: UtcDatetime | None = None

    @computed_field
    @property
    def url_path(self) -> str:
        return f"/s/{self.token}"

    @computed_field
    @property
    def state(self) -> Literal["active", "expired", "revoked"]:
        if self.revoked_at:
            return "revoked"
        if self.expires_at and self.expires_at <= datetime.now(timezone.utc):
            return "expired"
        return "active"


class ShareLinkListResponse(BaseModel):
    links: list[ShareLinkResponse]


class SharedGraphNode(BaseModel):
    id: str
    type: str | None = None
    name: str | None = None


class SharedGraphEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class SharedGraph(BaseModel):
    nodes: list[SharedGraphNode] = Field(default_factory=list)
    edges: list[SharedGraphEdge] = Field(default_factory=list)
    probability: float | None = None


class SharedGoTerm(BaseModel):
    id: str
    name: str | None = None
    p: float | None = None
    rank: int | None = None
    genes: list[str] = Field(default_factory=list)


class SharedWarning(BaseModel):
    code: str | None = None
    message: str | None = None


class SharedHypothesis(BaseModel):
    key: str
    causal_gene: str | None = None
    probability: float | None = None
    status: str
    summary: str | None = None
    graph: SharedGraph | None = None
    selected_go_term: SharedGoTerm | None = None
    go_terms: list[SharedGoTerm] = Field(default_factory=list)


class SharedMainHypothesis(SharedHypothesis):
    variant: str | None = None
    variant_rsid: str | None = None
    phenotype: str | None = None
    tissue: str | None = None
    warnings: list[SharedWarning] = Field(default_factory=list)
    generated_at: UtcDatetime | None = None


class SharedSnapshot(BaseModel):
    hypothesis: SharedMainHypothesis
    related: list[SharedHypothesis] = Field(default_factory=list)
    related_truncated: bool = False


class SharedHypothesisResponse(SharedSnapshot):
    schema_version: int
    shared_at: UtcDatetime
    expires_at: UtcDatetime | None = None
