import json
from typing import List, Dict, Any
from datetime import datetime, timezone
import redis

class RedisStatusCache:
    def __init__(self, redis_url: str):
        self.r = redis.Redis().from_url(redis_url, decode_responses=True)

    def _history_key(self, hyp_id: str) -> str:
        return f"hyp:{hyp_id}:history"

    def _latest_key(self, hyp_id: str) -> str:
        return f"hyp:{hyp_id}:latest"

    def add_update(self, hyp_id: str, update: Dict[str, Any]) -> None:
        """
        Store an update in a sorted set scored by epoch millis.
        Also tracks hyp_id in an 'in-progress' set and caches the latest update.
        """
        ts_iso = update.get("timestamp")
        if not ts_iso:
            ts_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
            update["timestamp"] = ts_iso

        try:
            score = int(datetime.fromisoformat(ts_iso.replace("Z", "")).timestamp() * 1000)
        except Exception:
            score = int(datetime.now(timezone.utc).timestamp() * 1000)

        payload = json.dumps(update, separators=(",", ":"), ensure_ascii=False)
        self.r.zadd(self._history_key(hyp_id), {payload: score}) # add update to sorted set
        self.r.sadd("hyp:inprogress:set", hyp_id) # ds to store in progress
        self.r.set(self._latest_key(hyp_id), payload) # ds to store latest update 

    def get_history(self, hyp_id: str) -> List[Dict[str, Any]]:
        raw = self.r.zrange(self._history_key(hyp_id), 0, -1)
        return [json.loads(x) for x in raw] if raw else []

    def get_latest(self, hyp_id: str) -> Dict[str, Any] | None:
        raw = self.r.get(self._latest_key(hyp_id))
        return json.loads(raw) if raw else None

    def clear_history(self, hyp_id: str) -> None:
        self.r.delete(self._history_key(hyp_id))
        self.r.delete(self._latest_key(hyp_id))
        self.r.srem("hyp:inprogress:set", hyp_id)
        self.r.sadd("hyp:persisted:set", hyp_id) 

    def is_persisted(self, hyp_id: str) -> bool:
        return bool(self.r.sismember("hyp:persisted:set", hyp_id))

    def list_inprogress(self) -> List[str]:
        members = self.r.smembers("hyp:inprogress:set") or set()
        return list(members)