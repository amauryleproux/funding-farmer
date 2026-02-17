from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import aiohttp

from whale_tracker.config import Config


log = logging.getLogger("whale_tracker.client")
ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")


class HyperliquidClient:
    def __init__(self, config: Config):
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._url = f"{self.config.api_base_url.rstrip('/')}/info"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.api_timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def post_info(self, payload: dict[str, Any]) -> Any:
        retries = max(1, self.config.api_retries)
        backoff = 1.0

        for attempt in range(1, retries + 1):
            try:
                session = await self._ensure_session()
                async with session.post(self._url, json=payload) as resp:
                    if resp.status == 429:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message="rate limited",
                            headers=resp.headers,
                        )
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as exc:
                if exc.status == 429 and attempt < retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise

    async def get_leaderboard_addresses(self) -> list[str]:
        data = await self.post_info({"type": "leaderboard"})
        serialized = json.dumps(data)
        addresses = sorted(set(ADDRESS_RE.findall(serialized)))
        if not addresses:
            log.warning("Leaderboard returned no addresses")
        return addresses

    async def get_clearinghouse_state(self, address: str) -> dict[str, Any]:
        data = await self.post_info({"type": "clearinghouseState", "user": address})
        return data if isinstance(data, dict) else {}

    async def get_user_fills(self, address: str) -> list[dict[str, Any]]:
        data = await self.post_info({"type": "userFills", "user": address})
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    async def get_vault_details(self, vault_address: str) -> dict[str, Any]:
        data = await self.post_info({"type": "vaultDetails", "vaultAddress": vault_address})
        return data if isinstance(data, dict) else {}

    async def get_user_funding(self, address: str, start_time_ms: int) -> list[dict[str, Any]]:
        data = await self.post_info(
            {"type": "userFunding", "user": address, "startTime": start_time_ms}
        )
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
