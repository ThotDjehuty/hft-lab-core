"""
Polymarket CLOB REST Connector
==============================

Minimal, dependency-light REST client for the public **Polymarket CLOB API**
and **Gamma API** (metadata / search).

Polymarket is a prediction-market exchange (https://polymarket.com).
This connector uses only the *public* endpoints — no API key required for
read-only market data.

Public endpoints used
---------------------
CLOB API (``https://clob.polymarket.com``):
- ``GET /markets``       – list active markets
- ``GET /book``          – Level-2 order book for a token
- ``GET /trades``        – Recent trades for a token
- ``GET /last-trade-price`` – Last traded price

Gamma API (``https://gamma-api.polymarket.com``):
- ``GET /public-search`` – full-text search across events & markets
- ``GET /events``        – list / filter events
- ``GET /markets``       – list / filter individual markets

Rate limits (as of 2025): ~10 req/s per IP.

Usage
-----
>>> from python.connectors.polymarket_rest import PolymarketCLOBClient
>>> client = PolymarketCLOBClient()
>>> markets = client.get_markets(limit=5)
>>> print(markets[0]["question"])
>>> book = client.get_order_book(markets[0]["tokens"][0]["token_id"])
>>> print(book.best_bid, book.best_ask, book.spread)
>>>
>>> # Gamma search
>>> from python.connectors.polymarket_rest import PolymarketGammaClient
>>> gamma = PolymarketGammaClient()
>>> results = gamma.search_events("Bitcoin price")
>>> for ev in results:
...     print(ev["title"], [m["question"] for m in ev.get("markets", [])])
"""

from __future__ import annotations

import time
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceLevel:
    """Single level in the order book."""
    price: float
    size: float


@dataclass
class OrderBookData:
    """Level-2 snapshot from the Polymarket CLOB API."""
    token_id: str
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def bid_depth(self) -> float:
        """Total size on the bid side."""
        return sum(l.size for l in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total size on the ask side."""
        return sum(l.size for l in self.asks)

    @property
    def book_imbalance(self) -> Optional[float]:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth), in [-1, 1]."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return None
        return (self.bid_depth - self.ask_depth) / total

    def vwap(self, side: str = "bid", levels: int = 5) -> Optional[float]:
        """Volume-weighted average price for the given side."""
        book = self.bids if side == "bid" else self.asks
        top = book[:levels]
        total_sz = sum(l.size for l in top)
        if total_sz == 0:
            return None
        return sum(l.price * l.size for l in top) / total_sz

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "book_imbalance": self.book_imbalance,
            "vwap_bid5": self.vwap("bid", 5),
            "vwap_ask5": self.vwap("ask", 5),
            "timestamp": self.timestamp,
        }


@dataclass
class MarketSnapshot:
    """Compact snapshot of a single Polymarket market."""
    condition_id: str
    question: str
    tokens: List[Dict[str, Any]]
    active: bool
    volume: float
    liquidity: float
    start_date: Optional[str]
    end_date: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketCLOBClient:
    """
    Public Polymarket CLOB REST client.

    Uses only the anonymous public endpoints — no API key required.

    Parameters
    ----------
    base_url : str
        Override the default CLOB endpoint.
    timeout : float
        HTTP timeout in seconds.
    retry_delay : float
        Seconds to wait between retries on rate-limit responses.
    """

    DEFAULT_BASE = "https://clob.polymarket.com"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        timeout: float = 10.0,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_delay = retry_delay

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Issue a GET request; returns parsed JSON."""
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "hfthot-lab-core/1.2.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── Public API ────────────────────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 20,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List active Polymarket markets.

        Parameters
        ----------
        limit : int
            Maximum number of markets to return.
        active_only : bool
            Filter to active, open markets only.

        Returns
        -------
        list[dict]
            Raw market objects from the API.
        """
        params: Dict[str, Any] = {"limit": limit}
        if active_only:
            params["active"] = "true"
        try:
            data = self._get("/markets", params)
            markets: List[Dict[str, Any]] = (
                data if isinstance(data, list)
                else data.get("data", [])
            )
            return markets[:limit]
        except Exception as exc:
            raise ConnectionError(f"Failed to fetch markets: {exc}") from exc

    def get_order_book(self, token_id: str, depth: int = 20) -> OrderBookData:
        """
        Fetch a Level-2 order book snapshot.

        Parameters
        ----------
        token_id : str
            The CLOB token ID for the YES or NO outcome.
        depth : int
            Number of price levels per side to return.

        Returns
        -------
        OrderBookData
        """
        try:
            data = self._get("/book", {"token_id": token_id})
        except Exception as exc:
            raise ConnectionError(f"Failed to fetch order book for {token_id}: {exc}") from exc

        def parse_levels(raw: List[Dict[str, str]]) -> List[PriceLevel]:
            levels = [
                PriceLevel(price=float(l["price"]), size=float(l["size"]))
                for l in raw
            ]
            return levels[:depth]

        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        # API returns bids in ascending order — reverse to best-first
        bids = sorted(parse_levels(bids_raw), key=lambda l: l.price, reverse=True)
        asks = sorted(parse_levels(asks_raw), key=lambda l: l.price)

        return OrderBookData(token_id=token_id, bids=bids, asks=asks)

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """
        Fetch the last traded price for a token.

        Returns ``None`` if no trades exist or the endpoint is unavailable.
        """
        try:
            data = self._get("/last-trade-price", {"token_id": token_id})
            price_str = data.get("price")
            return float(price_str) if price_str is not None else None
        except Exception:
            return None

    def get_recent_trades(self, token_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a token.

        Parameters
        ----------
        token_id : str
        limit : int
            Maximum number of recent trades.

        Returns
        -------
        list[dict]  – each with ``price``, ``size``, ``side``, ``timestamp``
        """
        try:
            data = self._get("/trades", {"token_id": token_id, "limit": limit})
            trades = data if isinstance(data, list) else data.get("data", [])
            return trades[:limit]
        except Exception as exc:
            raise ConnectionError(f"Failed to fetch trades for {token_id}: {exc}") from exc

    def get_multiple_books(
        self,
        token_ids: List[str],
        delay: float = 0.35,
    ) -> Dict[str, OrderBookData]:
        """
        Fetch order books for multiple tokens with rate-limit delay.

        Parameters
        ----------
        token_ids : list[str]
        delay : float
            Seconds to wait between requests (respects ~3 req/s limit).

        Returns
        -------
        dict[str, OrderBookData]
        """
        results: Dict[str, OrderBookData] = {}
        for tid in token_ids:
            try:
                results[tid] = self.get_order_book(tid)
            except Exception:
                pass  # silently skip failed fetches
            if tid != token_ids[-1]:
                time.sleep(delay)
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Gamma API client  (metadata / search)
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketGammaClient:
    """
    Public Polymarket **Gamma API** client.

    The Gamma API exposes event & market metadata, full-text search, and
    aggregate statistics — no API key required.

    Parameters
    ----------
    base_url : str
        Override the default Gamma endpoint.
    timeout : float
        HTTP timeout in seconds.
    """

    DEFAULT_BASE = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "hfthot-lab-core/1.2.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit_per_type: int = 10,
        events_status: Optional[str] = "active",
    ) -> Dict[str, Any]:
        """
        Full-text search across events, markets and profiles.

        Parameters
        ----------
        query : str
            Free-form search text (e.g. "Bitcoin price", "ETH", "Trump").
        limit_per_type : int
            Max results per entity type.
        events_status : str | None
            Filter events by status: ``"active"``, ``"closed"``, or None for all.

        Returns
        -------
        dict  – ``{"events": [...], "tags": [...], "profiles": [...]}``
        """
        params: Dict[str, Any] = {
            "q": query,
            "limit_per_type": limit_per_type,
        }
        if events_status:
            params["events_status"] = events_status
        return self._get("/public-search", params)

    def search_events(
        self,
        query: str,
        *,
        limit: int = 10,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for prediction-market **events** matching *query*.

        Each event may contain multiple markets (outcomes).  Returns the
        events list only, with embedded ``markets`` sub-objects.

        Parameters
        ----------
        query : str
            Search text.
        limit : int
            Max events to return.
        active_only : bool
            If True, restrict to active (non-closed) events.

        Returns
        -------
        list[dict]
        """
        status = "active" if active_only else None
        data = self.search(query, limit_per_type=limit, events_status=status)
        events: List[Dict[str, Any]] = data.get("events", [])
        return events[:limit]

    # ── Events ────────────────────────────────────────────────────────────

    def get_events(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        active: Optional[bool] = True,
        slug: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List events from the Gamma API with optional filters.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if slug:
            params["slug"] = slug
        data = self._get("/events", params)
        if isinstance(data, list):
            return data
        return data.get("data", data.get("events", []))

    # ── Markets ───────────────────────────────────────────────────────────

    def get_markets(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        active: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """
        List individual markets from the Gamma API.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        data = self._get("/markets", params)
        if isinstance(data, list):
            return data
        return data.get("data", data.get("markets", []))
