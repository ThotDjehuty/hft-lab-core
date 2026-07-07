"""
Yahoo Finance OHLCV Connector
=============================

Thin, zero-dependency wrapper around the Yahoo Finance v8 JSON API.

No API key required.  Suitable for educational and back-research use.

Usage
-----
>>> from python.connectors.yahoo_finance import YahooFinanceClient
>>> client = YahooFinanceClient()
>>> bars = client.get_ohlcv("BTC-USD", period="1mo", interval="1d")
>>> print(bars[:3])
"""

from __future__ import annotations

import urllib.request
import urllib.parse
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class OHLCVBar:
    """Single OHLCV candle."""
    symbol: str
    timestamp: float  # Unix epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def return_(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0


class YahooFinanceClient:
    """
    Minimal Yahoo Finance v8/v11 client.

    Parameters
    ----------
    timeout : float
        HTTP timeout in seconds.
    """

    CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def _get(self, url: str, params: Dict[str, Any]) -> Any:
        full_url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": "hfthot-lab-core/1.2.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_ohlcv(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> List[OHLCVBar]:
        """
        Download OHLCV bars from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. "BTC-USD", "AAPL", "ETH-USD").
        period : str
            Lookback period: ``1d``, ``5d``, ``1mo``, ``3mo``, ``6mo``,
            ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.
        interval : str
            Bar interval: ``1m``, ``2m``, ``5m``, ``15m``, ``30m``,
            ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``, ``1mo``, ``3mo``.

        Returns
        -------
        list[OHLCVBar]
        """
        url = self.CHART_URL.format(symbol=symbol)
        try:
            data = self._get(url, {"period1": "0", "period2": "9999999999",
                                   "range": period, "interval": interval,
                                   "includePrePost": "false"})
        except Exception as exc:
            raise ConnectionError(f"Yahoo Finance request failed for {symbol}: {exc}") from exc

        try:
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected Yahoo Finance response for {symbol}") from exc

        bars: List[OHLCVBar] = []
        n = len(timestamps)
        opens = quote.get("open", [None] * n)
        highs = quote.get("high", [None] * n)
        lows = quote.get("low", [None] * n)
        closes = quote.get("close", [None] * n)
        volumes = quote.get("volume", [None] * n)

        for i in range(n):
            o = opens[i]
            h = highs[i]
            lo = lows[i]
            c = closes[i]
            v = volumes[i]
            if None in (o, h, lo, c):
                continue
            bars.append(OHLCVBar(
                symbol=symbol,
                timestamp=float(timestamps[i]),
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=float(v) if v is not None else 0.0,
            ))
        return bars

    def get_close_prices(self, symbol: str, period: str = "1y") -> List[float]:
        """Return a list of daily close prices (convenience wrapper)."""
        return [b.close for b in self.get_ohlcv(symbol, period=period, interval="1d")]
