"""
Price Cache - Reduce API Calls and Speed Up Price Fetching

Caches prices for a configurable TTL (default 5 minutes).
Useful for avoiding rate limits and speeding up repeated requests.

Usage:
    cache = PriceCache(ttl_seconds=300)
    price = cache.get_price('RELIANCE')
    if price is None:
        price = fetch_from_api('RELIANCE')
        cache.set_price('RELIANCE', price)
"""

import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PriceCache:
    """Simple in-memory price cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize price cache.

        Args:
            ttl_seconds: Time-to-live for cached prices (default: 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[float, float, float]] = {}  # symbol -> (price, volume, timestamp)
        logger.info(f"✅ Price cache initialized (TTL: {ttl_seconds}s)")

    def get_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get cached price for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with 'close' and 'volume' if cached and fresh, None otherwise
        """
        if symbol not in self.cache:
            return None

        price, volume, timestamp = self.cache[symbol]
        age = time.time() - timestamp

        if age > self.ttl_seconds:
            # Expired
            del self.cache[symbol]
            return None

        logger.debug(f"  📦 Cache HIT: {symbol} (age: {age:.0f}s)")
        return {'close': price, 'volume': volume}

    def set_price(self, symbol: str, price: float, volume: float = 1000000):
        """
        Cache price for symbol.

        Args:
            symbol: Stock symbol
            price: Current price
            volume: Current volume (default: 1M)
        """
        self.cache[symbol] = (price, volume, time.time())
        logger.debug(f"  💾 Cache SET: {symbol} = ₹{price:.2f}")

    def clear(self):
        """Clear all cached prices."""
        self.cache.clear()
        logger.info("  🗑️  Cache cleared")

    def clear_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired = [
            symbol for symbol, (_, _, timestamp) in self.cache.items()
            if now - timestamp > self.ttl_seconds
        ]

        for symbol in expired:
            del self.cache[symbol]

        if expired:
            logger.info(f"  🗑️  Cleared {len(expired)} expired cache entries")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        now = time.time()
        fresh = sum(1 for _, _, ts in self.cache.values() if now - ts <= self.ttl_seconds)

        return {
            'total_entries': len(self.cache),
            'fresh_entries': fresh,
            'expired_entries': len(self.cache) - fresh
        }

    def is_market_hours(self) -> bool:
        """
        Check if current time is within NSE market hours.

        NSE: 9:15 AM - 3:30 PM IST (Mon-Fri)

        Returns:
            True if within market hours, False otherwise
        """
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if within market hours (9:15 AM - 3:30 PM)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_cache_ttl(self) -> int:
        """
        Get recommended cache TTL based on market hours.

        Returns:
            Recommended TTL in seconds
        """
        if self.is_market_hours():
            # During market hours: shorter TTL (1 minute)
            return 60
        else:
            # After hours: longer TTL (1 hour)
            return 3600
