"""
Alert system for sending trading notifications.

Supports multiple channels:
- Telegram (primary)
- Email (future)
- SMS (future)
"""

import requests
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeAlert:
    """Trade execution alert data."""
    action: str  # BUY, SELL, EXIT
    symbol: str
    price: float
    quantity: int
    reason: str = ""
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class TelegramAlerter:
    """
    Send trading alerts via Telegram.

    Setup:
    1. Create a bot via @BotFather on Telegram
    2. Get bot token
    3. Get your chat_id by messaging the bot and checking:
       https://api.telegram.org/bot<TOKEN>/getUpdates
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram alerter.

        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            enabled: Whether alerts are enabled (set False to disable)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

        if self.enabled:
            logger.info("Telegram alerter initialized")
        else:
            logger.info("Telegram alerter disabled")

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: Message text (supports Markdown)
            parse_mode: Parse mode (Markdown or HTML)

        Returns:
            True if successful
        """
        if not self.enabled:
            logger.debug(f"Alert disabled: {message}")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram alert sent successfully")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def send_trade_alert(self, trade: TradeAlert) -> bool:
        """
        Send trade execution alert.

        Args:
            trade: TradeAlert object

        Returns:
            True if successful
        """
        # Format emoji based on action
        emoji_map = {
            "BUY": "📈",
            "SELL": "📉",
            "EXIT": "🔄",
            "EXIT_LONG": "📉"
        }
        emoji = emoji_map.get(trade.action, "💼")

        # Build message
        message = f"{emoji} *{trade.action}*\n\n"
        message += f"*Symbol:* {trade.symbol}\n"
        message += f"*Price:* ₹{trade.price:,.2f}\n"
        message += f"*Quantity:* {trade.quantity}\n"
        message += f"*Value:* ₹{trade.price * trade.quantity:,.0f}\n"

        if trade.reason:
            message += f"*Reason:* {trade.reason}\n"

        if trade.pnl is not None:
            pnl_emoji = "✅" if trade.pnl > 0 else "❌"
            message += f"\n{pnl_emoji} *P&L:* ₹{trade.pnl:,.2f} ({trade.pnl_pct:+.2f}%)\n"

        message += f"\n*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def send_risk_alert(self, alert_type: str, message: str) -> bool:
        """
        Send risk-related alert.

        Args:
            alert_type: Type of risk alert (WARNING, CRITICAL, INFO)
            message: Alert message

        Returns:
            True if successful
        """
        emoji_map = {
            "CRITICAL": "🚨",
            "WARNING": "⚠️",
            "INFO": "ℹ️"
        }
        emoji = emoji_map.get(alert_type.upper(), "⚠️")

        alert = f"{emoji} *RISK ALERT - {alert_type.upper()}*\n\n{message}"
        return self.send_message(alert)

    def send_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_return_pct: float,
        open_positions: int,
        trades_today: int
    ) -> bool:
        """
        Send daily performance summary.

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            daily_return_pct: Daily return percentage
            open_positions: Number of open positions
            trades_today: Number of trades executed today

        Returns:
            True if successful
        """
        emoji = "📊"
        pnl_emoji = "✅" if daily_pnl > 0 else "❌" if daily_pnl < 0 else "➖"

        message = f"{emoji} *DAILY SUMMARY*\n\n"
        message += f"*Portfolio Value:* ₹{portfolio_value:,.0f}\n"
        message += f"{pnl_emoji} *Today's P&L:* ₹{daily_pnl:,.2f} ({daily_return_pct:+.2f}%)\n"
        message += f"*Open Positions:* {open_positions}\n"
        message += f"*Trades Today:* {trades_today}\n"
        message += f"\n*Date:* {datetime.now().strftime('%Y-%m-%d')}"

        return self.send_message(message)

    def send_system_alert(self, message: str, is_error: bool = False) -> bool:
        """
        Send system status alert.

        Args:
            message: System message
            is_error: Whether this is an error

        Returns:
            True if successful
        """
        emoji = "❌" if is_error else "✅"
        alert = f"{emoji} *SYSTEM*\n\n{message}"
        return self.send_message(alert)


class ConsoleAlerter:
    """Fallback alerter that just logs to console."""

    def __init__(self):
        """Initialize console alerter."""
        logger.info("Console alerter initialized (fallback mode)")

    def send_message(self, message: str, **kwargs) -> bool:
        """Log message to console."""
        logger.info(f"ALERT: {message}")
        return True

    def send_trade_alert(self, trade: TradeAlert) -> bool:
        """Log trade alert."""
        logger.info(f"TRADE ALERT: {trade.action} {trade.symbol} @ ₹{trade.price} x {trade.quantity}")
        return True

    def send_risk_alert(self, alert_type: str, message: str) -> bool:
        """Log risk alert."""
        logger.warning(f"RISK ALERT ({alert_type}): {message}")
        return True

    def send_daily_summary(self, *args, **kwargs) -> bool:
        """Log daily summary."""
        logger.info(f"DAILY SUMMARY: {args}")
        return True

    def send_system_alert(self, message: str, is_error: bool = False) -> bool:
        """Log system alert."""
        level = logging.ERROR if is_error else logging.INFO
        logger.log(level, f"SYSTEM: {message}")
        return True


def create_alerter(bot_token: Optional[str] = None, chat_id: Optional[str] = None):
    """
    Create an alerter based on available configuration.

    Args:
        bot_token: Telegram bot token (optional)
        chat_id: Telegram chat ID (optional)

    Returns:
        TelegramAlerter if configured, else ConsoleAlerter
    """
    if bot_token and chat_id:
        return TelegramAlerter(bot_token, chat_id)
    else:
        logger.warning("Telegram not configured, using console alerter")
        return ConsoleAlerter()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with console alerter (no Telegram config)
    alerter = create_alerter()

    # Test trade alert
    trade = TradeAlert(
        action="BUY",
        symbol="RELIANCE",
        price=2500.50,
        quantity=10,
        reason="Momentum breakout"
    )
    alerter.send_trade_alert(trade)

    # Test risk alert
    alerter.send_risk_alert("WARNING", "Portfolio exposure at 28% (near 30% limit)")

    # Test daily summary
    alerter.send_daily_summary(
        portfolio_value=105000,
        daily_pnl=2500,
        daily_return_pct=2.4,
        open_positions=4,
        trades_today=2
    )

    print("\nTo use Telegram alerts:")
    print("1. Create a bot via @BotFather on Telegram")
    print("2. Get your chat_id from https://api.telegram.org/bot<TOKEN>/getUpdates")
    print("3. Initialize: alerter = TelegramAlerter(bot_token='...', chat_id='...')")
