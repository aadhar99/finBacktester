"""
Alert Manager - Email & Telegram Notifications

Sends alerts for:
- New trading signals generated
- Positions opened (entries)
- Positions closed (exits via SL/TP)
- Daily performance summaries
- System errors

Configuration:
    Set environment variables or create .env file:
    - EMAIL_ENABLED=true
    - EMAIL_FROM=your-email@gmail.com
    - EMAIL_PASSWORD=your-app-password
    - EMAIL_TO=recipient@email.com
    - SMTP_SERVER=smtp.gmail.com
    - SMTP_PORT=587

    - TELEGRAM_ENABLED=true
    - TELEGRAM_BOT_TOKEN=your-bot-token
    - TELEGRAM_CHAT_ID=your-chat-id

Usage:
    alerts = AlertManager()
    alerts.send_new_signal('RELIANCE', 'BUY', 2500, 97)
    alerts.send_position_exit('TCS', 3500, 3650, 150, 'Take Profit Hit')
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AlertManager:
    """Send alerts via Email and Telegram."""

    def __init__(self):
        """Initialize alert manager with configuration."""

        # Email configuration
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_to = os.getenv('EMAIL_TO')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))

        # Telegram configuration
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Validate configuration
        if self.email_enabled:
            if not all([self.email_from, self.email_password, self.email_to]):
                logger.warning("⚠️  Email alerts enabled but missing credentials")
                self.email_enabled = False
            else:
                logger.info("✅ Email alerts enabled")

        if self.telegram_enabled:
            if not all([self.telegram_bot_token, self.telegram_chat_id]):
                logger.warning("⚠️  Telegram alerts enabled but missing credentials")
                self.telegram_enabled = False
            else:
                logger.info("✅ Telegram alerts enabled")

        if not self.email_enabled and not self.telegram_enabled:
            logger.warning("⚠️  No alert channels enabled")

    def _send_email(self, subject: str, body: str, html: bool = False):
        """Send email notification."""
        if not self.email_enabled:
            return

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to

            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)

            logger.info(f"  📧 Email sent: {subject}")

        except Exception as e:
            logger.error(f"  ❌ Email failed: {e}")

    def _send_telegram(self, message: str):
        """Send Telegram notification."""
        if not self.telegram_enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, data=data, timeout=10)

            if response.status_code == 200:
                logger.info(f"  💬 Telegram sent")
            else:
                logger.error(f"  ❌ Telegram failed: {response.text}")

        except Exception as e:
            logger.error(f"  ❌ Telegram failed: {e}")

    def send_new_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        confidence: float,
        pattern: str = "UNKNOWN"
    ):
        """
        Send alert for new trading signal.

        Args:
            symbol: Stock symbol
            signal_type: BUY or SELL
            price: Entry price
            confidence: Signal confidence (0-100)
            pattern: Pattern type
        """
        # Email
        subject = f"🔔 New {signal_type} Signal: {symbol}"
        body = f"""
New Trading Signal Detected!

Symbol: {symbol}
Signal: {signal_type}
Price: ₹{price:.2f}
Confidence: {confidence:.0f}%
Pattern: {pattern}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Smart Money Trading System
        """.strip()

        self._send_email(subject, body)

        # Telegram
        telegram_msg = f"""
🔔 *New {signal_type} Signal*

*{symbol}* @ ₹{price:.2f}
Confidence: {confidence:.0f}%
Pattern: {pattern}

_{datetime.now().strftime('%H:%M:%S')}_
        """.strip()

        self._send_telegram(telegram_msg)

    def send_position_opened(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ):
        """
        Send alert for position opened.

        Args:
            symbol: Stock symbol
            signal_type: BUY or SELL
            price: Entry price
            quantity: Number of shares
            stop_loss: SL price
            take_profit: TP price
            confidence: Confidence level
        """
        position_value = price * quantity

        # Email
        subject = f"✅ Position Opened: {symbol}"
        body = f"""
Paper Trade Executed!

Symbol: {symbol}
Action: {signal_type}
Entry Price: ₹{price:.2f}
Quantity: {quantity} shares
Position Value: ₹{position_value:,.0f}

Stop Loss: ₹{stop_loss:.2f}
Take Profit: ₹{take_profit:.2f}

Confidence: {confidence:.0f}%
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Smart Money Trading System
        """.strip()

        self._send_email(subject, body)

        # Telegram
        telegram_msg = f"""
✅ *Position Opened*

*{symbol}* {signal_type}
Entry: ₹{price:.2f} x {quantity}
Value: ₹{position_value:,.0f}

SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}

_{datetime.now().strftime('%H:%M:%S')}_
        """.strip()

        self._send_telegram(telegram_msg)

    def send_position_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
        days_held: int
    ):
        """
        Send alert for position exit.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/Loss in rupees
            pnl_pct: P&L percentage
            reason: Exit reason (SL/TP/Manual)
            days_held: Days position was held
        """
        emoji = "🎉" if pnl > 0 else "⚠️"
        result = "PROFIT" if pnl > 0 else "LOSS"

        # Email
        subject = f"{emoji} Position Closed: {symbol} ({result})"
        body = f"""
Position Closed!

Symbol: {symbol}
Entry Price: ₹{entry_price:.2f}
Exit Price: ₹{exit_price:.2f}

P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)
Result: {result}

Exit Reason: {reason}
Days Held: {days_held}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Smart Money Trading System
        """.strip()

        self._send_email(subject, body)

        # Telegram
        telegram_msg = f"""
{emoji} *Position Closed*

*{symbol}* - {result}
Entry: ₹{entry_price:.2f} → Exit: ₹{exit_price:.2f}

*P&L: ₹{pnl:,.0f}* ({pnl_pct:+.2f}%)

Reason: {reason}
Held: {days_held} days

_{datetime.now().strftime('%H:%M:%S')}_
        """.strip()

        self._send_telegram(telegram_msg)

    def send_daily_summary(
        self,
        portfolio_value: float,
        total_return_pct: float,
        open_positions: int,
        total_trades: int,
        win_rate: float,
        new_signals: int = 0
    ):
        """
        Send daily portfolio summary.

        Args:
            portfolio_value: Current portfolio value
            total_return_pct: Total return percentage
            open_positions: Number of open positions
            total_trades: Total trades executed
            win_rate: Win rate percentage
            new_signals: New signals today
        """
        emoji = "📈" if total_return_pct >= 0 else "📉"

        # Email
        subject = f"{emoji} Daily Summary - Portfolio: ₹{portfolio_value:,.0f}"
        body = f"""
Daily Portfolio Summary

Portfolio Value: ₹{portfolio_value:,.0f}
Total Return: {total_return_pct:+.2f}%

Open Positions: {open_positions}
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%

New Signals Today: {new_signals}

Date: {datetime.now().strftime('%Y-%m-%d')}

---
Smart Money Trading System
        """.strip()

        self._send_email(subject, body)

        # Telegram
        telegram_msg = f"""
{emoji} *Daily Summary*

Portfolio: ₹{portfolio_value:,.0f}
Return: {total_return_pct:+.2f}%

Open: {open_positions} | Trades: {total_trades}
Win Rate: {win_rate:.1f}%

New Signals: {new_signals}

_{datetime.now().strftime('%Y-%m-%d')}_
        """.strip()

        self._send_telegram(telegram_msg)

    def send_error_alert(self, error_message: str, context: str = ""):
        """
        Send alert for system errors.

        Args:
            error_message: Error message
            context: Additional context
        """
        # Email
        subject = "❌ System Error Alert"
        body = f"""
System Error Detected!

Error: {error_message}

Context: {context}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check logs for details.

---
Smart Money Trading System
        """.strip()

        self._send_email(subject, body)

        # Telegram
        telegram_msg = f"""
❌ *System Error*

{error_message}

{context}

_{datetime.now().strftime('%H:%M:%S')}_
        """.strip()

        self._send_telegram(telegram_msg)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create alert manager
    alerts = AlertManager()

    # Test alerts (uncomment to test)
    # alerts.send_new_signal('RELIANCE', 'BUY', 2500, 97, 'CLUSTERED_BUYING')
    # alerts.send_position_opened('TCS', 'BUY', 3500, 100, 3325, 4025, 95)
    # alerts.send_position_exit('INFY', 1500, 1650, 15000, 10, 'Take Profit Hit', 7)
    # alerts.send_daily_summary(1050000, 5.0, 3, 10, 70, 2)
    # alerts.send_error_alert('Database connection failed', 'daily_scan.py line 123')

    print("\n✅ Alert Manager ready!")
    print("Set environment variables to enable alerts:")
    print("  EMAIL_ENABLED=true")
    print("  TELEGRAM_ENABLED=true")
