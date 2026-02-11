"""
CSV Parser — Parse bulk deals and FII/DII data from CSV files downloaded from NSE.
"""

import logging
from datetime import datetime, date
from typing import List

import pandas as pd

from ai.agents.scrapers.nse_bulk_deals import BulkDeal
from ai.agents.scrapers.nse_fii_dii import FIIDIIFlow

logger = logging.getLogger(__name__)


def parse_bulk_deals_csv(filepath: str) -> List[BulkDeal]:
    """
    Parse NSE bulk deals CSV file.

    Expected columns (flexible matching):
    - Deal Date / date
    - Symbol / symbol
    - Client Name / clientName
    - Trade Type / tradeType (BUY/SELL)
    - Quantity / qty
    - Avg. Price / avgPrice

    Args:
        filepath: Path to CSV file

    Returns:
        List of BulkDeal dataclasses
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")

    # Map known NSE column variants
    col_map = {
        "deal_date": "date",
        "dealdate": "date",
        "client_name": "client_name",
        "clientname": "client_name",
        "trade_type": "deal_type",
        "tradetype": "deal_type",
        "qty": "quantity",
        "avg_price": "price",
        "avgprice": "price",
        "avg_price_(rs)": "price",
    }

    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    deals: List[BulkDeal] = []

    for _, row in df.iterrows():
        try:
            deal_date = _parse_date(str(row.get("date", "")))
            symbol = str(row.get("symbol", "")).strip()
            client_name = str(row.get("client_name", "")).strip()
            deal_type = str(row.get("deal_type", "")).strip().upper()
            quantity = int(float(row.get("quantity", 0)))
            price = float(row.get("price", 0))

            if not symbol or quantity <= 0 or price <= 0:
                continue
            if deal_type not in ("BUY", "SELL"):
                continue

            deals.append(BulkDeal(
                date=deal_date,
                symbol=symbol,
                client_name=client_name,
                deal_type=deal_type,
                quantity=quantity,
                price=price,
                value=round(quantity * price, 2),
            ))
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping CSV row: {e}")
            continue

    logger.info(f"📄 Parsed {len(deals)} bulk deals from {filepath}")
    return deals


def parse_fii_dii_csv(filepath: str) -> List[FIIDIIFlow]:
    """
    Parse FII/DII flows CSV file.

    Expected columns (flexible matching):
    - Date
    - Category (FII/DII or FPI/DII)
    - Buy Value / buy_value
    - Sell Value / sell_value
    - Net Value / net_value

    Args:
        filepath: Path to CSV file

    Returns:
        List of FIIDIIFlow dataclasses
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    col_map = {
        "buyvalue": "buy_value",
        "sellvalue": "sell_value",
        "netvalue": "net_value",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    flows: List[FIIDIIFlow] = []

    for _, row in df.iterrows():
        try:
            flow_date = _parse_date(str(row.get("date", "")))
            category = str(row.get("category", "")).strip().upper()

            if "FII" in category or "FPI" in category:
                category = "FII"
            elif "DII" in category:
                category = "DII"
            else:
                continue

            buy_value = _parse_number(row.get("buy_value", 0))
            sell_value = _parse_number(row.get("sell_value", 0))
            net_value = _parse_number(row.get("net_value", buy_value - sell_value))

            flows.append(FIIDIIFlow(
                date=flow_date,
                category=category,
                buy_value=buy_value,
                sell_value=sell_value,
                net_value=net_value,
            ))
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping CSV row: {e}")
            continue

    logger.info(f"📄 Parsed {len(flows)} FII/DII flow records from {filepath}")
    return flows


def _parse_date(date_str: str) -> date:
    """Parse date string supporting multiple formats."""
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _parse_number(value) -> float:
    """Parse a number that may contain commas."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace(",", "").strip() or "0")
    return 0.0
