"""
HTML Parser — Extract tables from NSE HTML pages using BeautifulSoup.
"""

import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_nse_table(
    html_content: str,
    table_class: Optional[str] = None,
    table_id: Optional[str] = None,
    table_index: int = 0,
) -> pd.DataFrame:
    """
    Extract a table from NSE HTML content into a pandas DataFrame.

    Args:
        html_content: Raw HTML string
        table_class: CSS class of the target table (optional)
        table_id: HTML id of the target table (optional)
        table_index: If multiple tables match, use this index (default: 0)

    Returns:
        DataFrame with table contents. Empty DataFrame if no table found.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Find the target table
    kwargs = {}
    if table_class:
        kwargs["class_"] = table_class
    if table_id:
        kwargs["id"] = table_id

    tables = soup.find_all("table", **kwargs)

    if not tables:
        logger.warning("No matching table found in HTML content")
        return pd.DataFrame()

    if table_index >= len(tables):
        logger.warning(f"Table index {table_index} out of range ({len(tables)} tables found)")
        return pd.DataFrame()

    table = tables[table_index]

    # Extract header row
    headers = []
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
    else:
        # Try first row as header
        first_row = table.find("tr")
        if first_row:
            headers = [cell.get_text(strip=True) for cell in first_row.find_all(["th", "td"])]

    # Extract body rows
    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells and cells != headers:
            rows.append(cells)

    if not rows:
        logger.warning("Table found but contains no data rows")
        return pd.DataFrame()

    # Build DataFrame
    if headers and len(headers) == len(rows[0]):
        df = pd.DataFrame(rows, columns=headers)
    else:
        df = pd.DataFrame(rows)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")

    logger.info(f"📄 Parsed HTML table: {len(df)} rows x {len(df.columns)} columns")
    return df
