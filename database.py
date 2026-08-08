"""
SQLite database helpers for BIST 100 Daily AI Analyst.

Manages the `trades` table and provides full CRUD helpers:
add, list (today / by date / all), fetch by id, update, delete,
plus derived holdings computed from BUY/SELL history.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Database file lives next to this module
DB_PATH = Path(__file__).resolve().parent / "portfolio.db"

VALID_ACTIONS = ("BUY", "SELL")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the trades table if it does not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                notes TEXT DEFAULT ''
            )
            """
        )
        conn.commit()


def _validate_trade_fields(action: str, quantity: float, price: float) -> str:
    """Validate shared trade fields; returns the normalized action."""
    action = (action or "").upper().strip()
    if action not in VALID_ACTIONS:
        raise ValueError("action must be 'BUY' or 'SELL'")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    return action


def _clean_ticker(ticker: str) -> str:
    """Store bare uppercase ticker codes without the '.IS' suffix."""
    cleaned = (ticker or "").upper().strip().replace(".IS", "")
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def add_trade(
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    trade_date: Optional[str] = None,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> int:
    """
    Insert a new trade and return its row id.

    Args:
        ticker: BIST ticker without suffix (e.g. 'THYAO').
        action: 'BUY' or 'SELL'.
        quantity: Number of shares.
        price: Unit price in TRY.
        trade_date: ISO date string (YYYY-MM-DD). Defaults to today.
        notes: Optional free-text notes.
    """
    action = _validate_trade_fields(action, quantity, price)
    ticker = _clean_ticker(ticker)
    if not trade_date:
        trade_date = date.today().isoformat()

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO trades (date, ticker, action, quantity, price, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trade_date, ticker, action, quantity, price, notes or ""),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_trade_by_id(trade_id: int, db_path: Path = DB_PATH) -> Optional[dict[str, Any]]:
    """Return a single trade as a dict, or None when not found."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, date, ticker, action, quantity, price, notes
            FROM trades
            WHERE id = ?
            """,
            (trade_id,),
        ).fetchone()
    return dict(row) if row else None


def update_trade(
    trade_id: int,
    trade_date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> bool:
    """
    Update all editable fields of a trade.

    Returns True if a row was updated, False when the id does not exist.
    """
    action = _validate_trade_fields(action, quantity, price)
    ticker = _clean_ticker(ticker)
    if not trade_date:
        trade_date = date.today().isoformat()

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE trades
            SET date = ?, ticker = ?, action = ?, quantity = ?, price = ?, notes = ?
            WHERE id = ?
            """,
            (trade_date, ticker, action, quantity, price, notes or "", trade_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_trade(trade_id: int, db_path: Path = DB_PATH) -> bool:
    """
    Delete a trade by id.

    Returns True if a row was deleted, False otherwise.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_todays_trades(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all trades logged for today's date as a DataFrame."""
    return get_trades_by_date(date.today().isoformat(), db_path=db_path)


def get_trades_by_date(trade_date: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all trades for a given ISO date string."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, date, ticker, action, quantity, price, notes
            FROM trades
            WHERE date = ?
            ORDER BY id DESC
            """,
            conn,
            params=(trade_date,),
        )
    return df


def get_all_trades(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return the full trade history ordered by date descending."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, date, ticker, action, quantity, price, notes
            FROM trades
            ORDER BY date DESC, id DESC
            """,
            conn,
        )
    return df


def get_current_holdings(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Compute net holdings from all BUY/SELL history.

    Returns a DataFrame with columns: ticker, quantity, avg_buy_price, cost_basis.
    Only tickers with positive net quantity are included.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT ticker, action, quantity, price
            FROM trades
            ORDER BY date ASC, id ASC
            """,
            conn,
        )

    empty = pd.DataFrame(columns=["ticker", "quantity", "avg_buy_price", "cost_basis"])
    if df.empty:
        return empty

    holdings: dict[str, dict] = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        qty = float(row["quantity"])
        price = float(row["price"])

        if ticker not in holdings:
            holdings[ticker] = {"quantity": 0.0, "cost_basis": 0.0}

        if row["action"] == "BUY":
            holdings[ticker]["cost_basis"] += qty * price
            holdings[ticker]["quantity"] += qty
        else:  # SELL — reduce quantity proportionally against avg cost
            current_qty = holdings[ticker]["quantity"]
            if current_qty <= 0:
                continue
            sell_qty = min(qty, current_qty)
            avg_cost = holdings[ticker]["cost_basis"] / current_qty
            holdings[ticker]["cost_basis"] -= sell_qty * avg_cost
            holdings[ticker]["quantity"] -= sell_qty

    rows = []
    for ticker, data in holdings.items():
        qty = data["quantity"]
        if qty > 1e-9:  # ignore floating-point dust
            avg = data["cost_basis"] / qty if qty else 0.0
            rows.append(
                {
                    "ticker": ticker,
                    "quantity": round(qty, 4),
                    "avg_buy_price": round(avg, 4),
                    "cost_basis": round(data["cost_basis"], 2),
                }
            )

    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def count_active_positions(db_path: Path = DB_PATH) -> int:
    """Return the number of tickers with a positive net holding."""
    return len(get_current_holdings(db_path))


# Ensure schema exists on import
init_db()
