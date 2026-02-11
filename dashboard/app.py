"""
Strategy Backtester Dashboard.

Reads backtest results from SQLite audit store.
Pages: Strategy Comparison (default), Strategy Results, Trade Analysis, Audit Log.
Supports running backtests directly from the dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import sys
import time
import json
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.sqlite_store import SQLiteStore
from data.fetcher import DataFetcher

# Symbol configuration
SYMBOL_CONFIG = {
    "NIFTY50": {"lot_size": 25, "label": "Nifty 50"},
    "BANKNIFTY": {"lot_size": 15, "label": "Bank Nifty"},
}

# Normalize legacy symbol names in params
SYMBOL_NORMALIZE = {"NIFTY": "NIFTY50"}

# Page config
st.set_page_config(
    page_title="Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] { background-color: #0e1117; }
/* Metric cards */
[data-testid="stMetric"] {
    background: #1a1a2e;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #262640;
}
/* Hide Streamlit hamburger + footer for clean look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Profit/loss colors */
.profit { color: #00cc00; font-weight: bold; }
.loss { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return SQLiteStore()


@st.cache_resource
def get_fetcher():
    return DataFetcher()


def color_pnl(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00cc00'
        elif val < 0:
            return 'color: #ff0000'
    return ''


def normalize_symbol(sym):
    """Normalize legacy symbol names (e.g. NIFTY -> NIFTY50)."""
    return SYMBOL_NORMALIZE.get(sym, sym)


# ── Sidebar ──

def render_sidebar(store: SQLiteStore):
    """Render sidebar: header button for Comparison, run selector, then page radio, then backtest expander."""
    fetcher = get_fetcher()

    with st.sidebar:
        st.header("Strategy Backtester")

        # Strategy Comparison as a button/header — clicking resets to comparison view
        if st.button("Strategy Comparison", use_container_width=True):
            st.session_state.selected_run_id = None
            st.session_state.detail_page = None
            st.rerun()

        st.markdown("---")

        # Run selector — always visible
        runs = store.get_runs()
        strategy_runs = runs[runs['strategy_name'] == 'NiftyShortAgent'] if not runs.empty else runs

        run_id = None
        page = None

        if not strategy_runs.empty:
            options = {}
            for _, r in strategy_runs.iterrows():
                label = f"Run #{r['id']} — {r['created_at'][:16]}"
                try:
                    params = json.loads(r.get('params_json', '{}') or '{}')
                    range_val = params.get('min_first_candle_range', '?')
                    sym = normalize_symbol(params.get('symbol', 'NIFTY50'))
                    label += f" ({sym}, range={range_val})"
                except (json.JSONDecodeError, TypeError):
                    pass
                options[label] = r['id']

            # Determine current selection index
            current_run = st.session_state.get('selected_run_id')
            option_keys = list(options.keys())
            default_idx = 0
            if current_run is not None:
                for i, k in enumerate(option_keys):
                    if options[k] == current_run:
                        default_idx = i
                        break

            selected_label = st.selectbox("Select Run", option_keys, index=default_idx, key="run_selector")
            run_id = options[selected_label]
            st.session_state.selected_run_id = run_id

            # Page radio — only show when a run is selected
            page = st.radio(
                "",
                ["Strategy Results", "Trade Analysis", "Audit Log"],
                index=0,
                label_visibility="collapsed",
                key="page_nav"
            )
            st.session_state.detail_page = page
        else:
            st.info("No runs found. Run a backtest below.")

        st.markdown("---")

        # Backtest params in expander
        with st.expander("Run New Backtest", expanded=False):
            # Symbol inside expander
            symbol = st.selectbox(
                "Symbol",
                list(SYMBOL_CONFIG.keys()),
                format_func=lambda s: f"{s} ({SYMBOL_CONFIG[s]['label']})",
                key="symbol_selector"
            )
            default_lot = SYMBOL_CONFIG[symbol]["lot_size"]

            min_range = st.number_input("Min 1st Candle Range", value=75.0, step=5.0, format="%.0f", key="param_min_range")
            entry_candle = st.number_input("Entry Candle", value=3, min_value=1, max_value=20, step=1, key="param_entry_candle")
            swing_lookback = st.number_input("Swing Lookback", value=5, min_value=1, max_value=20, step=1, key="param_swing_lookback")
            lot_size = st.number_input("Lot Size", value=default_lot, min_value=1, step=5, key="param_lot_size")
            entry_cutoff = st.text_input("Entry Cutoff Time (HH:MM)", value="14:00", key="param_entry_cutoff")
            stop_loss = st.number_input("Stop Loss Points", value=0.0, min_value=0.0, step=10.0, format="%.0f", key="param_stop_loss")
            interval = st.selectbox("Interval", ["5m", "15m", "30m", "1h"], index=1, key="param_interval")
            days = st.number_input("Days of Data", value=60, min_value=5, max_value=60, step=5, key="param_days")

            # Show cached data date range
            date_from, date_until = fetcher.get_cached_date_range(symbol, interval, days)
            if date_from and date_until:
                st.caption(f"Data from: {date_from}")
                st.caption(f"Data until: {date_until}")

            # Check if backtest is running
            is_running = st.session_state.get('backtest_running', False)

            if st.button("Run Backtest", disabled=is_running, type="primary"):
                project_root = Path(__file__).parent.parent
                cmd = [
                    sys.executable, str(project_root / "scripts" / "run_intraday.py"),
                    "--symbol", symbol,
                    "--min-range", str(min_range),
                    "--entry-candle", str(int(entry_candle)),
                    "--swing-lookback", str(int(swing_lookback)),
                    "--lot-size", str(int(lot_size)),
                    "--entry-cutoff", entry_cutoff,
                    "--stop-loss", str(stop_loss),
                    "--interval", interval,
                    "--days", str(int(days)),
                ]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(project_root)
                )
                st.session_state.backtest_process = process
                st.session_state.backtest_running = True
                st.rerun()

            # Poll running backtest
            if is_running:
                process = st.session_state.get('backtest_process')
                if process is not None:
                    retcode = process.poll()
                    if retcode is None:
                        st.info("Backtest running...")
                        time.sleep(2)
                        st.rerun()
                    elif retcode == 0:
                        st.success("Backtest complete!")
                        st.session_state.backtest_running = False
                        st.session_state.pop('backtest_process', None)
                        get_store.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        stderr = process.stderr.read().decode() if process.stderr else ""
                        st.error(f"Backtest failed (code {retcode})")
                        if stderr:
                            st.code(stderr[:500])
                        st.session_state.backtest_running = False
                        st.session_state.pop('backtest_process', None)

        return run_id, page


# ── Page: Strategy Comparison (DEFAULT / Home) ──

def page_strategy_comparison(store: SQLiteStore):
    # Main page header
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Strategy Backtester</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Intraday Short Strategy Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    runs_df = store.get_runs()
    runs_df = runs_df[runs_df['strategy_name'] == 'NiftyShortAgent'] if not runs_df.empty else runs_df

    if runs_df.empty:
        st.info("No backtest runs found. Use the sidebar to run a backtest.")
        return

    # Build comparison table with params expanded
    table_data = []
    for _, row in runs_df.iterrows():
        try:
            params = json.loads(row.get('params_json', '{}') or '{}')
        except (json.JSONDecodeError, TypeError):
            params = {}
        raw_sym = params.get('symbol', 'NIFTY50')
        table_data.append({
            'Run #': row['id'],
            'Date': row['created_at'][:16] if row.get('created_at') else '',
            'Symbol': normalize_symbol(raw_sym),
            'Min Range': params.get('min_first_candle_range', '?'),
            'Entry Candle': params.get('entry_candle_index', '?'),
            'Swing Lookback': params.get('swing_high_lookback', '?'),
            'Lot Size': params.get('lot_size', '?'),
            'Entry Cutoff': params.get('entry_cutoff_time', 'N/A'),
            'Stop Loss': params.get('stop_loss_points', 'N/A'),
            'Total P&L (Rs)': f"{row.get('total_pnl_rupees', 0) or 0:,.0f}",
            'Trades': row.get('total_trades', 0) or 0,
            'Win Rate %': f"{row.get('win_rate', 0) or 0:.1f}",
            'Sharpe': f"{row.get('sharpe_ratio', 0) or 0:.2f}",
            'Max DD Pts': f"{row.get('max_drawdown', 0) or 0:.0f}",
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Multi-select for comparison
    run_ids = runs_df['id'].tolist()
    run_labels = {rid: f"Run #{rid}" for rid in run_ids}
    default_ids = run_ids[:min(3, len(run_ids))]
    selected_ids = st.multiselect(
        "Select runs to compare",
        options=run_ids,
        default=default_ids,
        format_func=lambda x: run_labels.get(x, str(x))
    )

    if not selected_ids:
        return

    # Quick-compare metrics side by side
    st.subheader("Quick Compare")
    compare_data = []
    for rid in selected_ids:
        run = store.get_run(rid)
        if run:
            compare_data.append({
                'Metric': f"Run #{rid}",
                'P&L (Rs)': f"{run.get('total_pnl_rupees', 0) or 0:,.0f}",
                'P&L (Pts)': f"{run.get('total_pnl_points', 0) or 0:+.0f}",
                'Trades': run.get('total_trades', 0) or 0,
                'Win Rate': f"{run.get('win_rate', 0) or 0:.1f}%",
                'Sharpe': f"{run.get('sharpe_ratio', 0) or 0:.2f}",
                'Max DD': f"{run.get('max_drawdown', 0) or 0:.0f} pts",
            })
    if compare_data:
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

    # Overlay equity curves
    fig = go.Figure()
    for rid in selected_ids:
        trades = store.get_trades(rid)
        if not trades.empty and 'pnl_rupees' in trades.columns:
            cum_pnl = trades['pnl_rupees'].cumsum()
            equity = 100_000 + cum_pnl
            fig.add_trace(go.Scatter(
                x=trades['date'],
                y=equity,
                mode='lines+markers',
                name=f"Run #{rid}"
            ))

    fig.add_hline(y=100_000, line_dash="dash", line_color="gray",
                  annotation_text="Initial Capital")
    fig.update_layout(
        title="Equity Curves Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Rs.)",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Strategy Results (per-run) ──

def page_strategy_results(store: SQLiteStore, run_id):
    st.title("Strategy Results")

    run_info = store.get_run(run_id)
    trades_df = store.get_trades(run_id)

    if run_info is None:
        st.error("Run not found")
        return

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_pnl = run_info.get('total_pnl_rupees', 0) or 0
    total_pts = run_info.get('total_pnl_points', 0) or 0
    win_rate = run_info.get('win_rate', 0) or 0
    sharpe = run_info.get('sharpe_ratio', 0) or 0
    max_dd = run_info.get('max_drawdown', 0) or 0
    total_trades = run_info.get('total_trades', 0) or 0

    with col1:
        st.metric("Total P&L", f"Rs.{total_pnl:,.0f}", f"{total_pts:+.0f} pts")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{total_trades} trades")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_dd:.0f} pts")
    with col5:
        if not trades_df.empty and 'pnl_rupees' in trades_df.columns:
            wins = trades_df[trades_df['pnl_points'] > 0]
            losses = trades_df[trades_df['pnl_points'] <= 0]
            gross_profit = wins['pnl_rupees'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['pnl_rupees'].sum()) if len(losses) > 0 else 0
            pf = gross_profit / gross_loss if gross_loss > 0 else 0
            st.metric("Profit Factor", f"{pf:.2f}")
        else:
            st.metric("Profit Factor", "N/A")

    st.markdown("---")

    # Equity curve
    if not trades_df.empty and 'pnl_rupees' in trades_df.columns:
        trades_df['cumulative_pnl'] = trades_df['pnl_rupees'].cumsum()
        equity_values = 100_000 + trades_df['cumulative_pnl']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df['date'],
            y=equity_values,
            mode='lines+markers',
            name='Equity',
            line=dict(color='#00cc00' if total_pnl >= 0 else '#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 0, 0.05)' if total_pnl >= 0 else 'rgba(255, 0, 0, 0.05)'
        ))
        fig.add_hline(y=100_000, line_dash="dash", line_color="gray",
                       annotation_text="Initial Capital")
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (Rs.)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Daily P&L bar chart
    if not trades_df.empty and 'pnl_rupees' in trades_df.columns:
        daily_pnl = trades_df.groupby('date')['pnl_rupees'].sum().reset_index()
        colors = ['#00cc00' if v >= 0 else '#ff4444' for v in daily_pnl['pnl_rupees']]

        fig_daily = go.Figure(go.Bar(
            x=daily_pnl['date'],
            y=daily_pnl['pnl_rupees'],
            marker_color=colors,
            name='Daily P&L'
        ))
        fig_daily.update_layout(
            title="Daily P&L (Rs.)",
            xaxis_title="Date",
            yaxis_title="P&L (Rs.)",
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True)


# ── Page: Trade Analysis (per-run) ──

def page_trade_analysis(store: SQLiteStore, run_id):
    st.title("Trade Analysis")

    trades_df = store.get_trades(run_id)

    if trades_df.empty:
        st.info("No trades found for this run.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        trade_filter = st.selectbox("Filter", ["All Trades", "Winners", "Losers"])
    with col2:
        exit_reasons = ["All"] + sorted(trades_df['exit_reason'].dropna().unique().tolist())
        reason_filter = st.selectbox("Exit Reason", exit_reasons)
    with col3:
        dates = sorted(trades_df['date'].unique().tolist())
        date_options = ["All Dates"] + dates
        date_filter = st.selectbox("Date", date_options)

    filtered = trades_df.copy()
    if trade_filter == "Winners":
        filtered = filtered[filtered['pnl_points'] > 0]
    elif trade_filter == "Losers":
        filtered = filtered[filtered['pnl_points'] <= 0]

    if reason_filter != "All":
        filtered = filtered[filtered['exit_reason'] == reason_filter]

    if date_filter != "All Dates":
        filtered = filtered[filtered['date'] == date_filter]

    # Summary row
    if not filtered.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Trades Shown", len(filtered))
        with c2:
            st.metric("Total P&L", f"Rs.{filtered['pnl_rupees'].sum():,.0f}")
        with c3:
            st.metric("Avg P&L/Trade", f"{filtered['pnl_points'].mean():.1f} pts")
        with c4:
            win_count = len(filtered[filtered['pnl_points'] > 0])
            st.metric("Win Rate", f"{win_count / len(filtered) * 100:.1f}%" if len(filtered) > 0 else "N/A")

    # Trade table
    display_cols = ['date', 'entry_time', 'entry_price', 'exit_time', 'exit_price',
                    'pnl_points', 'pnl_rupees', 'exit_reason', 'swing_high',
                    'candle_1_low', 'candle_1_high', 'brokerage']
    available_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available_cols].copy()

    # Format numeric columns
    for col in ['entry_price', 'exit_price', 'pnl_points', 'pnl_rupees', 'swing_high',
                'candle_1_low', 'candle_1_high', 'brokerage']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # P&L distribution
    if not filtered.empty:
        fig = px.histogram(
            filtered, x='pnl_points', nbins=20,
            title="P&L Distribution (Points)",
            labels={'pnl_points': 'P&L Points'}
        )
        fig.update_traces(marker_color='#1f77b4')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # CSV download
    csv = trades_df.to_csv(index=False)
    st.download_button(
        label="Download Trades CSV",
        data=csv,
        file_name=f"trades_run_{run_id}.csv",
        mime="text/csv"
    )


# ── Page: Audit Log (per-run, enhanced with date range) ──

def page_audit_log(store: SQLiteStore, run_id):
    st.title("Candle Audit Log")

    # Load all audit data for the run
    audit_df = store.get_candle_audit(run_id)
    if audit_df.empty:
        st.info("No audit records found for this run.")
        return

    dates = sorted(audit_df['date'].unique().tolist())

    # Date range picker
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.selectbox("Start Date", dates, index=0, key="audit_start")
    with col2:
        end_idx = len(dates) - 1
        end_date = st.selectbox("End Date", dates, index=end_idx, key="audit_end")

    # Filter by date range
    range_audit = audit_df[(audit_df['date'] >= start_date) & (audit_df['date'] <= end_date)]

    if range_audit.empty:
        st.info("No candle data in selected range.")
        return

    # Summary across the range
    signals = range_audit[range_audit['signal_generated'] == 1]
    conditions = range_audit[range_audit['condition_met'] == 1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Candles", len(range_audit))
    with c2:
        st.metric("Signals Generated", len(signals))
    with c3:
        st.metric("Conditions Met", len(conditions))
    with c4:
        range_dates = range_audit['date'].nunique()
        st.metric("Trading Days", range_dates)

    st.markdown("---")

    # Full candle table for the range
    st.subheader("Candle-by-Candle Evaluation")
    display_cols = ['date', 'candle_number', 'time', 'open', 'high', 'low', 'close',
                    'condition_met', 'signal_generated', 'notes']
    available = [c for c in display_cols if c in range_audit.columns]
    st.dataframe(range_audit[available], use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # Candlestick chart for a single date within the range
    range_dates_list = sorted(range_audit['date'].unique().tolist())
    st.subheader("Candlestick Chart")
    chart_date = st.selectbox("Select date for chart", range_dates_list, key="audit_chart_date")

    day_audit = range_audit[range_audit['date'] == chart_date]

    if not day_audit.empty:
        # Day summary
        day_signals = day_audit[day_audit['signal_generated'] == 1]
        day_conditions = day_audit[day_audit['condition_met'] == 1]

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("Candles", len(day_audit))
        with dc2:
            st.metric("Signals", len(day_signals))
        with dc3:
            prev_open = day_audit.iloc[0].get('prev_day_open', None)
            prev_close = day_audit.iloc[0].get('prev_day_close', None)
            if prev_open and prev_close:
                st.metric("Prev Day Body", f"{min(prev_open, prev_close):.0f} - {max(prev_open, prev_close):.0f}")

        fig = go.Figure(data=[go.Candlestick(
            x=day_audit['time'],
            open=day_audit['open'],
            high=day_audit['high'],
            low=day_audit['low'],
            close=day_audit['close'],
            name='Price'
        )])

        # Mark signal candles
        for _, row in day_signals.iterrows():
            fig.add_annotation(
                x=row['time'], y=row['high'],
                text="SIGNAL", showarrow=True,
                arrowhead=2, arrowcolor="red",
                font=dict(color="red", size=10)
            )

        # Mark exit condition candles
        for _, row in day_conditions.iterrows():
            if row['signal_generated'] == 0:
                fig.add_annotation(
                    x=row['time'], y=row['low'],
                    text="EXIT", showarrow=True,
                    arrowhead=2, arrowcolor="blue", ay=30,
                    font=dict(color="blue", size=10)
                )

        fig.update_layout(
            title=f"Candles - {chart_date}",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # CSV download for date range
    csv = range_audit.to_csv(index=False)
    st.download_button(
        label="Download Audit CSV (Date Range)",
        data=csv,
        file_name=f"audit_run_{run_id}_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )


# ── Password Gate ──

def check_password():
    """Return True if password is correct or not configured."""
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return True  # No password set — open access

    if st.session_state.get("authenticated"):
        return True

    st.markdown("<h2 style='text-align: center;'>Strategy Backtester</h2>", unsafe_allow_html=True)
    entered = st.text_input("Password", type="password", key="pw_input")
    if st.button("Login", type="primary"):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


# ── Main ──

def main():
    if not check_password():
        return

    store = get_store()
    run_id, page = render_sidebar(store)

    # If no run selected or no page, show comparison
    if run_id is None or page is None:
        page_strategy_comparison(store)
    elif page == "Strategy Results":
        page_strategy_results(store, run_id)
    elif page == "Trade Analysis":
        page_trade_analysis(store, run_id)
    elif page == "Audit Log":
        page_audit_log(store, run_id)


if __name__ == "__main__":
    main()
