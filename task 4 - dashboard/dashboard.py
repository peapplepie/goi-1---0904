from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class StockInfo:
    """Thông tin cơ bản của cổ phiếu"""
    ticker: str
    current_price: float
    price_change_pct: float
    volume: int
    high_52w: float
    low_52w: float
    rsi14: float
    signal: str


class StockDashboard:
    """Dashboard hiển thị phân tích cổ phiếu"""
    
    def __init__(self, data_path: Path, signal_path: Optional[Path] = None):
        self.data_path = data_path
        self.signal_path = signal_path
        self.df = None
        self.signals = None
        self.tickers = []
        
    def load_data(self) -> bool:
        """Đọc dữ liệu đã làm sạch và tín hiệu"""
        try:
            # Đọc dữ liệu
            if not self.data_path.exists():
                st.error(f"Không tìm thấy file dữ liệu: {self.data_path}")
                return False
            
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.tickers = sorted(self.df['Ticker'].unique())
            
            # Đọc tín hiệu nếu có
            if self.signal_path and self.signal_path.exists():
                self.signals = pd.read_csv(self.signal_path, encoding='utf-8-sig')
            
            return True
        except Exception as e:
            st.error(f"Lỗi đọc dữ liệu: {e}")
            return False
    
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Lấy dữ liệu của một cổ phiếu trong khoảng thời gian"""
        mask = (
            (self.df['Ticker'] == ticker) &
            (self.df['Date'] >= pd.to_datetime(start_date)) &
            (self.df['Date'] <= pd.to_datetime(end_date))
        )
        return self.df.loc[mask].sort_values('Date')
    
    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Lấy thông tin cơ bản của cổ phiếu"""
        stock_data = self.df[self.df['Ticker'] == ticker].sort_values('Date')
        
        if stock_data.empty:
            return None
        
        latest = stock_data.iloc[-1]
        prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
        
        # Lấy tín hiệu
        signal = 'HOLD'
        rsi14 = None
        if self.signals is not None:
            sig_row = self.signals[self.signals['Ticker'] == ticker]
            if not sig_row.empty:
                signal = sig_row.iloc[0]['Signal']
                rsi14 = sig_row.iloc[0].get('RSI14', None)
        
        return StockInfo(
            ticker=ticker,
            current_price=latest['Close'],
            price_change_pct=((latest['Close'] - prev['Close']) / prev['Close'] * 100),
            volume=int(latest['Volume']),
            high_52w=stock_data['High'].max(),
            low_52w=stock_data['Low'].min(),
            rsi14=rsi14 if rsi14 is not None else 50.0,
            signal=signal
        )
    
    def plot_candlestick(self, ticker: str, start_date: datetime, end_date: datetime) -> go.Figure:
        """Vẽ biểu đồ nến với các chỉ báo"""
        stock_data = self.get_stock_data(ticker, start_date, end_date)
        
        if stock_data.empty:
            return None
        
        # Tạo subplot với 3 hàng
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{ticker} - Giá', 'Khối lượng giao dịch', 'RSI')
        )
        
        # 1. Biểu đồ nến
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(stock_data['Close'], stock_data['Open'])]
        
        fig.add_trace(
            go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # SMA 20 và 50
        stock_data['SMA20'] = stock_data['Close'].rolling(20).mean()
        stock_data['SMA50'] = stock_data['Close'].rolling(50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=stock_data['Date'],
                y=stock_data['SMA20'],
                name='SMA20',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data['Date'],
                y=stock_data['SMA50'],
                name='SMA50',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # 2. Volume
        fig.add_trace(
            go.Bar(
                x=stock_data['Date'],
                y=stock_data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # 3. RSI
        if 'RSI14' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['RSI14'],
                    name='RSI14',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Cập nhật layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Giá", row=1, col=1)
        fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    def render_sidebar(self) -> tuple[str, datetime, datetime]:
        """Render sidebar và trả về lựa chọn"""
        st.sidebar.header("Điều khiển")
        
        # Chọn mã cổ phiếu
        selected_ticker = st.sidebar.selectbox(
            "Chọn mã cổ phiếu:",
            self.tickers,
            index=0
        )
        
        # Chọn khoảng thời gian
        min_date = self.df['Date'].min().date()
        max_date = self.df['Date'].max().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày:",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày:",
                max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Thông tin dữ liệu
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Thông tin dữ liệu")
        st.sidebar.write(f"Tổng số mã: {len(self.tickers)}")
        st.sidebar.write(f"Dữ liệu từ: {min_date} đến {max_date}")
        
        # Nút làm mới
        if st.sidebar.button("Làm mới dữ liệu"):
            st.rerun()
        
        return selected_ticker, start_date, end_date
    
    def render_dashboard(self):
        """Render toàn bộ dashboard"""
        st.set_page_config(
            page_title="Stock Analytics Platform",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("Hệ thống phân tích và sàng lọc cổ phiếu tự động")
        st.markdown("---")
        
        # Load dữ liệu
        if not self.load_data():
            st.info("Vui lòng chạy Task 1, 2, 3 trước để có dữ liệu")
            return
        
        # Sidebar
        selected_ticker, start_date, end_date = self.render_sidebar()
        
        # Main content
        # Thông tin nhanh
        stock_info = self.get_stock_info(selected_ticker)
        
        if stock_info:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Giá gần nhất",
                    f"{stock_info.current_price:,.2f}",
                    f"{stock_info.price_change_pct:+.2f}%"
                )
            with col2:
                st.metric("Khối lượng", f"{stock_info.volume:,.0f}")
            with col3:
                st.metric("Cao nhất 52 tuần", f"{stock_info.high_52w:,.2f}")
            with col4:
                st.metric("Thấp nhất 52 tuần", f"{stock_info.low_52w:,.2f}")
            with col5:
                signal_color = "green" if stock_info.signal == "BUY" else "red" if stock_info.signal == "SELL" else "gray"
                st.metric(
                    "Tín hiệu",
                    stock_info.signal,
                    delta=None
                )
        
        # Biểu đồ
        fig = self.plot_candlestick(selected_ticker, start_date, end_date)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu trong khoảng thời gian này")
        
        # Bảng dữ liệu chi tiết
        with st.expander("Xem dữ liệu chi tiết"):
            stock_data = self.get_stock_data(selected_ticker, start_date, end_date)
            if not stock_data.empty:
                display_df = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(30)
                st.dataframe(
                    display_df.style.format({
                        'Open': '{:.2f}',
                        'High': '{:.2f}',
                        'Low': '{:.2f}',
                        'Close': '{:.2f}',
                        'Volume': '{:,.0f}'
                    })
                )
        
        # Báo cáo tín hiệu
        st.markdown("---")
        st.header("Báo cáo tín hiệu giao dịch")
        
        if self.signals is not None and not self.signals.empty:
            # Phân loại
            buy_count = len(self.signals[self.signals['Signal'] == 'BUY'])
            sell_count = len(self.signals[self.signals['Signal'] == 'SELL'])
            hold_count = len(self.signals[self.signals['Signal'] == 'HOLD'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MUA (BUY)", buy_count, delta=None)
            with col2:
                st.metric("BÁN (SELL)", sell_count, delta=None)
            with col3:
                st.metric("GIỮ (HOLD)", hold_count, delta=None)
            
            # Hiển thị bảng tín hiệu
            st.dataframe(
                self.signals.style.format({
                    'Close': '{:.2f}',
                    'RSI14': '{:.1f}',
                    'SMA20': '{:.2f}',
                    'SMA50': '{:.2f}',
                    'Price_Change_Pct': '{:.2f}%'
                })
            )
        else:
            st.info("Chưa có báo cáo tín hiệu. Vui lòng chạy Task 3 trước.")
        
        st.markdown("---")
        st.caption("Xây dựng với Streamlit, Plotly và Pandas")


def main():
    parser = argparse.ArgumentParser(description="Run stock analytics dashboard")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("task 2 - clean/cleaned_stock.csv"),
        help="File dữ liệu đã làm sạch từ Task 2"
    )
    parser.add_argument(
        "--signal-path",
        type=Path,
        default=Path("task 3 - signal/signal_report.csv"),
        help="File báo cáo tín hiệu từ Task 3"
    )
    
    args = parser.parse_args()
    
    # Chạy dashboard
    dashboard = StockDashboard(args.data_path, args.signal_path)
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()