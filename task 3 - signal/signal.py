from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class SignalReport:
    """Báo cáo tín hiệu giao dịch"""
    ticker: str
    date: str
    close: float
    signal: str
    rsi14: float
    sma20: float
    sma50: float
    price_change_pct: float = 0.0


class TechnicalIndicators:
    """Tính toán các chỉ báo kỹ thuật"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0):
        """Bollinger Bands"""
        sma = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower


class SignalGenerator:
    """Tạo tín hiệu mua/bán dựa trên chiến lược đa chỉ báo"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.signals: pd.DataFrame = pd.DataFrame()
        self.reports: list[SignalReport] = []
    
    def calculate_indicators(self) -> pd.DataFrame:
        """Tính tất cả chỉ báo kỹ thuật"""
        print("  Đang tính toán chỉ báo...")
        
        # SMA 20 và 50
        self.df['SMA20'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: TechnicalIndicators.sma(x, 20)
        )
        self.df['SMA50'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: TechnicalIndicators.sma(x, 50)
        )
        
        # EMA 12 và 26
        self.df['EMA12'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: TechnicalIndicators.ema(x, 12)
        )
        self.df['EMA26'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: TechnicalIndicators.ema(x, 26)
        )
        
        # RSI 14
        self.df['RSI14'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: TechnicalIndicators.rsi(x, 14)
        )
        
        # MACD
        def calc_macd(group):
            macd_line, signal_line, hist = TechnicalIndicators.macd(group)
            return pd.DataFrame({
                'MACD': macd_line,
                'MACD_Signal': signal_line,
                'MACD_Hist': hist
            }, index=group.index)
        
        macd_data = self.df.groupby('Ticker')['Close'].apply(calc_macd)
        self.df['MACD'] = macd_data['MACD']
        self.df['MACD_Signal'] = macd_data['MACD_Signal']
        self.df['MACD_Hist'] = macd_data['MACD_Hist']
        
        # Bollinger Bands
        def calc_bb(group):
            upper, mid, lower = TechnicalIndicators.bollinger_bands(group)
            return pd.DataFrame({
                'BB_Upper': upper,
                'BB_Mid': mid,
                'BB_Lower': lower
            }, index=group.index)
        
        bb_data = self.df.groupby('Ticker')['Close'].apply(calc_bb)
        self.df['BB_Upper'] = bb_data['BB_Upper']
        self.df['BB_Mid'] = bb_data['BB_Mid']
        self.df['BB_Lower'] = bb_data['BB_Lower']
        
        print("  Hoàn thành tính toán chỉ báo")
        return self.df
    
    def generate_signals(self) -> pd.DataFrame:
        """Tạo tín hiệu dựa trên chiến lược đa yếu tố"""
        print("  Đang tạo tín hiệu...")
        
        # Điều kiện BUY (mua)
        buy_condition = (
            (self.df['Close'] > self.df['SMA50']) &           # Giá > SMA50 (xu hướng tăng)
            (self.df['RSI14'] < 40) &                         # RSI thấp (chưa quá mua)
            (self.df['MACD'] > self.df['MACD_Signal']) &      # MACD tăng
            (self.df['Close'] < self.df['BB_Upper'])          # Giá dưới BB Upper
        )
        
        # Điều kiện SELL (bán)
        sell_condition = (
            (self.df['Close'] < self.df['SMA50']) &           # Giá < SMA50 (xu hướng giảm)
            (self.df['RSI14'] > 60) &                         # RSI cao (chưa quá bán)
            (self.df['MACD'] < self.df['MACD_Signal']) &      # MACD giảm
            (self.df['Close'] > self.df['BB_Lower'])          # Giá trên BB Lower
        )
        
        # Gán tín hiệu
        self.df['Signal'] = 'HOLD'
        self.df.loc[buy_condition, 'Signal'] = 'BUY'
        self.df.loc[sell_condition, 'Signal'] = 'SELL'
        
        print("  Hoàn thành tạo tín hiệu")
        return self.df
    
    def get_latest_signals(self) -> pd.DataFrame:
        """Lấy tín hiệu mới nhất cho mỗi cổ phiếu"""
        print("  Đang lấy tín hiệu mới nhất...")
        
        latest = self.df.groupby('Ticker').last().reset_index()
        latest = latest[['Ticker', 'Date', 'Close', 'Signal', 'RSI14', 'SMA20', 'SMA50']]
        
        # Tính % thay đổi giá
        def calc_price_change(group):
            if len(group) > 1:
                return ((group['Close'].iloc[-1] - group['Close'].iloc[-2]) / group['Close'].iloc[-2]) * 100
            return 0
        
        price_changes = self.df.groupby('Ticker').apply(calc_price_change).reset_index()
        price_changes.columns = ['Ticker', 'Price_Change_Pct']
        latest = latest.merge(price_changes, on='Ticker', how='left')
        
        return latest
    
    def generate_reports(self, latest: pd.DataFrame) -> list[SignalReport]:
        """Tạo báo cáo cho từng cổ phiếu"""
        reports = []
        for _, row in latest.iterrows():
            report = SignalReport(
                ticker=row['Ticker'],
                date=row['Date'],
                close=row['Close'],
                signal=row['Signal'],
                rsi14=row['RSI14'],
                sma20=row['SMA20'],
                sma50=row['SMA50'],
                price_change_pct=row.get('Price_Change_Pct', 0.0)
            )
            reports.append(report)
        return reports
    
    def run(self) -> tuple[pd.DataFrame, list[SignalReport]]:
        """Chạy toàn bộ quy trình"""
        self.calculate_indicators()
        self.generate_signals()
        latest = self.get_latest_signals()
        self.reports = self.generate_reports(latest)
        return latest, self.reports


def load_data(input_file: Path) -> Optional[pd.DataFrame]:
    """Đọc dữ liệu đã làm sạch từ Task 2"""
    if not input_file.exists():
        print(f"  Không tìm thấy file: {input_file}")
        return None
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"  Đọc thành công: {input_file} ({len(df)} dòng)")
        return df
    except Exception as e:
        print(f"  Lỗi đọc file: {e}")
        return None


def save_results(latest: pd.DataFrame, reports: list[SignalReport], output_dir: Path):
    """Lưu kết quả tín hiệu"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu CSV
    csv_path = output_dir / "signal_report.csv"
    latest.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Đã lưu: {csv_path}")
    
    # Lưu JSON báo cáo chi tiết
    json_path = output_dir / "signal_report.json"
    json_data = [
        {
            "ticker": r.ticker,
            "date": r.date,
            "close": r.close,
            "signal": r.signal,
            "rsi14": r.rsi14,
            "sma20": r.sma20,
            "sma50": r.sma50,
            "price_change_pct": r.price_change_pct
        }
        for r in reports
    ]
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"  Đã lưu: {json_path}")


def print_summary(latest: pd.DataFrame):
    """In báo cáo tóm tắt"""
    print("\n" + "="*60)
    print("KẾT QUẢ TASK 3: TÍN HIỆU GIAO DỊCH")
    print("="*60)
    
    buy_signals = latest[latest['Signal'] == 'BUY']
    sell_signals = latest[latest['Signal'] == 'SELL']
    hold_signals = latest[latest['Signal'] == 'HOLD']
    
    print(f"\n  Tổng số cổ phiếu: {len(latest)}")
    print(f"  BUY: {len(buy_signals)}")
    print(f"  SELL: {len(sell_signals)}")
    print(f"  HOLD: {len(hold_signals)}")
    
    if not buy_signals.empty:
        print("\n  📈 DANH SÁCH MUA (BUY):")
        print(f"    {'Ticker':<10} {'Close':>10} {'RSI14':>8} {'SMA20':>10} {'SMA50':>10} {'Change %':>10}")
        print("    " + "-"*60)
        for _, row in buy_signals.iterrows():
            print(f"    {row['Ticker']:<10} {row['Close']:>10.2f} {row['RSI14']:>8.1f} {row['SMA20']:>10.2f} {row['SMA50']:>10.2f} {row['Price_Change_Pct']:>9.2f}%")
    
    if not sell_signals.empty:
        print("\n  📉 DANH SÁCH BÁN (SELL):")
        print(f"    {'Ticker':<10} {'Close':>10} {'RSI14':>8} {'SMA20':>10} {'SMA50':>10} {'Change %':>10}")
        print("    " + "-"*60)
        for _, row in sell_signals.iterrows():
            print(f"    {row['Ticker']:<10} {row['Close']:>10.2f} {row['RSI14']:>8.1f} {row['SMA20']:>10.2f} {row['SMA50']:>10.2f} {row['Price_Change_Pct']:>9.2f}%")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate trading signals from cleaned stock data")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("task 2 - clean/cleaned_stock.csv"),
        help="File dữ liệu đã làm sạch từ Task 2"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task 3 - signal"),
        help="Thư mục lưu kết quả"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TASK 3: TRADING SIGNAL GENERATOR")
    print("="*60)
    
    # Đọc dữ liệu
    df = load_data(args.input_file)
    if df is None or df.empty:
        print("  Không có dữ liệu để xử lý!")
        return
    
    # Khởi tạo signal generator
    generator = SignalGenerator(df)
    
    # Chạy
    latest, reports = generator.run()
    
    # Lưu kết quả
    save_results(latest, reports, args.output_dir)
    
    # In tóm tắt
    print_summary(latest)


if __name__ == "__main__":
    main()