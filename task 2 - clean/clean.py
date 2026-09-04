from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class CleaningReport:
    """Báo cáo kết quả làm sạch dữ liệu"""
    input_file: str
    input_rows: int
    output_rows: int
    missing_removed: int = 0
    duplicate_removed: int = 0
    invalid_date_removed: int = 0
    invalid_numeric_removed: int = 0
    invalid_ohlc_removed: int = 0
    tickers_found: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StockDataCleaner:
    """Làm sạch dữ liệu cổ phiếu từ CafeF"""
    
    def __init__(self):
        self.required_columns = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.aliases = {
            'Ticker': {'ticker', 'symbol', 'code', 'stock', 'ma_ck', 'ma'},
            'Date': {'date', 'trading_date', 'ngay', 'ngay_giao_dich', 'time'},
            'Open': {'open', 'opening', 'gia_mo_cua'},
            'High': {'high', 'highest', 'gia_cao_nhat'},
            'Low': {'low', 'lowest', 'gia_thap_nhat'},
            'Close': {'close', 'closing', 'gia_dong_cua', 'adjusted_close', 'adj_close'},
            'Volume': {'volume', 'vol', 'khoi_luong', 'kl'},
        }
    
    def _slug(self, value: str) -> str:
        """Chuyển đổi chuỗi thành slug"""
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    
    def _parse_dates(self, values: pd.Series) -> pd.Series:
        """Parse dates với nhiều định dạng"""
        text = values.astype("string").str.strip()
        parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
        
        for date_format in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            candidate = pd.to_datetime(text, format=date_format, errors="coerce")
            parsed = parsed.fillna(candidate)
        
        return parsed
    
    def _canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa tên cột"""
        by_slug = {self._slug(col): col for col in df.columns}
        rename: dict[object, str] = {}
        
        for canonical, names in self.aliases.items():
            for name in names:
                if name in by_slug:
                    rename[by_slug[name]] = canonical
                    break
        
        # Nếu chưa tìm thấy Date, thử tìm cột có chứa 'date' hoặc 'ngay'
        if 'Date' not in rename.values():
            for slug, column in by_slug.items():
                if 'date' in slug or 'ngay' in slug or slug in {'dtyyyymmdd', 'yyyymmdd'}:
                    rename[column] = 'Date'
                    break
        
        return df.rename(columns=rename)
    
    def read_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Đọc file CSV với nhiều định dạng khác nhau"""
        try:
            # Thử đọc với nhiều encoding và separator
            for encoding in ("utf-8-sig", "cp1258", "cp1252", "utf-8"):
                for sep in (None, ",", ";", "\t", " "):
                    try:
                        df = pd.read_csv(
                            file_path, 
                            sep=sep, 
                            engine="python", 
                            encoding=encoding,
                            on_bad_lines="skip"
                        )
                        if len(df.columns) >= 2 and not df.empty:
                            print(f"  Đọc thành công: {file_path.name} (encoding={encoding}, sep={sep})")
                            return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
        except Exception as e:
            print(f"  Lỗi đọc file {file_path.name}: {e}")
        
        return None
    
    def clean_dataframe(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Làm sạch DataFrame"""
        if df is None or df.empty:
            return df
        
        # 1. Chuẩn hóa cột
        df = self._canonical_columns(df)
        
        # 2. Chỉ giữ các cột cần thiết
        existing_cols = [col for col in self.required_columns if col in df.columns]
        df = df[existing_cols]
        
        # 3. Xử lý ngày tháng
        if 'Date' in df.columns:
            df['Date'] = self._parse_dates(df['Date'])
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                report.invalid_date_removed = invalid_dates
                df = df.dropna(subset=['Date'])
        
        # 4. Chuyển đổi numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', ''),
                    errors='coerce'
                )
        
        # 5. Xóa dòng có giá trị numeric sai
        for col in numeric_cols:
            if col in df.columns:
                invalid_numeric = df[col].isna().sum()
                if invalid_numeric > 0:
                    report.invalid_numeric_removed = max(report.invalid_numeric_removed, invalid_numeric)
        
        # 6. Kiểm tra OHLC hợp lệ
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close']) |
                (df['Open'] <= 0) |
                (df['High'] <= 0) |
                (df['Low'] <= 0) |
                (df['Close'] <= 0) |
                (df['Volume'] < 0)
            ).sum()
            
            if invalid_ohlc > 0:
                report.invalid_ohlc_removed = invalid_ohlc
                df = df[
                    (df['High'] >= df['Low']) &
                    (df['High'] >= df['Open']) &
                    (df['High'] >= df['Close']) &
                    (df['Low'] <= df['Open']) &
                    (df['Low'] <= df['Close']) &
                    (df['Open'] > 0) &
                    (df['High'] > 0) &
                    (df['Low'] > 0) &
                    (df['Close'] > 0) &
                    (df['Volume'] >= 0)
                ]
        
        # 7. Xóa missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            report.missing_removed = missing_before
            df = df.dropna()
        
        # 8. Xóa duplicate
        duplicate_before = df.duplicated().sum()
        if duplicate_before > 0:
            report.duplicate_removed = duplicate_before
            df = df.drop_duplicates()
        
        return df
    
    def process_file(self, input_path: Path, output_dir: Path) -> Optional[CleaningReport]:
        """Xử lý một file duy nhất"""
        print(f"\n📁 Đang xử lý: {input_path.name}")
        
        # Đọc file
        df = self.read_file(input_path)
        if df is None:
            print(f"  ❌ Không thể đọc file: {input_path.name}")
            return None
        
        report = CleaningReport(
            input_file=input_path.name,
            input_rows=len(df)
        )
        
        # Làm sạch
        df_cleaned = self.clean_dataframe(df, report)
        
        if df_cleaned.empty:
            print(f"  ⚠️ Dữ liệu sau khi làm sạch rỗng!")
            return None
        
        # Cập nhật report
        report.output_rows = len(df_cleaned)
        report.tickers_found = df_cleaned['Ticker'].nunique() if 'Ticker' in df_cleaned.columns else 0
        
        if 'Date' in df_cleaned.columns and not df_cleaned.empty:
            report.date_range_start = df_cleaned['Date'].min().strftime('%Y-%m-%d')
            report.date_range_end = df_cleaned['Date'].max().strftime('%Y-%m-%d')
        
        # Lưu file
        output_path = output_dir / f"cleaned_{input_path.stem}.csv"
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ Đã lưu: {output_path.name} ({len(df_cleaned)} dòng)")
        
        return report
    
    def merge_all_files(self, input_dir: Path, output_dir: Path, output_name: str = "cleaned_stock.csv") -> pd.DataFrame:
        """Gộp tất cả các file đã làm sạch"""
        print("\n📦 Đang gộp tất cả các file...")
        
        all_files = list(input_dir.glob("cleaned_*.csv"))
        if not all_files:
            print("  ❌ Không tìm thấy file nào để gộp!")
            return pd.DataFrame()
        
        all_dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  ✅ Đã đọc: {file.name} ({len(df)} dòng)")
            except Exception as e:
                print(f"  ❌ Lỗi đọc {file.name}: {e}")
        
        if not all_dfs:
            print("  ❌ Không có dữ liệu để gộp!")
            return pd.DataFrame()
        
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.drop_duplicates()
        merged = merged.sort_values(['Ticker', 'Date'])
        
        output_path = output_dir / output_name
        merged.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  ✅ Đã lưu file gộp: {output_path} ({len(merged)} dòng)")
        
        return merged


def main():
    parser = argparse.ArgumentParser(description="Clean and validate stock data from CafeF")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=Path("task 1 - data/extracted"),
        help="Thư mục chứa dữ liệu thô từ Task 1"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("task 2 - clean"),
        help="Thư mục lưu dữ liệu đã làm sạch"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TASK 2: DATA CLEANING")
    print("="*60)
    
    # Tạo thư mục output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo cleaner
    cleaner = StockDataCleaner()
    
    # Tìm tất cả file CSV trong input_dir
    if args.input_dir.exists():
        csv_files = list(args.input_dir.glob("*.csv")) + list(args.input_dir.glob("*.CSV"))
        if csv_files:
            print(f"\n📂 Tìm thấy {len(csv_files)} file CSV")
            for file in csv_files:
                cleaner.process_file(file, args.output_dir)
        else:
            print("\n⚠️ Không tìm thấy file CSV nào. Thử đọc file từ thư mục task 1 - data...")
            # Fallback: đọc từ task 1 - data
            fallback_dir = Path("task 1 - data")
            csv_files = list(fallback_dir.glob("*.csv")) + list(fallback_dir.glob("*.CSV"))
            for file in csv_files:
                if "extracted" not in str(file):
                    cleaner.process_file(file, args.output_dir)
    else:
        # Fallback: đọc từ task 1 - data
        fallback_dir = Path("task 1 - data")
        csv_files = list(fallback_dir.glob("*.csv")) + list(fallback_dir.glob("*.CSV"))
        for file in csv_files:
            cleaner.process_file(file, args.output_dir)
    
    # Gộp tất cả file đã làm sạch
    merged = cleaner.merge_all_files(args.output_dir, args.output_dir)
    
    # In báo cáo tổng quan
    print("\n" + "="*60)
    print("KẾT QUẢ TASK 2")
    print("="*60)
    print(f"Tổng số dòng: {len(merged)}")
    print(f"Số cột: {len(merged.columns)}")
    print(f"Các cột: {merged.columns.tolist() if not merged.empty else 'Không có'}")
    if not merged.empty and 'Ticker' in merged.columns:
        print(f"Số cổ phiếu: {merged['Ticker'].nunique()}")
    if not merged.empty and 'Date' in merged.columns:
        print(f"Khoảng ngày: {merged['Date'].min()} -> {merged['Date'].max()}")
    print("="*60)


if __name__ == "__main__":
    main()