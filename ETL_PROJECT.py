import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Any, Optional, Union

class AutoETL:
    def __init__(self):
        self.column_types = {}
        self.cleaning_report = {}
        self.data_summary = {}
    
    def detect_column_type(self, series: pd.Series) -> str:
        """يكتشف نوع العمود تلقائياً"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        # تحقق إذا كان عمود تاريخ
        date_count = non_null.astype(str).apply(self.try_parse_date).notna().sum()
        if date_count / len(non_null) > 0.7:
            return "date"
        
        # تحقق إذا كان عمود رقمي
        numeric_count = pd.to_numeric(non_null, errors='coerce').notna().sum()
        if numeric_count / len(non_null) > 0.7:
            return "numeric"
        
        # تحقق إذا كان عمود فئوي (قيم محدودة)
        unique_ratio = len(non_null.unique()) / len(non_null)
        if 0 < unique_ratio <= 0.3 and len(non_null.unique()) < 50:
            return "categorical"
        
        # تحقق إذا كان عمود نصي
        if non_null.astype(str).str.len().mean() > 10 and unique_ratio > 0.5:
            return "text"
        
        return "unknown"
    
    def try_parse_date(self, s: str) -> Union[datetime.date, pd.NaT]:
        """يحاول قراءة التاريخ بعدة صيغ"""
        if not isinstance(s, str) or pd.isna(s):
            return pd.NaT
        
        date_formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
            "%m-%d-%Y", "%m/%d/%Y", "%Y.%m.%d", "%d.%m.%Y",
            "%m.%d.%Y", "%Y%m%d", "%d%m%Y", "%m%d%Y",
            "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(s.strip(), fmt).date()
            except ValueError:
                continue
        return pd.NaT
    
    def clean_numeric_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """ينظف العمود الرقمي"""
        original_na = series.isna().sum()
        
        cleaned = series.astype(str).str.replace(r"[^\d\.,\-+]", "", regex=True)
        cleaned = cleaned.str.replace(",", ".", regex=False)  # تحويل الفواصل إلى نقاط
        cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
        
        result = pd.to_numeric(cleaned, errors="coerce")
        
        new_na = result.isna().sum()
        self.cleaning_report[col_name].append(f"تم تنظيف العمود الرقمي، القيم غير الصالحة: {new_na - original_na}")
        
        return result
    
    def clean_text_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """ينظف العمود النصي"""
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.replace({
            "": np.nan, "nan": np.nan, "None": np.nan, 
            "NULL": np.nan, "N/A": np.nan, "n/a": np.nan,
            "null": np.nan, "NaN": np.nan
        })
        
        # تقليل التكرار في النصوص الطويلة
        if cleaned.str.len().mean() > 20:
            cleaned = cleaned.str.replace(r"\s+", " ", regex=True)  # إزالة المسافات الزائدة
        
        self.cleaning_report[col_name].append("تم تنظيف العمود النصي")
        return cleaned
    
    def clean_date_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """ينظف عمود التاريخ"""
        result = series.astype(str).apply(self.try_parse_date)
        successful = result.notna().sum()
        total = len(series)
        
        self.cleaning_report[col_name].append(
            f"تم تحويل {successful} من {total} قيمة إلى تواريخ بنجاح"
        )
        
        return result
    
    def clean_categorical_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """ينظف العمود الفئوي"""
        cleaned = series.astype(str).str.strip().str.title()  # توحيد التنسيق
        cleaned = cleaned.replace({
            "": np.nan, "nan": np.nan, "None": np.nan, 
            "NULL": np.nan, "N/A": np.nan, "n/a": np.nan
        })
        
        # تحديد إذا كانت هناك قيم قليلة متكررة (فئوية حقيقية)
        value_counts = cleaned.value_counts()
        if len(value_counts) < 50:
            # نعتبرها فئوية، نعالج القيم النادرة
            threshold = max(2, 0.01 * len(cleaned))  # حد القيم النادرة
            rare_values = value_counts[value_counts < threshold].index
            if len(rare_values) > 0:
                cleaned = cleaned.replace(rare_values, "Other")
                self.cleaning_report[col_name].append(
                    f"تم دمج {len(rare_values)} قيمة نادرة تحت فئة 'Other'"
                )
        
        self.cleaning_report[col_name].append("تم تنظيف العمود الفئوي")
        return cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """يتعامل مع القيم المفقودة"""
        df = df.copy()
        columns_to_drop = []
        
        for col in df.columns:
            na_ratio = df[col].isna().mean()
            
            if na_ratio > threshold:
                columns_to_drop.append(col)
                self.cleaning_report[col].append(
                    f"تم حذف العمود لاحتوائه على {na_ratio:.1%} قيم مفقودة"
                )
            elif na_ratio > 0:
                col_type = self.column_types.get(col, "unknown")
                
                if col_type == "numeric":
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.cleaning_report[col].append(
                        f"تم تعويض القيم المفقودة بالوسيط: {median_val}"
                    )
                elif col_type == "categorical":
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    self.cleaning_report[col].append(
                        f"تم تعويض القيم المفقودة بالمنوال: {mode_val}"
                    )
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    self.cleaning_report[col].append(
                        f"تم تعويض القيم المفقودة بالقيمة: {mode_val}"
                    )
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            self.cleaning_report["global"].append(
                f"تم حذف الأعمدة: {', '.join(columns_to_drop)}"
            )
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """يتعامل مع القيم الشاذة في الأعمدة الرقمية"""
        df = df.copy()
        
        for col, col_type in self.column_types.items():
            if col_type == "numeric" and col in df.columns:
                # حساب Z-score للكشف عن القيم الشاذة
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
                if outliers.any():
                    # استبدال القيم الشاذة بالحدود
                    lower_bound = df[col].quantile(0.05)
                    upper_bound = df[col].quantile(0.95)
                    
                    df.loc[outliers, col] = np.where(
                        df.loc[outliers, col] > upper_bound, 
                        upper_bound, 
                        np.where(
                            df.loc[outliers, col] < lower_bound,
                            lower_bound,
                            df.loc[outliers, col]
                        )
                    )
                    
                    self.cleaning_report[col].append(
                        f"تم معالجة {outliers.sum()} قيمة شاذة"
                    )
        
        return df
    
    def extract(self, data_source: Union[str, pd.DataFrame, dict]) -> pd.DataFrame:
        """مرحلة الاستخراج - قراءة البيانات من مصدر متعدد"""
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        
        elif isinstance(data_source, str):
            # قراءة من ملف
            file_path = Path(data_source)
            
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"صيغة الملف غير مدعومة: {file_path.suffix}")
        
        elif isinstance(data_source, dict):
            # تحويل قاموس إلى DataFrame
            return pd.DataFrame(data_source)
        
        else:
            raise ValueError("مصدر البيانات غير مدعوم")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """مرحلة التحويل - تنظيف وتحويل البيانات"""
        # إعادة تعيين التقارير
        self.column_types = {}
        self.cleaning_report = {"global": []}
        
        # إنشاء تقرير تنظيف لكل عمود
        for col in df.columns:
            self.cleaning_report[col] = []
        
        # اكتشاف أنواع الأعمدة
        for col in df.columns:
            self.column_types[col] = self.detect_column_type(df[col])
            self.cleaning_report[col].append(
                f"تم اكتشاف نوع العمود: {self.column_types[col]}"
            )
        
        # تنظيف كل عمود حسب نوعه
        for col in df.columns:
            col_type = self.column_types[col]
            
            if col_type == "numeric":
                df[col] = self.clean_numeric_column(df[col], col)
            elif col_type == "date":
                df[col] = self.clean_date_column(df[col], col)
            elif col_type == "categorical":
                df[col] = self.clean_categorical_column(df[col], col)
            else:  # نصي أو غير معروف
                df[col] = self.clean_text_column(df[col], col)
        
        # التعامل مع القيم المفقودة
        df = self.handle_missing_values(df)
        
        # التعامل مع القيم الشاذة
        df = self.handle_outliers(df)
        
        # إزالة الصفوف المكررة
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        if removed_duplicates > 0:
            self.cleaning_report["global"].append(
                f"تم إزالة {removed_duplicates} صف مكرر"
            )
        
        # إنشاء ملخص البيانات
        self.create_data_summary(df)
        
        return df
    
    def create_data_summary(self, df: pd.DataFrame):
        """إنشاء ملخص للبيانات"""
        self.data_summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_types": self.column_types,
            "missing_values": df.isna().sum().to_dict(),
            "numeric_columns": {
                col: {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
                for col in df.columns if self.column_types.get(col) == "numeric"
            },
            "date_columns": {
                col: {
                    "min": df[col].min(),
                    "max": df[col].max()
                }
                for col in df.columns if self.column_types.get(col) == "date"
            }
        }
    
    def load(self, df: pd.DataFrame, output_path: str = None, db_connection = None):
        """مرحلة التحميل - حفظ البيانات النظيفة"""
        # حفظ إلى ملف إذا تم تحديد مسار
        if output_path:
            output_path = Path(output_path)
            
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif output_path.suffix.lower() == '.json':
                df.to_json(output_path, orient='records', indent=2)
        
        # حفظ إلى قاعدة بيانات إذا تم توفير اتصال
        if db_connection:
            if isinstance(db_connection, sqlite3.Connection):
                table_name = "cleaned_data"
                df.to_sql(table_name, db_connection, if_exists='replace', index=False)
            # يمكن إضافة دعم لأنواع قواعد بيانات أخرى هنا
        
        return df
    
    def run_etl(self, data_source, output_path=None, db_connection=None):
        """تشغيل pipeline ETL كامل"""
        print("بدء عملية ETL...")
        
        # الاستخراج
        print("مرحلة الاستخراج...")
        raw_data = self.extract(data_source)
        print(f"تم استخراج {len(raw_data)} صف و {len(raw_data.columns)} عمود")
        
        # التحويل
        print("مرحلة التحويل...")
        cleaned_data = self.transform(raw_data)
        print("تم تحويل البيانات بنجاح")
        
        # التحميل
        print("مرحلة التحميل...")
        result = self.load(cleaned_data, output_path, db_connection)
        
        if output_path:
            print(f"تم حفظ البيانات في: {output_path}")
        
        print("اكتملت عملية ETL بنجاح!")
        
        return result
    
    def generate_report(self) -> str:
        """إنشاء تقرير عن عملية التنظيف"""
        report = "تقرير عملية تنظيف البيانات\n"
        report += "=" * 50 + "\n\n"
        
        report += "ملخص البيانات:\n"
        report += f"- عدد الصفوف: {self.data_summary['total_rows']}\n"
        report += f"- عدد الأعمدة: {self.data_summary['total_columns']}\n"
        report += f"- أنواع الأعمدة: {json.dumps(self.column_types, ensure_ascii=False, indent=2)}\n\n"
        
        report += "تفاصيل عملية التنظيف:\n"
        for col, actions in self.cleaning_report.items():
            if actions:  # فقط إذا كانت هناك إجراءات
                report += f"\nالعمود {col}:\n"
                for action in actions:
                    report += f"  - {action}\n"
        
        return report

# مثال على الاستخدام
if __name__ == "__main__":
    # إنشاء بيانات تجريبية متنوعة
    sample_data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "date_col": ["2023-01-01", "2023/02/15", "15-03-2023", "invalid", "2023.05.20", 
                    "20230101", "2023-07-04", "07/08/2023", "2023-09-10", "2023-12-25"],
        "numeric_col": ["100.50", "200,75", "N/A", "300.25", "invalid", "400.50", 
                       "500.75", "600,25", "700.50", "800.75"],
        "text_col": ["  John  ", "Mary", "N/A", " Bob ", "Alice", "None", "Eve", "Charlie", "David", "Frank"],
        "categorical_col": ["A", "B", "A", "C", "B", "A", "D", "B", "A", "E"],
        "mixed_col": ["100", "200", "text", "300", "400", "more text", "500", "600", "700", "800"]
    }
    
    # إنشاء وتشغيل نظام ETL
    etl_system = AutoETL()
    
    # تشغيل عملية ETL كاملة
    cleaned_data = etl_system.run_etl(
        data_source=sample_data,
        output_path="cleaned_data.csv"
    )
    
    # إنشاء تقرير
    report = etl_system.generate_report()
    print(report)
    
    # حفظ التقرير في ملف
    with open("cleaning_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("تم حفظ التقرير في cleaning_report.txt")