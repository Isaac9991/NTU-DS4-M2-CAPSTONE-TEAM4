"""
ETL Pipeline: Olist E-Commerce CSV → BigQuery Star Schema

Extracts data from CSV files, transforms into a star schema
(2 fact tables + 7 dimension tables), and loads into BigQuery.

Star Schema Tables:
  - fact_order_item  (fact)
  - fact_payment     (fact)
  - dim_customer     (dimension)
  - dim_seller       (dimension)
  - dim_product      (dimension)
  - dim_date         (dimension)
  - dim_time         (dimension)
  - dim_geolocation  (dimension)
  - dim_order        (dimension)
  - dim_review       (dimension)
"""

import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from textblob import TextBlob

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
BQ_PROJECT = os.environ.get("GCP_PROJECT", "project-62927d44-80c9-4d90-a3b")
BQ_DATASET = os.environ.get("BQ_DATASET", "brazilian_ecommerce")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "transformed")


# ── Extract ──────────────────────────────────────────────────────────────────
def extract() -> dict[str, pd.DataFrame]:
    """Read all source CSV files into DataFrames."""
    files = {
        "customers": "olist_customers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }
    dfs = {}
    for key, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        dfs[key] = pd.read_csv(path)
        print(f"  Extracted {key}: {len(dfs[key]):,} rows")
    return dfs


# ── Transform: Dimension Tables ─────────────────────────────────────────────
def build_dim_customer(dfs: dict) -> pd.DataFrame:
    """dim_customer from olist_customers_dataset."""
    df = dfs["customers"][
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
         "customer_city", "customer_state"]
    ].copy()
    df["customer_zip_code_prefix"] = pd.to_numeric(
        df["customer_zip_code_prefix"], errors="coerce"
    ).astype("Int64")
    return df.drop_duplicates(subset=["customer_id"])


def build_dim_seller(dfs: dict) -> pd.DataFrame:
    """dim_seller from olist_sellers_dataset."""
    df = dfs["sellers"][
        ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
    ].copy()
    df["seller_zip_code_prefix"] = pd.to_numeric(
        df["seller_zip_code_prefix"], errors="coerce"
    ).astype("Int64")
    return df.drop_duplicates(subset=["seller_id"])


def build_dim_product(dfs: dict) -> pd.DataFrame:
    """dim_product from products + category name translation."""
    products = dfs["products"].copy()
    translation = dfs["category_translation"].copy()

    df = products.merge(translation, on="product_category_name", how="left")
    df = df[[
        "product_id", "product_category_name_english",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm"
    ]].copy()

    for col in ["product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df.drop_duplicates(subset=["product_id"])


def build_dim_date(dfs: dict) -> pd.DataFrame:
    """dim_date derived from all order purchase timestamps."""
    orders = dfs["orders"].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    dates = orders["order_purchase_timestamp"].dropna().dt.normalize().unique()
    dates = pd.Series(dates).sort_values().reset_index(drop=True)

    df = pd.DataFrame({"full_date": dates})
    df["full_date"] = pd.to_datetime(df["full_date"])
    # date_key as YYYYMMDD integer
    df["date_key"] = (
        df["full_date"].dt.year * 10000
        + df["full_date"].dt.month * 100
        + df["full_date"].dt.day
    )
    df["year"] = df["full_date"].dt.year
    df["quarter"] = df["full_date"].dt.quarter
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()
    df["day_of_week"] = df["full_date"].dt.dayofweek  # 0=Monday
    df["day_name"] = df["full_date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    return df[
        ["date_key", "full_date", "year", "quarter", "month",
         "month_name", "day_of_week", "day_name", "is_weekend"]
    ].drop_duplicates(subset=["date_key"])


def build_dim_time(dfs: dict) -> pd.DataFrame:
    """dim_time derived from all order purchase timestamps."""
    orders = dfs["orders"].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    ts = orders["order_purchase_timestamp"].dropna()

    times = pd.DataFrame({
        "hour": ts.dt.hour,
        "minute": ts.dt.minute,
        "second": ts.dt.second,
    }).drop_duplicates().sort_values(["hour", "minute", "second"]).reset_index(drop=True)

    # time_key as HHMMSS integer
    times["time_key"] = times["hour"] * 10000 + times["minute"] * 100 + times["second"]
    times["full_time"] = pd.to_datetime(
        times["hour"].astype(str).str.zfill(2) + ":"
        + times["minute"].astype(str).str.zfill(2) + ":"
        + times["second"].astype(str).str.zfill(2),
        format="%H:%M:%S"
    ).dt.time

    def time_of_day(h):
        if 5 <= h < 12:
            return "Morning"
        elif 12 <= h < 17:
            return "Afternoon"
        elif 17 <= h < 21:
            return "Evening"
        else:
            return "Night"

    times["time_of_day"] = times["hour"].apply(time_of_day)

    return times[
        ["time_key", "full_time", "hour", "minute", "second", "time_of_day"]
    ].drop_duplicates(subset=["time_key"])


def build_dim_geolocation(dfs: dict) -> pd.DataFrame:
    """dim_geolocation from olist_geolocation_dataset.

    Aggregates to one row per zip_code_prefix with average lat/lng.
    Links to dim_customers and dim_sellers via zip_code_prefix.
    """
    geo = dfs["geolocation"].copy()
    geo["geolocation_zip_code_prefix"] = pd.to_numeric(
        geo["geolocation_zip_code_prefix"], errors="coerce"
    ).astype("Int64")

    df = geo.groupby("geolocation_zip_code_prefix").agg(
        geolocation_lat=("geolocation_lat", "mean"),
        geolocation_lng=("geolocation_lng", "mean"),
        geolocation_city=("geolocation_city", "first"),
        geolocation_state=("geolocation_state", "first"),
    ).reset_index()

    df.rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"}, inplace=True)
    df["geolocation_lat"] = df["geolocation_lat"].round(6)
    df["geolocation_lng"] = df["geolocation_lng"].round(6)
    return df


def build_dim_order(dfs: dict) -> pd.DataFrame:
    """dim_order from olist_orders_dataset.

    Captures order-level attributes: status and all delivery timestamps.
    """
    orders = dfs["orders"].copy()

    timestamp_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    return orders[[
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]].drop_duplicates(subset=["order_id"])


def build_fact_payment(dfs: dict) -> pd.DataFrame:
    """fact_payment from olist_order_payments_dataset.

    One row per payment record (order_id + payment_sequential).
    """
    payments = dfs["order_payments"].copy()
    payments["payment_value"] = pd.to_numeric(
        payments["payment_value"], errors="coerce"
    )
    payments["payment_installments"] = pd.to_numeric(
        payments["payment_installments"], errors="coerce"
    ).astype("Int64")

    # Composite key for uniqueness
    payments["payment_id"] = (
        payments["order_id"] + "_" + payments["payment_sequential"].astype(str)
    )

    return payments[[
        "payment_id", "order_id", "payment_sequential",
        "payment_type", "payment_installments", "payment_value",
    ]]


def _sentiment(text: str) -> tuple[str, float]:
    """Return (label, score) using TextBlob polarity."""
    if not isinstance(text, str) or text.strip() == "":
        return ("Neutral", 0.0)
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return (label, round(polarity, 4))


def build_dim_review(dfs: dict) -> pd.DataFrame:
    """dim_review from reviews + order_items (to get product_id) + sentiment."""
    reviews = dfs["order_reviews"].copy()
    order_items = dfs["order_items"][["order_id", "product_id"]].drop_duplicates()

    # Join reviews → order_items to get product_id
    df = reviews.merge(order_items, on="order_id", how="left")

    # Sentiment analysis on review_comment_message
    sentiment_results = df["review_comment_message"].apply(_sentiment)
    df["sentiment_label"] = sentiment_results.apply(lambda x: x[0])
    df["sentiment_score"] = sentiment_results.apply(lambda x: x[1])

    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"], errors="coerce"
    )
    df["review_answer_timestamp"] = pd.to_datetime(
        df["review_answer_timestamp"], errors="coerce"
    )

    return df[[
        "review_id", "order_id", "product_id", "review_score",
        "review_comment_title", "review_comment_message",
        "sentiment_label", "sentiment_score",
        "review_creation_date", "review_answer_timestamp"
    ]]


# ── Transform: Fact Table ────────────────────────────────────────────────────
def build_fact_order_item(dfs: dict, dim_date: pd.DataFrame) -> pd.DataFrame:
    """fact_order_item joining order_items + orders + payments."""
    items = dfs["order_items"].copy()
    orders = dfs["orders"][["order_id", "customer_id", "order_purchase_timestamp"]].copy()
    payments = dfs["order_payments"].copy()

    # Total order value per order (sum of all payment values)
    total_order = payments.groupby("order_id")["payment_value"].sum().reset_index()
    total_order.rename(columns={"payment_value": "total_order_value"}, inplace=True)

    # Join items with orders to get customer_id and timestamp
    df = items.merge(orders, on="order_id", how="left")
    df = df.merge(total_order, on="order_id", how="left")

    # Parse timestamp
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Build date_key (YYYYMMDD)
    df["date_key"] = (
        df["order_purchase_timestamp"].dt.year * 10000
        + df["order_purchase_timestamp"].dt.month * 100
        + df["order_purchase_timestamp"].dt.day
    ).astype("Int64")

    # Build order_time_key (HHMMSS)
    df["order_time_key"] = (
        df["order_purchase_timestamp"].dt.hour * 10000
        + df["order_purchase_timestamp"].dt.minute * 100
        + df["order_purchase_timestamp"].dt.second
    ).astype("Int64")

    # Compose a unique order_item_id (order_id + item sequence)
    df["order_item_id"] = df["order_id"] + "_" + df["order_item_id"].astype(str)

    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"], errors="coerce"
    )

    return df[[
        "order_item_id", "order_id", "customer_id", "product_id",
        "seller_id", "date_key", "order_time_key",
        "price", "freight_value", "total_order_value",
        "shipping_limit_date"
    ]]


# ── Load: Save locally + upload to BigQuery ──────────────────────────────────
def save_local(tables: dict[str, pd.DataFrame]):
    """Save transformed tables as CSV for inspection."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, df in tables.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {name}.csv ({len(df):,} rows)")


def load_to_bigquery(tables: dict[str, pd.DataFrame]):
    """Upload each table to BigQuery, replacing existing data."""
    client = bigquery.Client(project=BQ_PROJECT)

    # Create dataset if it doesn't exist
    dataset_ref = bigquery.DatasetReference(BQ_PROJECT, BQ_DATASET)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)

    for table_name, df in tables.items():
        table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        )
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion
        print(f"  Loaded {table_name} → {table_id} ({len(df):,} rows)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXTRACT")
    print("=" * 60)
    dfs = extract()

    print("\n" + "=" * 60)
    print("TRANSFORM")
    print("=" * 60)

    print("\n  Building dim_customer...")
    dim_customer = build_dim_customer(dfs)
    print(f"    → {len(dim_customer):,} rows")

    print("  Building dim_seller...")
    dim_seller = build_dim_seller(dfs)
    print(f"    → {len(dim_seller):,} rows")

    print("  Building dim_product...")
    dim_product = build_dim_product(dfs)
    print(f"    → {len(dim_product):,} rows")

    print("  Building dim_date...")
    dim_date = build_dim_date(dfs)
    print(f"    → {len(dim_date):,} rows")

    print("  Building dim_time...")
    dim_time = build_dim_time(dfs)
    print(f"    → {len(dim_time):,} rows")

    print("  Building dim_geolocation...")
    dim_geolocation = build_dim_geolocation(dfs)
    print(f"    → {len(dim_geolocation):,} rows")

    print("  Building dim_order...")
    dim_order = build_dim_order(dfs)
    print(f"    → {len(dim_order):,} rows")

    print("  Building dim_review (with sentiment analysis)...")
    dim_review = build_dim_review(dfs)
    print(f"    → {len(dim_review):,} rows")

    print("  Building fact_order_item...")
    fact_order_item = build_fact_order_item(dfs, dim_date)
    print(f"    → {len(fact_order_item):,} rows")

    print("  Building fact_payment...")
    fact_payment = build_fact_payment(dfs)
    print(f"    → {len(fact_payment):,} rows")

    tables = {
        "dim_customer": dim_customer,
        "dim_seller": dim_seller,
        "dim_product": dim_product,
        "dim_date": dim_date,
        "dim_time": dim_time,
        "dim_geolocation": dim_geolocation,
        "dim_order": dim_order,
        "dim_review": dim_review,
        "fact_order_item": fact_order_item,
        "fact_payment": fact_payment,
    }

    print("\n" + "=" * 60)
    print("LOAD — Local CSV")
    print("=" * 60)
    save_local(tables)

    print("\n" + "=" * 60)
    print("LOAD — BigQuery")
    print("=" * 60)
    try:
        load_to_bigquery(tables)
        print("\nAll tables loaded to BigQuery successfully.")
    except Exception as e:
        print(f"\nBigQuery upload skipped: {e}")
        print("Transformed CSVs are available in the 'transformed/' folder.")
        print("To load to BigQuery, set GCP_PROJECT env var and authenticate:")
        print("  export GCP_PROJECT=your-project-id")
        print("  gcloud auth application-default login")

    print("\nDone.")


if __name__ == "__main__":
    main()
