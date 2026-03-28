"""
Investor Presentation Charts — Olist E-Commerce
Generates charts showing business strength and areas for improvement.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TRANSFORMED = os.path.join(os.path.dirname(__file__), "transformed")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
fact = pd.read_csv(os.path.join(TRANSFORMED, "fact_order_item.csv"))
orders = pd.read_csv(os.path.join(TRANSFORMED, "dim_order.csv"))
reviews = pd.read_csv(os.path.join(TRANSFORMED, "dim_review.csv"))
customers = pd.read_csv(os.path.join(TRANSFORMED, "dim_customer.csv"))
products = pd.read_csv(os.path.join(TRANSFORMED, "dim_product.csv"))
payments = pd.read_csv(os.path.join(TRANSFORMED, "fact_payment.csv"))

# Parse dates
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])
orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M")

# Join fact table with order dates
fact_with_dates = fact.merge(
    orders[["order_id", "order_purchase_timestamp", "month"]],
    on="order_id", how="left"
)

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
THRIVING_COLOR = "#2ecc71"
THRIVING_ACCENT = "#27ae60"
IMPROVE_COLOR = "#e74c3c"
IMPROVE_ACCENT = "#c0392b"
BLUE = "#3498db"
PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]


def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS IS THRIVING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CHARTS — Business is Thriving")
print("=" * 60)

# ── 1. Monthly Revenue Growth ────────────────────────────────────────────────
monthly_rev = (
    fact_with_dates.groupby("month")["price"]
    .sum().reset_index()
)
monthly_rev["month_dt"] = monthly_rev["month"].dt.to_timestamp()
# Exclude incomplete first/last months
monthly_rev = monthly_rev.iloc[1:-1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(monthly_rev["month_dt"], monthly_rev["price"], alpha=0.3, color=THRIVING_COLOR)
ax.plot(monthly_rev["month_dt"], monthly_rev["price"], color=THRIVING_ACCENT, linewidth=2.5, marker="o", markersize=5)
ax.set_title("Monthly Revenue Growth", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("")
ax.set_ylabel("Revenue (R$)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
plt.xticks(rotation=45)
save(fig, "01_monthly_revenue_growth.png")

# ── 2. Monthly Order Volume Growth ──────────────────────────────────────────
monthly_orders = (
    orders.groupby("month")["order_id"]
    .nunique().reset_index(name="order_count")
)
monthly_orders["month_dt"] = monthly_orders["month"].dt.to_timestamp()
monthly_orders = monthly_orders.iloc[1:-1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly_orders["month_dt"], monthly_orders["order_count"], width=20, color=THRIVING_COLOR, edgecolor=THRIVING_ACCENT)
ax.set_title("Monthly Order Volume", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Number of Orders", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=45)
save(fig, "02_monthly_order_volume.png")

# ── 3. New Customers Acquired per Month ─────────────────────────────────────
first_order = (
    orders.sort_values("order_purchase_timestamp")
    .drop_duplicates(subset=["customer_id"], keep="first")
)
first_order["month"] = first_order["order_purchase_timestamp"].dt.to_period("M")
new_cust = first_order.groupby("month")["customer_id"].nunique().reset_index(name="new_customers")
new_cust["month_dt"] = new_cust["month"].dt.to_timestamp()
new_cust = new_cust.iloc[1:-1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(new_cust["month_dt"], new_cust["new_customers"], width=20, color=BLUE, edgecolor="#2980b9")
ax.set_title("New Customers Acquired per Month", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("New Customers", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=45)
save(fig, "03_new_customers_per_month.png")

# ── 4. Top 10 Product Categories by Revenue ─────────────────────────────────
fact_prod = fact.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
cat_rev = (
    fact_prod.groupby("product_category_name_english")["price"]
    .sum().nlargest(10).sort_values()
)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(cat_rev.index, cat_rev.values, color=PALETTE * 2, edgecolor="white")
ax.set_title("Top 10 Product Categories by Revenue", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Revenue (R$)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
for bar, val in zip(bars, cat_rev.values):
    ax.text(val + cat_rev.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"R${val:,.0f}", va="center", fontsize=9)
save(fig, "04_top10_categories_revenue.png")

# ── 5. Customer Geographic Reach ────────────────────────────────────────────
state_orders = (
    orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    .groupby("customer_state")["order_id"].nunique()
    .sort_values(ascending=False).head(15).sort_values()
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(state_orders.index, state_orders.values, color=THRIVING_COLOR, edgecolor=THRIVING_ACCENT)
ax.set_title("Orders by State — Top 15", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Number of Orders", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
save(fig, "05_orders_by_state.png")

# ── 6. Payment Method Distribution (shows credit card dominance) ────────────
pay_dist = payments.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    pay_dist.values, labels=pay_dist.index, autopct="%1.1f%%",
    colors=PALETTE, startangle=90, textprops={"fontsize": 11}
)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("Revenue by Payment Method", fontsize=16, fontweight="bold", pad=15)
save(fig, "06_payment_method_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# ROOM FOR IMPROVEMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CHARTS — Room for Improvement")
print("=" * 60)

# ── 7. Delivery Performance — Late vs On-Time ───────────────────────────────
delivered = orders[orders["order_status"] == "delivered"].dropna(
    subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
).copy()
delivered["days_late"] = (
    delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
).dt.days
delivered["delivery_status"] = delivered["days_late"].apply(
    lambda d: "On Time" if d <= 0 else ("1–7 days late" if d <= 7 else "7+ days late")
)
del_counts = delivered["delivery_status"].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
colors_del = [THRIVING_COLOR, "#f39c12", IMPROVE_COLOR]
labels_order = ["On Time", "1–7 days late", "7+ days late"]
values = [del_counts.get(l, 0) for l in labels_order]
wedges, texts, autotexts = ax.pie(
    values, labels=labels_order, autopct="%1.1f%%",
    colors=colors_del, startangle=90, textprops={"fontsize": 12}
)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("Delivery Performance", fontsize=16, fontweight="bold", pad=15)
save(fig, "07_delivery_performance.png")

# ── 8. Average Delivery Delay Over Time ─────────────────────────────────────
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")
monthly_delay = delivered.groupby("month")["days_late"].mean().reset_index()
monthly_delay["month_dt"] = monthly_delay["month"].dt.to_timestamp()
monthly_delay = monthly_delay.iloc[1:-1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly_delay["month_dt"], monthly_delay["days_late"], width=20,
       color=[IMPROVE_COLOR if d > 0 else THRIVING_COLOR for d in monthly_delay["days_late"]],
       edgecolor="white")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_title("Average Delivery Delay by Month (days vs estimated)", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Avg Days Late (negative = early)", fontsize=12)
plt.xticks(rotation=45)
save(fig, "08_avg_delivery_delay_by_month.png")

# ── 9. Review Score Distribution ────────────────────────────────────────────
score_dist = reviews["review_score"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
colors_score = [IMPROVE_COLOR, IMPROVE_ACCENT, "#f39c12", BLUE, THRIVING_COLOR]
ax.bar(score_dist.index, score_dist.values, color=colors_score, edgecolor="white", width=0.7)
for i, (score, count) in enumerate(zip(score_dist.index, score_dist.values)):
    ax.text(score, count + 500, f"{count:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Review Score Distribution", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Review Score", fontsize=12)
ax.set_ylabel("Number of Reviews", fontsize=12)
ax.set_xticks([1, 2, 3, 4, 5])
pct_low = (score_dist.loc[1] + score_dist.loc[2]) / score_dist.sum() * 100
ax.annotate(f"{pct_low:.1f}% rated 1–2 stars", xy=(1.5, score_dist.loc[1]),
            xytext=(3, score_dist.max() * 0.85),
            arrowprops=dict(arrowstyle="->", color=IMPROVE_COLOR, lw=2),
            fontsize=12, color=IMPROVE_COLOR, fontweight="bold")
save(fig, "09_review_score_distribution.png")

# ── 10. Customer Retention — Repeat vs One-Time ─────────────────────────────
cust_orders = orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
orders_per_cust = cust_orders.groupby("customer_unique_id")["order_id"].nunique()
retention = pd.Series({
    "One-time": (orders_per_cust == 1).sum(),
    "Repeat (2+)": (orders_per_cust >= 2).sum(),
})

fig, ax = plt.subplots(figsize=(8, 8))
colors_ret = [IMPROVE_COLOR, THRIVING_COLOR]
wedges, texts, autotexts = ax.pie(
    retention.values, labels=retention.index, autopct="%1.1f%%",
    colors=colors_ret, startangle=90, textprops={"fontsize": 13},
    explode=(0.03, 0.03)
)
for t in autotexts:
    t.set_fontweight("bold")
    t.set_fontsize(14)
ax.set_title("Customer Retention: Repeat vs One-Time", fontsize=16, fontweight="bold", pad=15)
save(fig, "10_customer_retention.png")

# ── 11. Review Sentiment Breakdown ──────────────────────────────────────────
sent_dist = reviews["sentiment_label"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
colors_sent = {"Positive": THRIVING_COLOR, "Neutral": "#f39c12", "Negative": IMPROVE_COLOR}
bars = ax.bar(sent_dist.index, sent_dist.values,
              color=[colors_sent.get(l, BLUE) for l in sent_dist.index],
              edgecolor="white", width=0.6)
for bar, val in zip(bars, sent_dist.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 500,
            f"{val:,}\n({val/sent_dist.sum()*100:.1f}%)",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Review Sentiment Analysis", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Number of Reviews", fontsize=12)
save(fig, "11_review_sentiment.png")

# ── 12. Low-Performing Categories (highest % of 1–2 star reviews) ───────────
rev_prod = reviews.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
rev_prod = rev_prod.dropna(subset=["product_category_name_english"])
cat_stats = rev_prod.groupby("product_category_name_english").agg(
    total_reviews=("review_score", "count"),
    low_reviews=("review_score", lambda x: (x <= 2).sum()),
).reset_index()
cat_stats = cat_stats[cat_stats["total_reviews"] >= 100]  # meaningful sample
cat_stats["pct_low"] = cat_stats["low_reviews"] / cat_stats["total_reviews"] * 100
worst = cat_stats.nlargest(10, "pct_low").sort_values("pct_low")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(worst["product_category_name_english"], worst["pct_low"],
        color=IMPROVE_COLOR, edgecolor=IMPROVE_ACCENT)
for i, (pct, total) in enumerate(zip(worst["pct_low"], worst["total_reviews"])):
    ax.text(pct + 0.5, i, f"{pct:.1f}% ({total:,} reviews)", va="center", fontsize=9)
ax.set_title("Categories with Highest % of Low Ratings (1–2 stars)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("% of Reviews Rated 1–2 Stars", fontsize=12)
save(fig, "12_low_performing_categories.png")


print(f"\n✅ All charts saved to: {CHARTS_DIR}/")
