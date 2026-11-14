import os

import pandas as pd
import streamlit as st

import recommender as rec

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Vehicle Budget Recommender", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "vehicles.csv"))


@st.cache_resource
def load_models():
    # loads model.joblib (trained on Kaggle); trains again from the CSV if the file is missing
    return rec.load_or_train(load_data(), os.path.join(BASE, "model.joblib"))


df = load_data()
bundle, model_source = load_models()


def show_table(frame, with_qty_budget=None):
    """Shows a vehicle table with readable prices."""
    out = pd.DataFrame(
        {
            "Vehicle": frame["name"],
            "Category": frame["category"],
            "Price": frame["price_pkr"].map(rec.money),
            "Fuel": frame["fuel_type"],
            "Seats": frame["seats"],
            "Top speed (km/h)": frame["top_speed_kmh"],
        }
    )
    if with_qty_budget is not None:
        out["You can buy up to"] = (with_qty_budget // frame["price_pkr"]).astype(int)
    st.dataframe(out, hide_index=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Your budget")
    budget = st.number_input(
        "Budget (PKR)", min_value=1000, max_value=100_000_000_000, value=150_000, step=10_000
    )
    st.write(f"**{rec.money(budget)}**")

    st.subheader("Options")
    all_categories = sorted(df["category"].unique())
    chosen = st.multiselect("Only these categories (empty = all)", all_categories)
    max_items = st.slider("Max vehicles in one combination", 1, 4, 3)
    min_spend = st.slider("Use at least this much of the budget (%)", 50, 100, 85, step=5)

    st.caption(
        "Prices are approximate market estimates (see `date` and `source` columns in the dataset) "
        "and change often. Verify before buying."
    )

pool = df[df["category"].isin(chosen)] if chosen else df
affordable = pool[pool["price_pkr"] <= budget].sort_values("price_pkr", ascending=False)

# ------------------------------------------------------------------ header
st.title("🚗 Vehicle Budget Recommender")
st.write("Enter your budget and see what you can buy - from a cycle all the way to a helicopter.")

c1, c2, c3 = st.columns(3)
c1.metric("Vehicles you can afford", len(affordable))
if len(affordable):
    c2.metric("Cheapest", rec.money(affordable["price_pkr"].min()))
    c3.metric("Most expensive you can afford", rec.money(affordable["price_pkr"].max()))
else:
    c2.metric("Cheapest in dataset", rec.money(pool["price_pkr"].min()))
    c3.metric("You are short by", rec.money(pool["price_pkr"].min() - budget))

tab_buy, tab_combo, tab_similar, tab_price, tab_data = st.tabs(
    ["What can I buy?", "Combinations", "Similar vehicles (KNN)", "Price estimator (Random Forest)", "Dataset"]
)

# ------------------------------------------------------------------ tab 1
with tab_buy:
    if affordable.empty:
        st.info("Nothing in the dataset fits this budget yet. Try a higher budget or clear the category filter.")
    else:
        st.subheader("Everything that fits your budget")
        st.bar_chart(affordable["category"].value_counts())
        for category, group in affordable.groupby("category", sort=False):
            st.markdown(f"**{category}** ({len(group)})")
            show_table(group, with_qty_budget=budget)

# ------------------------------------------------------------------ tab 2
with tab_combo:
    st.subheader("Ways to spend your budget")
    st.caption(
        f"Combinations of up to {max_items} vehicles that use at least {min_spend}% of your budget. "
        "The same vehicle can be repeated (e.g. 2 cycles)."
    )
    combos = rec.find_combos(pool, budget, max_items=max_items, min_spend_ratio=min_spend / 100)
    if not combos:
        st.info("No combination found. Try lowering the 'use at least' percentage or raising the budget.")
    for combo in combos:
        with st.container(border=True):
            st.markdown(f"**{combo['summary']}**")
            st.caption(f"Total {rec.money(combo['total'])}  |  left over {rec.money(combo['remaining'])}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Vehicle": it["name"],
                            "Category": it["category"],
                            "Quantity": it["qty"],
                            "Unit price": rec.money(it["unit_price"]),
                        }
                        for it in combo["items"]
                    ]
                ),
                hide_index=True,
            )

# ------------------------------------------------------------------ tab 3
with tab_similar:
    st.subheader("Find similar vehicles")
    st.caption(
        "K-Nearest Neighbours compares category, fuel type, price, top speed and seats "
        "and returns the closest matches."
    )
    pick = st.selectbox("Pick a vehicle", df.sort_values("price_pkr")["name"].tolist())
    only_budget = st.checkbox("Only show vehicles within my budget", value=True)
    similar = rec.similar_vehicles(df, bundle, pick, max_price=budget if only_budget else None, k=5)
    if similar.empty:
        st.info("No similar vehicle fits inside your budget.")
    else:
        show_table(similar)

# ------------------------------------------------------------------ tab 4
with tab_price:
    st.subheader("Estimate the price of a vehicle")
    st.caption(
        "A Random Forest trained on the dataset guesses a price from the specs. "
        "With only a few dozen vehicles this is a rough estimate - it is typically off by a factor of 2-3."
    )
    a, b = st.columns(2)
    category = a.selectbox("Category", all_categories)
    fuel = b.selectbox("Fuel type", sorted(df["fuel_type"].unique()))
    seats = a.number_input("Seats", min_value=1, max_value=60, value=2, step=1)
    speed = b.number_input("Top speed (km/h)", min_value=5, max_value=1500, value=100, step=5)
    if st.button("Estimate price"):
        estimate = rec.predict_price(bundle, category, fuel, seats, speed)
        st.success(f"Estimated price: {rec.money(estimate)}")

# ------------------------------------------------------------------ tab 5
with tab_data:
    st.subheader("Dataset")
    st.dataframe(df, hide_index=True)
    st.download_button("Download vehicles.csv", df.to_csv(index=False), file_name="vehicles.csv")

st.caption(f"Models loaded from: {model_source}")
