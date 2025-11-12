# app.py
import streamlit as st
import pandas as pd
import glob, os
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- PAGE --------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍴", layout="centered")
st.title("🍴 Statewise Restaurant Food Recommender System")
st.caption("Restaurant-level FP-Growth with Top Picks & per-restaurant recommendations")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    food_df = pd.read_csv("indian_food_use.csv")
    food_df.columns = food_df.columns.str.strip().str.lower()

    restaurant_files = sorted(glob.glob("*.csv"))
    restaurant_files = [f for f in restaurant_files if ("west_bengal" in f.lower()) or ("rajasthan" in f.lower())]

    restaurants = {}
    for f in restaurant_files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        if "transaction_items" not in df.columns:
            continue
        restaurants[name] = df.copy()

    return food_df, restaurants

food_df, restaurants = load_data()
st.info(f"Loaded food metadata ({food_df.shape[0]} rows) and {len(restaurants)} restaurant files: {', '.join(restaurants.keys())}")

# -------------------- HELPERS --------------------
def parse_txn_items(t):
    return [x.split(" (")[0].strip().lower() for x in str(t).split(";") if x.strip()]

def get_item_count(df, item):
    item = item.lower()
    return sum(item in parse_txn_items(t) for t in df["transaction_items"].astype(str))

def top_items_in_restaurant(df, top_n=10):
    counter = {}
    for t in df["transaction_items"].astype(str):
        for i in parse_txn_items(t):
            counter[i] = counter.get(i, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

def run_fp_with_fallback(df, supports=(0.1, 0.05, 0.01, 0.005)):
    """
    Try FP-Growth with decreasing support values until rules found or supports exhausted.
    Returns rules DataFrame (may be empty).
    """
    txns = [parse_txn_items(t) for t in df["transaction_items"].astype(str)]
    if not txns:
        return pd.DataFrame()
    for s in supports:
        try:
            te = TransactionEncoder()
            te_ary = te.fit(txns).transform(txns)
            df_enc = pd.DataFrame(te_ary, columns=te.columns_)
            freq_items = fpgrowth(df_enc, min_support=s, use_colnames=True)
            if freq_items.empty:
                continue
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.05)
            if not rules.empty:
                return rules
        except Exception:
            continue
    return pd.DataFrame()

# -------------------- UI --------------------
st.header("🍛 Step 1 • Choose Your Action")
choice = st.radio("Select an option:", ["🍴 Food Recommendation", "⭐ Top Picks"])

# -------------------- FOOD RECOMMENDATION --------------------
if choice == "🍴 Food Recommendation":
    st.subheader("🍴 Food Recommendation (Restaurant-level)")
    food_input = st.text_input("Enter a Dish Name:")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])
    state_choice = st.selectbox("Select State:", ["West Bengal", "Rajasthan"])

    if st.button("🔍 Find Top Restaurants"):
        if not food_input.strip():
            st.warning("Please enter a food item first.")
        else:
            state_key = state_choice.lower().replace(" ", "_")
            state_restaurants = [r for r in restaurants.keys() if r.startswith(state_key)]
            # find counts per restaurant
            rest_counts = [(r, get_item_count(restaurants[r], food_input)) for r in state_restaurants]
            top2 = sorted(rest_counts, key=lambda x: x[1], reverse=True)[:2]
            top2 = [r for r in top2 if r[1] > 0]
            if not top2:
                st.warning(f"No transactions found for '{food_input}' in {state_choice}.")
            else:
                st.success("🏆 Top Restaurants for this Dish:")
                st.dataframe(pd.DataFrame(top2, columns=["Restaurant", "Transactions"]))
                restaurant_choice = st.selectbox("Select a Restaurant for Detailed Recommendations:", [r[0] for r in top2])

                if st.button("🍽️ Get Recommendations"):
                    rdf = restaurants[restaurant_choice]
                    rules = run_fp_with_fallback(rdf)
                    if rules.empty:
                        st.warning(f"No frequent patterns found for {restaurant_choice} (tried multiple supports).")
                    else:
                        item = food_input.lower()
                        # mask rules where item appears
                        mask = rules.apply(
                            lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                        or item in [i.lower() for i in list(row["consequents"])],
                            axis=1
                        )
                        matched = rules[mask]
                        if matched.empty:
                            st.warning(f"No frequent combinations found for '{food_input.title()}' in {restaurant_choice}.")
                        else:
                            # prepare markers
                            top_rest3 = [i[0] for i in top_items_in_restaurant(rdf, top_n=3)]
                            state_top5 = []
                            for r in state_restaurants:
                                state_top5 += [i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)]
                            state_top5 = set(state_top5)

                            suggestions = set()
                            for _, rrow in matched.iterrows():
                                for it in list(rrow["antecedents"]) + list(rrow["consequents"]):
                                    if it.lower() != item:
                                        suggestions.add(it.lower())

                            rows = []
                            for s in suggestions:
                                cnt = get_item_count(rdf, s)
                                if cnt <= 0:
                                    continue
                                meta = food_df[food_df["name"].str.lower() == s]
                                diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                                origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                                marker = ""
                                if s in top_rest3:
                                    marker += "🟣"
                                if s in state_top5:
                                    marker += "🟡"
                                rows.append({
                                    "Dish": f"{marker} {s.title()}".strip(),
                                    "Diet": diet,
                                    "State of Origin": origin,
                                    "Transactions": cnt
                                })
                            if not rows:
                                st.warning("No related items found in this restaurant.")
                            else:
                                df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                                if diet_choice.lower() != "both":
                                    # filter by diet and drop Diet column from display
                                    df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                    df_out = df_out.drop(columns=["Diet"], errors="ignore")
                                st.success(f"🍽️ Recommendations for '{food_input.title()}' in {restaurant_choice}:")
                                st.dataframe(df_out.reset_index(drop=True))

# -------------------- TOP PICKS --------------------
else:
    st.subheader("⭐ Top Picks by State / Restaurant")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])
    state_choice = st.selectbox("Select State:", ["West Bengal", "Rajasthan"])

    state_key = state_choice.lower().replace(" ", "_")
    state_restaurants = [r for r in restaurants.keys() if r.startswith(state_key)]
    # state items according to indian_food_use.csv
    state_items = food_df[food_df["state"].str.lower() == state_choice.lower()]["name"].str.lower().tolist()

    total_counts = {}
    rest_map = {}
    for rest in state_restaurants:
        df = restaurants[rest]
        for t in df["transaction_items"].astype(str):
            for i in parse_txn_items(t):
                if i not in state_items:
                    continue
                if diet_choice.lower() != "both":
                    d = food_df.loc[food_df["name"].str.lower() == i, "diet"]
                    if not d.empty and d.iloc[0].lower() != diet_choice.lower():
                        continue
                total_counts[i] = total_counts.get(i, 0) + 1
                rest_map[i] = rest

    if not total_counts:
        st.warning(f"No state-origin items found in transactions for {state_choice}.")
    else:
        top_items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        rows = []
        for rank, (item, cnt) in enumerate(top_items):
            meta = food_df[food_df["name"].str.lower() == item]
            diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
            rest = rest_map.get(item, "Unknown")
            marker = "🟡" if rank < 5 else ""
            rows.append({
                "Dish": f"{marker} {item.title()}".strip(),
                "Diet": diet,
                "Restaurant": rest,
                "Transactions": cnt
            })
        df_top = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).reset_index(drop=True)
        st.success(f"🔝 Top Picks in {state_choice}:")
        st.dataframe(df_top)

        chosen_item = st.selectbox("Select a Dish from Top Picks for Recommendations:", df_top["Dish"])
        restaurant_choice = st.selectbox("Select a Restaurant:", state_restaurants)

        if st.button("🍽️ Get Recommendations from Top Pick"):
            rdf = restaurants[restaurant_choice]
            rules = run_fp_with_fallback(rdf)
            if rules.empty:
                st.warning(f"No frequent patterns found for {restaurant_choice}.")
            else:
                # extract raw item (remove leading markers)
                item = chosen_item.lstrip("🟡🟣 ").lower()
                mask = rules.apply(
                    lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                or item in [i.lower() for i in list(row["consequents"])],
                    axis=1
                )
                matched = rules[mask]
                if matched.empty:
                    st.warning(f"No frequent combinations found for '{chosen_item}' in {restaurant_choice}.")
                else:
                    top_rest3 = [i[0] for i in top_items_in_restaurant(rdf, top_n=3)]
                    state_top5 = []
                    for r in state_restaurants:
                        state_top5 += [i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)]
                    state_top5 = set(state_top5)

                    suggestions = set()
                    for _, rrow in matched.iterrows():
                        for it in list(rrow["antecedents"]) + list(rrow["consequents"]):
                            if it.lower() != item:
                                suggestions.add(it.lower())

                    rows = []
                    for s in suggestions:
                        cnt = get_item_count(rdf, s)
                        if cnt <= 0:
                            continue
                        meta = food_df[food_df["name"].str.lower() == s]
                        diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                        origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                        marker = ""
                        if s in top_rest3:
                            marker += "🟣"
                        if s in state_top5:
                            marker += "🟡"
                        rows.append({
                            "Dish": f"{marker} {s.title()}".strip(),
                            "Diet": diet,
                            "State of Origin": origin,
                            "Transactions": cnt
                        })
                    if not rows:
                        st.warning("No related items found for that selection in the chosen restaurant.")
                    else:
                        df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                        if diet_choice.lower() != "both":
                            df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                            df_out = df_out.drop(columns=["Diet"], errors="ignore")
                        st.success(f"🍽️ Recommendations for '{chosen_item}' in {restaurant_choice}:")
                        st.dataframe(df_out.reset_index(drop=True))
