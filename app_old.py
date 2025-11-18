import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍴", layout="centered")
st.title("🍴 Indian Food Recommender System")
st.caption("Discover popular Indian food combinations and local trends using FP-Growth")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    food_df = pd.read_csv("indian_food.csv")
    food_df.columns = food_df.columns.str.strip().str.lower()

    txn_df = pd.read_csv("national_transactions.csv")
    txn_df.columns = txn_df.columns.str.strip().str.lower()

    return food_df, txn_df

food_df, txn_df = load_data()
st.success(f"✅ Nationwide dataset loaded successfully — {txn_df.shape[0]} transactions.")

# -------------------- PREPROCESS TRANSACTIONS --------------------
transactions = []
for t in txn_df["transaction_items"]:
    items = [x.split(" (")[0].strip().lower() for x in t.split(";") if x.strip()]
    transactions.append(items)

# One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=[c.lower() for c in te.columns_])

# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.header("⚙️ FP-Growth Parameters")
min_sup = st.sidebar.slider(
    "Minimum Support (adjust for rarity of combos)",
    0.001, 0.05, 0.005, 0.001, format="%.3f"
)

frequent_itemsets = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
rules = rules.sort_values(by="lift", ascending=False)

st.sidebar.write(f"📊 Frequent Itemsets: {len(frequent_itemsets)}")
st.sidebar.write(f"📈 Association Rules: {len(rules)}")

# -------------------- TRENDING ANALYSIS --------------------
freq_counter = {}
for txn in transactions:
    for item in txn:
        freq_counter[item] = freq_counter.get(item, 0) + 1
top_local = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
top_local_items = [i[0] for i in top_local[:10]]

# -------------------- MAIN UI --------------------
st.header("🍛 Step 1 • Choose Recommendation Type")
choice = st.radio("Select an option:", ["🍴 Food Recommendation", "🔥 Trending Locally"])

# ==============================================================
# 🍴 FOOD RECOMMENDATION
# ==============================================================
if choice == "🍴 Food Recommendation":
    st.subheader("🍴 Food Recommendation (Combo Suggestions)")
    dish_input = st.text_input("Enter a Dish Name:")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    if st.button("🔍 Get Recommendations"):
        if dish_input.strip():
            chosen = dish_input.lower().strip()
            matches = [col for col in df_encoded.columns if chosen in col]
            if not matches:
                st.warning(f"No match found for '{dish_input}'.")
            else:
                chosen = matches[0]
                count = int(df_encoded[chosen].sum())
                st.info(f"📊 '{chosen.title()}' appears in {count} transactions")

                mask = rules.apply(
                    lambda row: chosen in [i.lower() for i in list(row["antecedents"])]
                    or chosen in [i.lower() for i in list(row["consequents"])],
                    axis=1,
                )
                matched = rules[mask]

                if matched.empty:
                    st.warning(f"No frequent combinations found for '{dish_input.title()}'.")
                else:
                    suggestions = set()
                    for _, r in matched.iterrows():
                        for i in list(r["antecedents"]) + list(r["consequents"]):
                            if i.lower() != chosen:
                                suggestions.add(i.lower())

                    if diet_choice.lower() in ["vegetarian", "non vegetarian"]:
                        filtered = food_df[
                            (food_df["name"].str.lower().isin(suggestions))
                            & (food_df["diet"].str.lower() == diet_choice.lower())
                        ]
                    else:
                        filtered = food_df[food_df["name"].str.lower().isin(suggestions)]

                    if filtered.empty:
                        st.warning(f"No {diet_choice} items found related to '{dish_input.title()}'.")
                    else:
                        filtered["frequency"] = filtered["name"].str.lower().map(freq_counter).fillna(0)
                        filtered = filtered.sort_values(by="frequency", ascending=False)

                        filtered["Dish"] = filtered["name"].apply(
                            lambda x: ("⭐ " + x.title()) if x.lower() in top_local_items else x.title()
                        )

                        if diet_choice.lower() == "both":
                            filtered["Diet"] = filtered["diet"].str.title()
                            filtered = filtered[["Dish", "Diet", "frequency", "state"]]
                        else:
                            filtered = filtered[["Dish", "frequency", "state"]]

                        filtered.rename(columns={"frequency": "Transactions", "state": "State"}, inplace=True)
                        st.success(f"🍽️ Dish combinations related to '{dish_input.title()}':")
                        st.dataframe(filtered.head(10).reset_index(drop=True))
        else:
            st.warning("Please enter a dish name.")

# ==============================================================
# 🔥 TRENDING LOCALLY (Filtered by state)
# ==============================================================
else:
    st.subheader("🔥 Most Trending Locally")
    
    # Show state names in dropdown
    available_states = sorted(food_df["state"].dropna().unique())
    state_input = st.selectbox("Select State:", available_states)

    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    if st.button("📈 Show Trending Dishes"):
        if not state_input:
            st.warning("Please select a state.")
        else:
            # Filter dishes that belong to the selected state
            state_filtered = food_df[food_df["state"].str.lower() == state_input.strip().lower()]

            # Keep only those items present in both the dataset and state list
            result = pd.DataFrame(top_local, columns=["Dish", "Transactions"])
            result = result[result["Dish"].isin(state_filtered["name"].str.lower())]

            # Apply diet filter
            if diet_choice.lower() in ["vegetarian", "non vegetarian"]:
                result = result[result["Dish"].isin(
                    food_df[
                        (food_df["diet"].str.lower() == diet_choice.lower())
                        & (food_df["state"].str.lower() == state_input.strip().lower())
                    ]["name"].str.lower()
                )]

            if result.empty:
                st.warning(f"No trending dishes found for {state_input.title()} ({diet_choice}).")
            else:
                result = result.sort_values(by="Transactions", ascending=False)
                result["Dish"] = result["Dish"].apply(
                    lambda x: ("⭐ " + x.title()) if x.lower() in top_local_items else x.title()
                )
                result["State"] = state_input.title()
                st.success(f"🔝 Top Trending Items in {state_input.title()}:")
                st.dataframe(result[["Dish", "Transactions", "State"]].reset_index(drop=True))
