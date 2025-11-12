# app.py — FINAL VERSION (Correct State Name Fix)
import streamlit as st
import pandas as pd
import glob, os
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍴", layout="centered")
st.title("🍴 Statewise Restaurant Food Recommender System")
st.caption("Restaurant-level FP-Growth with Top Picks & Per-Restaurant Recommendations")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    food_df = pd.read_csv("indian_food_use.csv")
    food_df.columns = food_df.columns.str.strip().str.lower()
    food_df["name_norm"] = food_df["name"].astype(str).str.strip().str.lower()
    food_df["state_norm"] = food_df["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # This set defines all allowed state prefixes
    allowed_prefixes = {"west_bengal", "rajasthan", "gujarat", "maharashtra"}

    restaurant_files = sorted(glob.glob("*.csv"))
    exclude = {"indian_food_use.csv", "national_transactions.csv"}
    restaurant_files = [f for f in restaurant_files if os.path.basename(f) not in exclude]

    restaurants = {}
    for f in restaurant_files:
        key = os.path.splitext(os.path.basename(f))[0].strip()
        key_lower = key.lower()
        
        # --- FIXED LOGIC ---
        # Check if the filename *starts with* any of the allowed prefixes
        found_prefix = None
        for p in allowed_prefixes:
            if key_lower.startswith(p):
                found_prefix = p
                break # Found the prefix this file belongs to

        if found_prefix is None:
            continue # This file doesn't match any allowed prefix, so skip it
        # --- END FIXED LOGIC ---
            
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        if "transaction_items" not in df.columns:
            # This check is still important
            continue
        df["transaction_items"] = df["transaction_items"].astype(str)
        restaurants[key_lower] = df.copy()
        
    return food_df, restaurants, allowed_prefixes

# Pass allowed_prefixes out for the UI
food_df, restaurants, allowed_prefixes = load_data()
st.info(f"Loaded food metadata ({food_df.shape[0]} rows) and {len(restaurants)} restaurant datasets: {', '.join(restaurants.keys())}")

# -------------------- HELPERS --------------------
def parse_txn_items(t: str):
    return [x.split(" (")[0].strip().lower() for x in str(t).split(";") if x.strip()]

def get_item_count(df: pd.DataFrame, item: str):
    item = item.strip().lower()
    return sum(item in parse_txn_items(t) for t in df["transaction_items"].astype(str))

def top_items_in_restaurant(df: pd.DataFrame, top_n=10):
    counter = {}
    for t in df["transaction_items"].astype(str):
        for i in parse_txn_items(t):
            counter[i] = counter.get(i, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

def run_fp_with_fallback(df: pd.DataFrame, supports=(0.1, 0.05, 0.02, 0.01, 0.005, 0.001)):
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

def restaurants_by_state(state_prefix: str):
    # Now uses startswith, matching the loading logic
    return [r for r in restaurants.keys() if r.startswith(state_prefix)]

# -------------------- SESSION STATE --------------------
if "top2_for_query" not in st.session_state:
    st.session_state.top2_for_query = None
if "top2_query_dish" not in st.session_state:
    st.session_state.top2_query_dish = None
if "top2_state" not in st.session_state:
    st.session_state.top2_state = None

# -------------------- UI --------------------
st.header("🍛 Step 1 • Choose Your Action")
choice = st.radio("Select an option:", ["🍴 Food Recommendation", "⭐ Top Picks"])

# --- Helper to get available prefixes from loaded files ---
def get_available_prefixes(loaded_restaurant_keys, all_prefixes):
    found_prefixes = set()
    for key in loaded_restaurant_keys:
        for prefix in all_prefixes:
            if key.startswith(prefix):
                found_prefixes.add(prefix)
                break
    return sorted(list(found_prefixes))

# -------------------- FOOD RECOMMENDATION --------------------
if choice == "🍴 Food Recommendation":
    st.subheader("🍴 Food Recommendation (Restaurant-level)")
    food_input = st.text_input("Enter a Dish Name:")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    # --- FIXED PREFIX LOGIC ---
    available_state_prefixes = get_available_prefixes(restaurants.keys(), allowed_prefixes)
    state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}
    
    if not available_state_prefixes:
        st.warning("No restaurant data files found. Please check your CSV files.")
    else:
        state_choice_prefix = st.selectbox("Select State:", available_state_prefixes, format_func=lambda x: state_display_map[x])
        state_choice = state_display_map[state_choice_prefix]

        if st.button("🔍 Find Top Restaurants"):
            state_restaurants = restaurants_by_state(state_choice_prefix)
            rest_counts = [(r, get_item_count(restaurants[r], food_input)) for r in state_restaurants]

            # Check total transactions *before* displaying anything
            total_txns = sum(count for _, count in rest_counts)

            if total_txns == 0:
                # Case 1: No transactions found anywhere
                st.warning(f"No transactions found for '{food_input.title()}' in {state_choice}.")
                # Clear any previous successful search
                st.session_state.top2_for_query = None
                st.session_state.top2_query_dish = None
                st.session_state.top2_state = None
            
            else:
                # Case 2: At least one transaction was found
                # Now, we can proceed with displaying the table and recommendations
                
                # Sort and mark top 2 restaurants with ⭐
                sorted_counts = sorted(rest_counts, key=lambda x: x[1], reverse=True)
                top2_names = [r[0] for r in sorted_counts[:2]]
                unified_table = []
                for name, count in sorted_counts:
                    star = "⭐" if name in top2_names else ""
                    unified_table.append({
                        "Restaurant": f"{name.title()} {star}".strip(),
                        "Transactions": count
                    })

                st.success(f"📊 Item count & 🏆 Top Restaurants for '{food_input.title()}' in {state_choice}:")
                st.dataframe(pd.DataFrame(unified_table))

                # Filter top 2 to *only* include those with > 0 txns for the dropdown
                top2 = [r for r in sorted_counts[:2] if r[1] > 0]
                
                # We know top2 is not empty because total_txns > 0
                st.session_state.top2_for_query = top2
                st.session_state.top2_query_dish = food_input.strip().lower()
                st.session_state.top2_state = state_choice_prefix

        if st.session_state.top2_for_query:
            top2 = st.session_state.top2_for_query
            choice_list = [r[0] for r in top2]
            restaurant_choice = st.selectbox("Select a Restaurant for Detailed Recommendations:", choice_list)

            if restaurant_choice and st.session_state.top2_query_dish:
                rdf = restaurants[restaurant_choice]
                rules = run_fp_with_fallback(rdf)
                if rules.empty:
                    st.warning(f"No frequent patterns found for {restaurant_choice}.")
                else:
                    item = st.session_state.top2_query_dish
                    mask = rules.apply(
                        lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                    or item in [i.lower() for i in list(row["consequents"])],
                        axis=1
                    )
                    matched = rules[mask]
                    if matched.empty:
                        st.warning(f"No frequent combinations found for '{item.title()}' in {restaurant_choice}.")
                    else:
                        state_prefix = st.session_state.top2_state
                        state_restaurants = restaurants_by_state(state_prefix)
                        top_rest3 = [i[0] for i in top_items_in_restaurant(rdf, top_n=3)]
                        state_top5 = set()
                        for r in state_restaurants:
                            state_top5.update([i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)])

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
                            meta = food_df[food_df["name_norm"] == s]
                            diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                            origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                            marker = ""
                            if s in top_rest3:
                                marker += "🟣"
                            if s in state_top5:
                                marker += "🟡"
                            rows.append({
                                "Dish": f"{marker} {s.title()}",
                                "Diet": diet,
                                "State of Origin": origin,
                                "Transactions": cnt
                            })
                        if rows:
                            df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                            if diet_choice.lower() != "both":
                                df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                df_out = df_out.drop(columns=["Diet"], errors="ignore")
                            st.success(f"🍽️ Recommendations for '{item.title()}' in {restaurant_choice}:")
                            st.dataframe(df_out.reset_index(drop=True))
                        else:
                            st.warning(f"No related items found for '{item.title()}' in {restaurant_choice}.")

        # --- LEGEND FOR FOOD RECOMMENDATION PAGE ---
        st.markdown("---")
        st.markdown("**Legend:**")
        st.markdown("- ⭐ **Gold** = One of the Top 2 restaurants for the chosen dish.")
        st.markdown("- 🟣 **Purple** = Top 3 items in the selected restaurant.")
        st.markdown("- 🟡 **Yellow** = Top 5 items across restaurants in the selected state.")

# -------------------- TOP PICKS --------------------
else:
    st.subheader("⭐ Top Picks by State / Restaurant")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    # --- FIXED PREFIX LOGIC ---
    available_state_prefixes = get_available_prefixes(restaurants.keys(), allowed_prefixes)
    state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}
    
    if not available_state_prefixes:
        st.warning("No restaurant data files found. Please check your CSV files.")
    else:
        state_choice_prefix = st.selectbox("Select State:", available_state_prefixes, format_func=lambda x: state_display_map[x])
        state_choice = state_display_map[state_choice_prefix]

        if state_choice_prefix:
            state_restaurants = restaurants_by_state(state_choice_prefix)
            # Filter metadata for items originating from the selected state
            state_items = food_df[food_df["state_norm"] == state_choice_prefix]["name_norm"].tolist()

            total_counts = {}
            per_item_rest_counts = {}

            for rest in state_restaurants:
                df = restaurants[rest]
                for t in df["transaction_items"].astype(str):
                    for i in parse_txn_items(t):
                        # Only count items that *originate* from this state
                        if i not in state_items:
                            continue
                        # Filter by diet if not "Both"
                        if diet_choice.lower() != "both":
                            d = food_df.loc[food_df["name_norm"] == i, "diet"]
                            if not d.empty and d.iloc[0].lower() != diet_choice.lower():
                                continue
                        # Add to total count for the state
                        total_counts[i] = total_counts.get(i, 0) + 1
                        # Add to per-restaurant count
                        per_item_rest_counts.setdefault(i, {})
                        per_item_rest_counts[i][rest] = per_item_rest_counts[i].get(rest, 0) + 1

            if not total_counts:
                st.warning(f"No state-origin items found in transactions for {state_choice} matching the diet filter.")
            else:
                # Find the best restaurant for each item
                best_rest_for_item = {i: max(r.items(), key=lambda x: x[1])[0] for i, r in per_item_rest_counts.items()}
                # Sort items by their total state-wide count
                top_items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                rows = []
                for rank, (item, total_cnt) in enumerate(top_items):
                    meta = food_df[food_df["name_norm"] == item]
                    diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                    rest = best_rest_for_item.get(item, "Unknown")
                    marker = "🟡" if rank < 5 else ""
                    
                    rows.append({
                        "Dish": f"{marker} {item.title()}",
                        "Diet": diet,
                        "Restaurant": rest,
                        "Overall Transactions": total_cnt # Renamed column
                    })
                
                df_top = pd.DataFrame(rows).sort_values(by="Overall Transactions", ascending=False).reset_index(drop=True)
                st.success(f"🔝 Top Picks in {state_choice}:")
                st.dataframe(df_top)

                # --- Recommendation based on Top Pick ---
                chosen_item = st.selectbox("Select a Dish from Top Picks for Recommendations:", df_top["Dish"])
                raw_item = chosen_item.lstrip("🟡🟣 ").lower()
                # Find the default restaurant (the best one) for the selected item
                default_rest = best_rest_for_item.get(raw_item, state_restaurants[0])
                try:
                    default_index = state_restaurants.index(default_rest)
                except ValueError:
                    default_index = 0 # Fallback if restaurant isn't in list
                
                restaurant_choice = st.selectbox("Select a Restaurant:", state_restaurants, index=default_index)

                if st.button("🍽️ Get Recommendations from Top Pick"):
                    rdf = restaurants[restaurant_choice]
                    rules = run_fp_with_fallback(rdf)
                    
                    if rules.empty:
                        st.warning(f"No frequent patterns found for {restaurant_choice}.")
                    else:
                        item = raw_item
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
                            state_top5 = set()
                            for r in state_restaurants:
                                state_top5.update([i[0] for i in top_items_in_restaurant(restaurants[r], top_n=5)])
                                
                            # Get all unique items from matching rules
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
                                meta = food_df[food_df["name_norm"] == s]
                                diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                                origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
                                marker = ""
                                if s in top_rest3:
                                    marker += "🟣"
                                if s in state_top5:
                                    marker += "🟡"
                                rows.append({
                                    "Dish": f"{marker} {s.title()}",
                                    "Diet": diet,
                                    "State of Origin": origin,
                                    "Transactions": cnt
                                })
                            
                            if not rows:
                                st.warning(f"No related items found for '{chosen_item}' in {restaurant_choice}.")
                            else:
                                df_out = pd.DataFrame(rows).sort_values(by="Transactions", ascending=False).head(10)
                                if diet_choice.lower() != "both":
                                    df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                    df_out = df_out.drop(columns=["Diet"], errors="ignore")
                                st.success(f"🍽️ Recommendations for '{chosen_item}' in {restaurant_choice}:")
                                st.dataframe(df_out.reset_index(drop=True))

        # --- LEGEND FOR TOP PICKS PAGE ---
        st.markdown("---")
        st.markdown("**Legend:**")
        # No '⭐' here
        st.markdown("- 🟣 **Purple** = Top 3 items in the selected restaurant.")
        st.markdown("- 🟡 **Yellow** = Top 5 items across restaurants in the selected state.")