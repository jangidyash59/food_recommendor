import streamlit as st
import pandas as pd
import glob, os, re
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍴", layout="centered")

# --- Custom CSS for Styling (Points 1, 2, 3, 8) ---
st.markdown("""
<style>
/* --- 1. Custom Title Gradient --- */
div[data-testid="stAppViewContainer"] > section > div[data-testid="stVerticalBlock"] > div:first-child {
    background: linear-gradient(90deg, #004d99, #66b3ff);
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}
div[data-testid="stAppViewContainer"] h1 {
    color: white; /* Make title text white */
}
div[data-testid="stAppViewContainer"] .stCaption {
    color: white !important; /* Make caption text white */
    opacity: 0.9;
}

/* --- 8. Table Styling (Headers, Zebra, Hover) --- */
div[data-testid="stDataFrame"] a[class^="col_heading"] {
    color: #004d99; /* Prominent header color */
    font-weight: bold;
    font-size: 1.1em;
}
div[data-testid="stDataFrame"] div[class^="row"]:nth-child(even) {
    background-color: #f5faff; /* Zebra striping - light blue */
}
div[data-testid="stDataFrame"] div[class^="row"]:hover {
    background-color: #e6f7ff; /* Lighter blue on hover */
    cursor: default;
}

/* --- 3. Spacing Between Sections --- */
.stSubheader {
    margin-top: 30px; /* Add space above subheaders */
    margin-bottom: 10px;
}
div[data-testid="stRadio"] {
    margin-bottom: 30px; /* Add space after radio buttons */
}
div[data-testid="stHorizontalBlock"] {
    margin-bottom: 20px; /* Add space between form elements */
}
</style>
""", unsafe_allow_html=True)


st.title("🍴 Statewise Restaurant Food Recommender System")
st.caption("Restaurant-level FP-Growth with Top Picks & Per-Restaurant Recommendations")


# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.header("Analysis Settings")
# --- 5. FP-Growth Tooltip ---
min_support = st.sidebar.slider(
    "Select Minimum Support",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f",
    help="FP-Growth 'Minimum Support' transaction ka woh percentage hai jismein itemset ka milna zaroori hai. "
         "Kam value (e.g., 0.001) se zyada patterns milenge. "
         "Zyada value (e.g., 0.05) se sirf strong patterns milenge."
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    try:
        # --- FIX for KeyError: 'name' ---
        # Add header=None and manually name the columns
        column_names = ['name', 'diet', 'state']
        food_df = pd.read_csv("indian_food.csv", header=None, names=column_names)
        # --- END FIX ---
        
    except FileNotFoundError:
        st.error("Error: `indian_food.csv` not found. Please add it to the directory.")
        return pd.DataFrame(), {}, set()
        
    food_df.columns = food_df.columns.str.strip().str.lower()
    food_df["name_norm"] = food_df["name"].astype(str).str.strip().str.lower()
    food_df["state_norm"] = food_df["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")

    restaurant_files = sorted(glob.glob("*.csv"))
    exclude = {"indian_food.csv", "national_transactions.csv"}
    restaurant_files = [f for f in restaurant_files if os.path.basename(f) not in exclude]

    restaurants = {}
    available_state_prefixes = set()
    
    file_pattern = re.compile(r'(.+)_Restaurant_#\d', re.IGNORECASE)

    for f in restaurant_files:
        key = os.path.splitext(os.path.basename(f))[0].strip()
        key_lower = key.lower()
        match = file_pattern.match(key_lower)
        
        if match:
            state_prefix = match.group(1)
            available_state_prefixes.add(state_prefix)
            
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip().str.lower()
                if "transaction_items" not in df.columns:
                    st.warning(f"Skipping '{f}': No 'transaction_items' column.")
                    continue
                df["transaction_items"] = df["transaction_items"].astype(str)
                restaurants[key_lower] = df.copy()
            except Exception as e:
                st.warning(f"Error loading '{f}': {e}")

    return food_df, restaurants, sorted(list(available_state_prefixes))


food_df, restaurants, available_state_prefixes = load_data()

if not restaurants:
    st.error("No restaurant transaction files found. Please add files in the format 'State_Name_Restaurant_#1.csv'.")
else:
    st.info(f"✅ Loaded food metadata ({food_df.shape[0]} rows) and {len(restaurants)} restaurant datasets.")


# -------------------- HELPERS --------------------
def parse_txn_items(t: str):
    t_cleaned = str(t).strip().strip('"')
    return [item.strip().lower() for item in t_cleaned.split(',') if item.strip()]

def get_item_count(df: pd.DataFrame, item: str):
    item = item.strip().lower()
    return sum(item in parse_txn_items(t) for t in df["transaction_items"].astype(str))

def top_items_in_restaurant(df: pd.DataFrame, top_n=10):
    counter = {}
    for t in df["transaction_items"].astype(str):
        for i in parse_txn_items(t):
            counter[i] = counter.get(i, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

def run_fpgrowth(df: pd.DataFrame, support: float):
    txns = [parse_txn_items(t) for t in df["transaction_items"].astype(str)]
    if not txns:
        return pd.DataFrame()
    try:
        te = TransactionEncoder()
        te_ary = te.fit(txns).transform(txns)
        df_enc = pd.DataFrame(te_ary, columns=te.columns_)
        
        freq_items = fpgrowth(df_enc, min_support=support, use_colnames=True)
        
        if freq_items.empty:
            return pd.DataFrame()
            
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.05)
        return rules
    except Exception:
        return pd.DataFrame()

def restaurants_by_state(state_prefix: str):
    return [r for r in restaurants.keys() if r.startswith(state_prefix.lower())]

# -------------------- SESSION STATE KEYS --------------------
if "top2_for_query" not in st.session_state:
    st.session_state.top2_for_query = None
if "top2_query_dish" not in st.session_state:
    st.session_state.top2_query_dish = None
if "top2_state" not in st.session_state:
    st.session_state.top2_state = None

# -------------------- UI --------------------
st.header("🍛 Step 1 • Choose Your Action")
choice = st.radio("Select an option:", ["🍴 Food Recommendation", "⭐ Top Picks"])
st.markdown("---") # --- 3. Spacing ---

# -------------------- FOOD RECOMMENDATION --------------------
if choice == "🍴 Food Recommendation":
    # --- 4. Catchy Heading ---
    st.subheader("🚀 Find Your Next Favorite Meal")

    unique_dishes = sorted(food_df["name_norm"].dropna().unique().tolist())
    unique_dishes_display = [name.title() for name in unique_dishes]

    food_input = st.selectbox(
        "Enter or Select a Dish Name:",
        options=[""] + unique_dishes_display,
        index=0,
        help="Start typing to search for a dish"
    ).strip()

    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    if not available_state_prefixes:
        st.warning("No restaurant data files found.")
    else:
        state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}

        default_index = 0
        if food_input:
            item = food_input.lower()
            state_counts = {}
            for prefix in available_state_prefixes:
                total = 0
                for rest in restaurants_by_state(prefix):
                    total += get_item_count(restaurants[rest], item)
                state_counts[prefix] = total

            if any(state_counts.values()):
                max_prefix = max(state_counts, key=state_counts.get)
                default_index = available_state_prefixes.index(max_prefix)
                st.info(
                    f"✅ Auto-selected state: **{state_display_map[max_prefix]}** — most transactions found for '{food_input.title()}'."
                )
            else:
                st.warning(f"No transactions found for '{food_input.title()}' in any state.")

        state_choice_prefix = st.selectbox(
            "Select State:",
            available_state_prefixes,
            index=default_index,
            format_func=lambda x: state_display_map[x]
        )
        state_choice = state_display_map[state_choice_prefix]

        if st.button("🔍 Find Top Restaurants"):
            state_restaurants = restaurants_by_state(state_choice_prefix)
            rest_counts = [(r, get_item_count(restaurants[r], food_input)) for r in state_restaurants]

            total_txns = sum(count for _, count in rest_counts)
            if total_txns == 0:
                st.warning(f"No transactions found for '{food_input.title()}' in {state_choice}.")
                st.session_state.top2_for_query = None
                st.session_state.top2_query_dish = None
                st.session_state.top2_state = None
            else:
                sorted_counts = sorted(rest_counts, key=lambda x: x[1], reverse=True)
                top2_names = [r[0] for r in sorted_counts[:2]]
                unified_table = []
                for name, count in sorted_counts:
                    # --- 2. Marker Change ---
                    star = "🏆" if name in top2_names else ""
                    display_name = name.replace('_', ' ').title()
                    # --- 7. Transaction Numbers ---
                    unified_table.append({"Restaurant": f"{display_name} {star}".strip(), "Orders": count})

                st.success(f"📊 Item count & 🏆 Top Restaurants for '{food_input.title()}' in {state_choice}:")
                st.dataframe(pd.DataFrame(unified_table))

                top2 = [r for r in sorted_counts[:2] if r[1] > 0]
                st.session_state.top2_for_query = top2
                st.session_state.top2_query_dish = food_input.lower()
                st.session_state.top2_state = state_choice_prefix

        st.write("") # --- 3. Spacing ---
        
        if st.session_state.top2_for_query:
            top2 = st.session_state.top2_for_query
            choice_map = {r[0]: r[0].replace('_', ' ').title() for r in top2}
            choice_list = list(choice_map.keys())
            
            restaurant_choice_key = st.selectbox(
                "Select a 🏆 Restaurant for Detailed Recommendations:",
                choice_list,
                format_func=lambda x: choice_map[x]
            )
            restaurant_choice_display = choice_map[restaurant_choice_key]

            if restaurant_choice_key and st.session_state.top2_query_dish:
                rdf = restaurants[restaurant_choice_key]
                
                # Run FP-Growth ONCE with the slider value
                rules = run_fpgrowth(rdf, support=min_support)

                if rules.empty:
                    # --- 6. Error Message ---
                    st.warning(f"😥 No frequent combinations found in {restaurant_choice_display} at {min_support:.3f} support. Sidebar se support value kam karke try karein.")
                else:
                    item = st.session_state.top2_query_dish
                    mask = rules.apply(
                        lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                    or item in [i.lower() for i in list(row["consequents"])],
                        axis=1
                    )
                    matched = rules[mask]

                    if matched.empty:
                        # --- 6. Error Message ---
                        st.warning(f"😥 No specific combinations found for '{item.title()}' in {restaurant_choice_display} at this support level. Try a different dish or lower support.")
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
                            
                            # --- 2. Marker Change ---
                            marker = ""
                            if s in top_rest3:
                                marker += "🔹" # Blue Diamond
                            if s in state_top5:
                                marker += "🔸" # Orange Diamond
                                
                            rows.append({
                                "Dish": f"{marker} {s.title()}",
                                "Diet": diet,
                                "State of Origin": origin,
                                # --- 7. Transaction Numbers ---
                                "Orders": cnt
                            })
                            
                        if rows:
                            df_out = pd.DataFrame(rows).sort_values(by="Orders", ascending=False).head(10)
                            if diet_choice.lower() != "both":
                                df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                df_out = df_out.drop(columns=["Diet"], errors="ignore")
                            st.success(f"🍽️ Recommendations for '{item.title()}' in {restaurant_choice_display}:")
                            st.dataframe(df_out.reset_index(drop=True))
                        else:
                            # --- 6. Error Message ---
                            st.warning(f"😥 No related items found for '{item.title()}' after filtering.")

        # legend
        st.markdown("---")
        st.markdown("**Legend:**")
        st.markdown("- 🏆 **Trophy** = Top 2 restaurant for the chosen dish.")
        st.markdown("- 🔹 **Blue Diamond** = Top 3 item in the selected restaurant.")
        st.markdown("- 🔸 **Orange Diamond** = Top 5 item across all restaurants in the state.")

# -------------------- TOP PICKS SECTION --------------------
else:
    # --- 4. Catchy Heading ---
    st.subheader("🌟 Discover Top Picks by State")
    diet_choice = st.selectbox("Filter by Diet Type:", ["Both", "Vegetarian", "Non Vegetarian"])

    if not available_state_prefixes:
        st.warning("No restaurant data files found. Please check your CSV files.")
    else:
        state_display_map = {p: p.replace("_", " ").title() for p in available_state_prefixes}
        state_choice_prefix = st.selectbox("Select State:", available_state_prefixes, format_func=lambda x: state_display_map[x])
        state_choice = state_display_map[state_choice_prefix]

        state_restaurants = restaurants_by_state(state_choice_prefix)
        state_items = food_df[food_df["state_norm"] == state_choice_prefix]["name_norm"].tolist()

        total_counts = {}
        per_item_rest_counts = {}
        for rest in state_restaurants:
            if rest not in restaurants: continue
            df = restaurants[rest]
            for t in df["transaction_items"].astype(str):
                for i in parse_txn_items(t):
                    if i not in state_items:
                        continue
                    if diet_choice.lower() != "both":
                        d = food_df.loc[food_df["name_norm"] == i, "diet"]
                        if not d.empty and d.iloc[0].lower() != diet_choice.lower():
                            continue
                    total_counts[i] = total_counts.get(i, 0) + 1
                    per_item_rest_counts.setdefault(i, {})
                    per_item_rest_counts[i][rest] = per_item_rest_counts[i].get(rest, 0) + 1

        if not total_counts:
            st.warning(f"No state-origin items found in transactions for {state_choice} matching the diet filter.")
        else:
            best_rest_for_item = {i: max(r.items(), key=lambda x: x[1])[0] for i, r in per_item_rest_counts.items()}
            top_items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            rows = []
            for rank, (item, total_cnt) in enumerate(top_items):
                meta = food_df[food_df["name_norm"] == item]
                diet = meta["diet"].iloc[0].title() if not meta.empty else "Unknown"
                rest_key = best_rest_for_item.get(item, "Unknown")
                rest_display = rest_key.replace('_', ' ').title()
                # --- 2. Marker Change ---
                marker = "🔸" if rank < 5 else ""
                rows.append({
                    "Dish": f"{marker} {item.title()}",
                    "Diet": diet,
                    "Top Restaurant": rest_display,
                    # --- 7. Transaction Numbers ---
                    "Total Orders": total_cnt
                })
            df_top = pd.DataFrame(rows).sort_values(by="Total Orders", ascending=False).reset_index(drop=True)
            st.success(f"🔝 Top Picks in {state_choice}:")
            st.dataframe(df_top)
            
            st.markdown("---") # --- 3. Spacing ---
            st.subheader("Get Recommendations from a Top Pick")

            chosen_item = st.selectbox("Select a Dish from Top Picks for Recommendations:", df_top["Dish"])
            # --- 2. Marker Change ---
            raw_item = chosen_item.lstrip("🔸🔹 ").lower()
            default_rest = best_rest_for_item.get(raw_item, state_restaurants[0])
            try:
                default_index = state_restaurants.index(default_rest)
            except ValueError:
                default_index = 0
            
            rest_choice_map = {r: r.replace('_', ' ').title() for r in state_restaurants}
            
            restaurant_choice_key = st.selectbox(
                "Select a Restaurant:",
                state_restaurants,
                index=default_index,
                format_func=lambda x: rest_choice_map[x]
            )
            restaurant_choice_display = rest_choice_map[restaurant_choice_key]

            if st.button("🍽️ Get Recommendations from Top Pick"):
                rdf = restaurants[restaurant_choice_key]

                # Run FP-Growth ONCE with the slider value
                rules = run_fpgrowth(rdf, support=min_support)
                used_support = min_support

                if rules.empty:
                    # --- 6. Error Message ---
                    st.warning(f"😥 No frequent combinations found in {restaurant_choice_display} at {min_support:.3f} support. Sidebar se support value kam karke try karein.")
                else:
                    item = raw_item
                    mask = rules.apply(
                        lambda row: item in [i.lower() for i in list(row["antecedents"])]
                                    or item in [i.lower() for i in list(row["consequents"])],
                        axis=1
                    )
                    matched = rules[mask]

                    if matched.empty:
                        # --- 6. Error Message ---
                        st.warning(f"😥 No specific combinations found for '{item.title()}' in {restaurant_choice_display} at this support level.")
                    else:
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
                            
                            # --- 2. Marker Change ---
                            marker = ""
                            if s in top_rest3:
                                marker += "🔹" # Blue
                            if s in state_top5:
                                marker += "🔸" # Orange
                                
                            rows.append({
                                "Dish": f"{marker} {s.title()}",
                                "Diet": diet,
                                "State of Origin": origin,
                                # --- 7. Transaction Numbers ---
                                "Orders": cnt
                            })
                        if rows:
                            df_out = pd.DataFrame(rows).sort_values(by="Orders", ascending=False).head(10)
                            if diet_choice.lower() != "both":
                                df_out = df_out[df_out["Diet"].str.lower() == diet_choice.lower()]
                                df_out = df_out.drop(columns=["Diet"], errors="ignore")
                            
                            msg = f"🍽️ Recommendations for '{item.title()}' in {restaurant_choice_display} (min_support={used_support:.3f})"
                            st.success(msg + ":")
                            st.dataframe(df_out.reset_index(drop=True))
                        else:
                            # --- 6. Error Message --
                            st.warning(f"😥 No related items found for '{item.title()}' after filtering.")