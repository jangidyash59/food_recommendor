import glob
import os
import re
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
import base64

def load_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

title_base64 = load_base64_image("title.png")

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=DeprecationWarning)
# ---------------------------------------------------------------------
# Page config + global styles
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="CraveMap - Food Recommender",
    page_icon="🍴CM",
    layout="wide",
)


GLOBAL_CSS = """
<style>
:root {
    --primary-blue: #0a4d68;
    --accent-peach: #f7a76c;
    --bg-mint: #f2fbf9;
    --card-shadow: 0 18px 40px rgba(15, 40, 81, 0.08);
    --table-zebra: #fef8f3;
}

body {
    font-family: "Segoe UI", sans-serif;
    background: var(--bg-mint);
}

.block-container {
    padding-top: 1.5rem;
}

.metric-card {
    background: white;
    padding: 1.25rem 1.5rem;
    border-radius: 18px;
    box-shadow: var(--card-shadow);
}

div[data-testid="stMarkdownContainer"] h2 {
    font-weight: 700;
    color: var(--primary-blue);
}

/* Table polishing */
div[data-testid="stDataFrame"] table {
    border-spacing: 0 !important;
    width: 100%;
}

div[data-testid="stDataFrame"] th {
    background-color: #e8f2ff !important;
    font-weight: 700 !important;
    color: #073763 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #351f57 !important;
    color: #ffffff !important;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #4a2a78 !important;
    color: #ffffff !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #5c3591 !important;
    transition: background 0.2s ease-in-out;
}

div[data-testid="stDataFrame"] td {
    border-bottom: 1px solid #edf0f7 !important;
    font-size: 0.95rem;
    color: #ffffff !important;
}

/* Streamlit tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: white;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.9rem;
    box-shadow: var(--card-shadow);
}

.info-chip {
    background: rgba(10, 77, 104, 0.08);
    padding: 0.4rem 0.75rem;
    border-radius: 10px;
    font-size: 0.9rem;
}

.hero-title {
    font-family: "Playfair Display", "Times New Roman", serif;
    font-size: 3rem;
    font-weight: 600;
    color: var(--primary-blue);
    margin-bottom: 0.5rem;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str], Dict[str, str]]:
    try:
        food_df = pd.read_csv(
            "indian_food.csv",
            header=None,
            names=["name", "diet", "state"],
        )
    except FileNotFoundError:
        st.error("`indian_food.csv` missing. Please add it to the project directory.")
        return pd.DataFrame(), {}, [], {}

    food_df.columns = food_df.columns.str.strip().str.lower()
    food_df["name_norm"] = food_df["name"].astype(str).str.strip().str.lower()
    food_df["state_norm"] = (
        food_df["state"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    food_df["diet"] = (
        food_df["diet"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    restaurant_files = [
        f
        for f in glob.glob("*.csv")
        if os.path.basename(f).lower() not in {"indian_food.csv", "national_transactions.csv"}
    ]

    restaurants: Dict[str, pd.DataFrame] = {}
    prefixes: set[str] = set()
    file_pattern = re.compile(r"(.+)_restaurant_#\d", re.IGNORECASE)

    for file_path in sorted(restaurant_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        match = file_pattern.match(file_name.lower())
        if not match:
            continue

        state_prefix = match.group(1)
        prefixes.add(state_prefix)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            if "transaction_items" not in df.columns:
                st.warning(f"Skipping '{file_name}': `transaction_items` column missing.")
                continue
            df["transaction_items"] = df["transaction_items"].astype(str)
            restaurants[file_name.lower()] = df
        except Exception as exc:  # pragma: no cover - defensive messaging
            st.warning(f"Unable to load '{file_name}': {exc}")

    prefix_list = sorted(prefixes)
    display_map = {p: p.replace("_", " ").title() for p in prefix_list}
    return food_df, restaurants, prefix_list, display_map


food_df, restaurants, available_state_prefixes, display_map = load_data()

if not restaurants:
    st.stop()

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def parse_transaction_items(transaction: str) -> List[str]:
    cleaned = str(transaction).strip().strip('"')
    return [chunk.strip().lower() for chunk in cleaned.split(",") if chunk.strip()]


def restaurants_by_state(prefix: str) -> List[str]:
    return [name for name in restaurants.keys() if name.startswith(prefix.lower())]


def item_order_count(df: pd.DataFrame, item: str) -> int:
    return sum(item in parse_transaction_items(txn) for txn in df["transaction_items"])


def top_items_in_restaurant(df: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, int]]:
    counter: Dict[str, int] = {}
    for txn in df["transaction_items"]:
        for item in parse_transaction_items(txn):
            counter[item] = counter.get(item, 0) + 1
    return sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def run_fp_growth(df: pd.DataFrame, support: float) -> pd.DataFrame:
    transactions = [parse_transaction_items(txn) for txn in df["transaction_items"]]
    if not transactions:
        return pd.DataFrame()

    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded, columns=te.columns_)
    freq_items = fpgrowth(encoded_df, min_support=support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()
    return association_rules(freq_items, metric="confidence", min_threshold=0.05)


def auto_select_state_for_dish(dish: str) -> Tuple[Optional[str], Dict[str, int]]:
    if not dish:
        return None, {}
    dish = dish.lower()
    counts: Dict[str, int] = {}
    for prefix in available_state_prefixes:
        total = 0
        for rest_key in restaurants_by_state(prefix):
            total += item_order_count(restaurants[rest_key], dish)
        counts[prefix] = total
    if not counts or max(counts.values()) == 0:
        return None, counts
    best = max(counts, key=counts.get)
    return best, counts


def summarize_restaurant_orders(state_prefix: str, dish: str) -> List[Dict[str, int]]:
    rows = []
    for rest_key in restaurants_by_state(state_prefix):
        df = restaurants[rest_key]
        count = item_order_count(df, dish)
        rows.append(
            {
                "key": rest_key,
                "Restaurant": rest_key.replace("_", " ").title(),
                "Orders": count,
            }
        )
    rows.sort(key=lambda row: row["Orders"], reverse=True)
    return rows


def build_pairing_table(
    rest_key: str,
    focus_item: str,
    diet_filter: str,
    state_prefix: str,
    support: float,
) -> Tuple[pd.DataFrame, str]:
    rdf = restaurants[rest_key]
    rules = run_fp_growth(rdf, support)
    if rules.empty:
        return pd.DataFrame(), f"No frequent itemsets at support={support:.3f}."

    focus_item = focus_item.lower()

    def contains_focus(items: List[str]) -> bool:
        return focus_item in [it.lower() for it in items]

    mask = rules.apply(
        lambda row: contains_focus(list(row["antecedents"]))
        or contains_focus(list(row["consequents"])),
        axis=1,
    )
    matched = rules[mask]
    if matched.empty:
        return pd.DataFrame(), "No rule contains the selected dish. Try lowering support."

    local_top = {item for item, _ in top_items_in_restaurant(rdf, top_n=3)}
    state_top: set[str] = set()
    for rest in restaurants_by_state(state_prefix):
        state_top.update(item for item, _ in top_items_in_restaurant(restaurants[rest], top_n=5))

    suggestions: set[str] = set()
    for _, row in matched.iterrows():
        for bucket in ("antecedents", "consequents"):
            for candidate in row[bucket]:
                candidate = candidate.lower()
                if candidate != focus_item:
                    suggestions.add(candidate)

    data_rows = []
    for suggestion in suggestions:
        orders = item_order_count(rdf, suggestion)
        if orders <= 0:
            continue
        meta = food_df[food_df["name_norm"] == suggestion]
        diet = meta["diet"].iloc[0] if not meta.empty else "Unknown"
        origin = meta["state"].iloc[0].title() if not meta.empty else "Unknown"
        marker = ""
        if suggestion in local_top:
            marker += "🔹 "
        if suggestion in state_top:
            marker += "🔸 "
        data_rows.append(
            {
                "Dish": f"{marker}{suggestion.title()}".strip(),
                "Diet": diet,
                "State of Origin": origin,
                "Orders": orders,
            }
        )

    if not data_rows:
        return pd.DataFrame(), "No matching orders after filtering."

    result = pd.DataFrame(data_rows).sort_values("Orders", ascending=False)
    if diet_filter.lower() != "both":
        result = result[result["Diet"].str.lower() == diet_filter.lower()]
        result = result.drop(columns=["Diet"], errors="ignore")

    return result.reset_index(drop=True), ""


def aggregate_state_top_picks(state_prefix: str, diet_filter: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    state_items = food_df[food_df["state_norm"] == state_prefix]["name_norm"].tolist()
    total_counts: Dict[str, int] = {}
    per_item_restaurant: Dict[str, Dict[str, int]] = {}

    for rest_key in restaurants_by_state(state_prefix):
        df = restaurants[rest_key]
        for txn in df["transaction_items"]:
            for item in parse_transaction_items(txn):
                if item not in state_items:
                    continue
                if diet_filter.lower() != "both":
                    diet = food_df.loc[food_df["name_norm"] == item, "diet"]
                    if not diet.empty and diet.iloc[0].lower() != diet_filter.lower():
                        continue
                total_counts[item] = total_counts.get(item, 0) + 1
                per_item_restaurant.setdefault(item, {})
                per_item_restaurant[item][rest_key] = per_item_restaurant[item].get(rest_key, 0) + 1

    top_items = sorted(total_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    rows = []
    best_restaurant_for_item: Dict[str, str] = {}
    for rank, (item, total) in enumerate(top_items):
        meta = food_df[food_df["name_norm"] == item]
        diet = meta["diet"].iloc[0] if not meta.empty else "Unknown"
        rest_counts = per_item_restaurant.get(item, {})
        best_rest = max(rest_counts.items(), key=lambda kv: kv[1])[0] if rest_counts else "unknown"
        best_restaurant_for_item[item] = best_rest
        rows.append(
            {
                "Dish": f"{'🔸 ' if rank < 5 else ''}{item.title()}".strip(),
                "Diet": diet,
                "Top Restaurant": best_rest.replace("_", " ").title(),
                "Total Orders": total,
            }
        )

    df_top = pd.DataFrame(rows)
    if not df_top.empty:
        df_top = df_top.dropna(how="all").reset_index(drop=True)
    return df_top, best_restaurant_for_item


def styled_dataframe(
    df: pd.DataFrame,
    caption: str,
    column_config: Dict | None = None,
    height: int | None = None,
):
    if df.empty:
        st.info("No rows to display.")
        return
    display_df = df.copy()
    st.caption(caption)
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        height=height,
        column_config=column_config or {},
    )


def _load_font(preferred: Sequence[str], size: int) -> ImageFont.ImageFont:
    for font_name in preferred:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()


def _font_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Hg")
    return bbox[3] - bbox[1]

def build_table_snapshot(
    title: str,
    metadata: Sequence[Tuple[str, str]],
    df: pd.DataFrame,
) -> bytes:
    df_str = df.fillna("").astype(str)
    rows: List[List[str]] = [df_str.columns.tolist()] + df_str.values.tolist()
    meta_lines = [f"{label}: {value}" for label, value in metadata if value]

    title_font = _load_font(["PlayfairDisplay-Regular.ttf", "arial.ttf"], 70)
    body_font = _load_font(["arial.ttf"], 36)
    table_font = _load_font(["arial.ttf", "Consolas.ttf"], 28)

    padding_x = 25
    padding_y = 15
    border = 2

    # Determine column widths based on text measurements
    col_count = len(rows[0])
    col_widths: List[int] = [0] * col_count
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    for col_idx in range(col_count):
        max_width = 0
        for row in rows:
            text = str(row[col_idx])
            bbox = table_font.getbbox(text) if text else table_font.getbbox(" ")
            width = bbox[2] - bbox[0]
            max_width = max(max_width, width)
        col_widths[col_idx] = max_width + padding_x * 2

    row_height = (_font_height(table_font)) + padding_y * 2
    table_width = sum(col_widths) + border * (col_count + 1)
    table_height = len(rows) * row_height + border * (len(rows) + 1)

    margin = 80
    meta_margin_left = margin + 30  # More margin for left-aligned metadata
    logo_width = 0
    logo_height = 0
    logo_img = None
    if os.path.exists("logo_full.png"):
        logo_img = Image.open("logo_full.png").convert("RGBA")
        orig_w, orig_h = logo_img.size

        # Constrain the logo box to the requested bounds while maintaining aspect ratio.
        # The user wants a box around width:1024, height:1536 and "a little bit large" -> apply a slight upscale.
        target_box_w = 1024
        target_box_h = 1536
        # Compute scale to fit inside the target box while preserving aspect ratio
        scale = min(target_box_w / orig_w, target_box_h / orig_h)
        # Make it a little larger (e.g., 15% bigger) but still constrained by the target box
        upscale_factor = 0.15
        scale *= upscale_factor
        # Ensure we don't exceed the target box after upscale
        scale = min(scale, target_box_w / orig_w, target_box_h / orig_h)

        logo_width = max(1, int(orig_w * scale))
        logo_height = max(1, int(orig_h * scale))
        # Resize with high-quality resampling
        logo_img = logo_img.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

    width = max(1600, table_width + margin * 2)
    base_height = (
        margin
        + max(_font_height(title_font), logo_height)
        + 25
        + _font_height(body_font)
        + 20
        + len(meta_lines) * (_font_height(body_font) + 10)
        + 40
        + table_height
        + margin
    )
    height = max(base_height, 1000)

    bg = (10, 77, 104)
    panel = (4, 33, 50)
    accent = (247, 167, 108)
    header_fill = (14, 64, 92)
    row_fill_even = (6, 27, 44)
    row_fill_odd = (8, 36, 56)

    image = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [margin // 2, margin // 2, width - margin // 2, height - margin // 2],
        fill=panel,
        outline=accent,
        width=4,
    )

    # Place logo at top right, preserving aspect ratio and transparency
    cursor_y = margin
    if logo_img:
        # Ensure the logo does not overflow the canvas
        logo_x = max(margin, width - margin - logo_width)
        image.paste(logo_img, (logo_x, cursor_y), logo_img)
    # Center the title
    title_bbox = title_font.getbbox(title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, cursor_y), title, fill=accent, font=title_font)
    cursor_y += _font_height(title_font) + 15

    # Center subtitle
    subtitle = "CraveMap Snapshot"
    subtitle_bbox = body_font.getbbox(subtitle)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (width - subtitle_width) // 2
    draw.text((subtitle_x, cursor_y), subtitle, fill=(230, 230, 230), font=body_font)
    cursor_y += _font_height(body_font) + 15

    # Left-align metadata with more margin
    meta_y = cursor_y
    for line in meta_lines:
        draw.text((meta_margin_left, meta_y), line, fill=(255, 255, 255), font=body_font)
        meta_y += _font_height(body_font) + 10

    cursor_y = meta_y + 20
    # Center the table
    table_top = cursor_y
    table_x = (width - table_width) // 2
    draw.rectangle(
        [table_x, table_top, table_x + table_width, table_top + table_height],
        outline=accent,
        width=border,
    )

    y_cursor = table_top + border
    for row_idx, row in enumerate(rows):
        x_cursor = table_x + border
        is_header = row_idx == 0
        row_fill = header_fill if is_header else (row_fill_even if row_idx % 2 == 0 else row_fill_odd)
        stroke_color = accent if is_header else (255, 255, 255)

        for col_idx, value in enumerate(row):
            cell_width = col_widths[col_idx]
            cell_height = row_height
            draw.rectangle(
                [x_cursor, y_cursor, x_cursor + cell_width, y_cursor + cell_height],
                fill=row_fill,
                outline=stroke_color,
                width=border,
            )

            text = str(value)
            bbox = table_font.getbbox(text) if text else table_font.getbbox(" ")
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x_cursor + (cell_width - text_width) / 2
            text_y = y_cursor + (cell_height - text_height) / 2
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=table_font)

            x_cursor += cell_width + border
        y_cursor += row_height + border

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

from urllib.parse import quote
def df_to_ascii_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to a simple clean ASCII table."""
    import re

    # Remove non-ASCII chars
    def clean(text):
        return re.sub(r"[^\x00-\x7F]+", "", str(text))

    headers = [clean(h) for h in df.columns]
    rows = [[clean(c) for c in row] for row in df.to_numpy()]

    # Determine column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build divider
    divider = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Build header
    header_row = "|" + "|".join(
        f" {headers[i].ljust(col_widths[i])} " for i in range(len(headers))
    ) + "|"

    # Build body
    body_rows = []
    for row in rows:
        body_rows.append(
            "|" + "|".join(
                f" {row[i].ljust(col_widths[i])} " for i in range(len(row))
            ) + "|"
        )

    return "\n".join([divider, header_row, divider] + body_rows + [divider])
from urllib.parse import quote

def share_to_social_media(
    snapshot_title: str,
    metadata: Sequence[Tuple[str, str]],
    df: pd.DataFrame,
    filename_slug: str,
    key: str,
) -> None:
    if df.empty:
        return

    safe_slug = filename_slug.replace(" ", "_").lower()
    ascii_table = df_to_ascii_table(df)

    text = (
        "⟫⟫ 𝐖𝐄𝐋𝐂𝐎𝐌𝐄 𝐓𝐎 ✦ 𝐂𝐑𝐀𝐕𝐄𝐌𝐀𝐏 ✦ ⟪⟪\n\n"
        "-> Planning a feast or a family dinner?\n"
        "-> Explore crowd-favourite Indian dishes and their perfect pairings!\n"
        "-> CraveMap - Food Recommender:\n"
        f"{ascii_table}\n"
        "-> Try the full app here:\n"
        "https://foodrecommender1.streamlit.app/\n"
        " Share CraveMap and spread the flavours of India "
    )

    encoded = quote(text)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<a href="https://wa.me/?text={encoded}" target="_blank" style="text-decoration:none;">'
            f'<button style="background-color:#25D366;color:white;border:none;padding:10px 20px;'
            f'border-radius:6px;cursor:pointer;font-size:16px;">WhatsApp</button></a>',
            unsafe_allow_html=True
        )

    with col2:
        image_bytes = build_table_snapshot(snapshot_title, metadata, df)
        st.download_button(
            "Download Image",
            data=image_bytes,
            file_name=f"{safe_slug}.png",
            mime="image/png"
        )

    with col3:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name=f"{safe_slug}.csv",
            mime="text/csv"
        )


def explanation_block(title: str, steps: List[str]) -> None:
    with st.expander(title):
        for idx, step in enumerate(steps, start=1):
            st.markdown(f"**Step {idx}.** {step}")


def render_hero() -> None:
    # Render a left title image and a right logo image without adding extra widgets
    # that would change the subsequent layout of the tabs.
    # Use a narrower right column so images occupy less area.
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([9, 1])

    # Left: title graphic (fallback to text if image missing)
    with col_left:
        if os.path.exists("title.png"):
            # reduce displayed size
            st.image("title.png", width=360)
        else:
            st.markdown(
                '<div class="hero-title">CraveMap - Food Recommender</div>',
                unsafe_allow_html=True,
            )

    # Right: compact logo aligned to the right (fallback to nothing if missing)
    with col_right:
        if os.path.exists("logo.png"):
            # keep logo very compact so it doesn't push layout
            st.image("logo.png", width=120)


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Playground Dials")

min_support = st.sidebar.slider(
    "Magic mix frequency (support)",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f",
    help=(
        "Lower values surface hidden pairings; higher values keep only the crowd favourites."
    ),
)

st.sidebar.metric("Loaded restaurants", len(restaurants))

# Reduced vertical margin (smaller gap than before)
st.sidebar.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)

with st.sidebar.expander("👨‍🍳 Behind the Tadka (i)", expanded=False):
    st.markdown(
        """
        <div style='line-height: 1.50; text-align: justify; font-size: 0.88rem; margin-top: -4px;'>
        We curate orders from every restaurant, let the <strong>FP-Growth</strong> treasure hunt uncover dishes
        that often travel together, and then spotlight the <em>hottest food pairings</em> so you can
        satisfy cravings without digging through raw data.  
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin: 0.45rem 0;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; margin-top:-6px;">
            <p style="font-size: 0.92rem; margin-bottom: 0.30rem;">
                Made with <span style="color:#ff6f00;">❤️</span> by
            </p>
            <p style="font-size: 1.08rem; font-weight:700; margin-top: 0;">
                Yash Jangid
            </p>
            <p style="font-size: 0.83rem; margin-top: -0.35rem;">
                <a href="https://github.com/jangidyash59" target="_blank">GitHub</a> •
                <a href="https://www.linkedin.com/in/yash-jangid-7131162a0/" target="_blank">LinkedIn</a> •
                <a href="mailto:jangidworld59@gmail.com">Email</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Reset session state if user changes key controls
st.session_state.setdefault("food_result", {})
st.session_state.setdefault("food_pairs", {})
st.session_state.setdefault("state_pairs", {})

# ---------------------------------------------------------------------
# Hero + tabs
# ---------------------------------------------------------------------
render_hero()
tab_food, tab_state = st.tabs(["🍽️ Food Recommendation", "🌟 Discover Top Picks"])


# ---------------------------------------------------------------------
# Tab 1 - Food recommendation
# ---------------------------------------------------------------------
with tab_food:
    st.subheader("Find the best restaurant for your craving")

    dishes = sorted(food_df["name_norm"].dropna().unique())
    dish_labels = [""] + [dish.title() for dish in dishes]
    selected_dish_label = st.selectbox(
        "Dish name",
        options=dish_labels,
        index=0,
        help="Start typing to search quickly.",
    )
    diet_filter_food = st.selectbox(
        "Diet preference",
        ["Both", "Vegetarian", "Non Vegetarian"],
        help="Filters only kick in for the final recommendation table.",
    )

    normalized_dish = selected_dish_label.lower()
    normalized_dish = normalized_dish.strip()

    auto_state, _ = auto_select_state_for_dish(normalized_dish)
    if normalized_dish and auto_state:
        st.info(
            f"Auto-suggested state: **{display_map[auto_state]}** "
            "because it records the highest order volume for this dish."
        )

    state_index = (
        available_state_prefixes.index(auto_state)
        if auto_state in available_state_prefixes
        else 0
    )
    chosen_state_prefix = st.selectbox(
        "State focus",
        available_state_prefixes,
        index=state_index if available_state_prefixes else 0,
        format_func=lambda key: display_map.get(key, key),
    )

    if st.button("Show top restaurants", type="primary", disabled=not normalized_dish):
        if not normalized_dish:
            st.warning("Please select a dish first.")
        else:
            summary = summarize_restaurant_orders(chosen_state_prefix, normalized_dish)
            st.session_state.food_result = {
                "dish": normalized_dish,
                "state": chosen_state_prefix,
                "state_label": display_map.get(chosen_state_prefix, chosen_state_prefix),
                "rows": summary,
            }
            st.session_state.food_pairs = {}

    if st.session_state.food_result.get("rows"):
        rows = st.session_state.food_result["rows"]
        df_summary = pd.DataFrame(
            [
                {
                    "Restaurant": f"{row['Restaurant']} {'🏆' if idx < 2 else ''}".strip(),
                    "Orders": row["Orders"],
                }
                for idx, row in enumerate(rows)
            ]
        )
        styled_dataframe(
            df_summary,
            caption="Click column headers to sort. Trophy marks the top 2 performers.",
            column_config={
                "Orders": st.column_config.NumberColumn("Orders", format="%d"),
            },
            height=min(400, 80 + 32 * len(df_summary)),
        )
        share_to_social_media(
            "CraveMap - Food Recommender",
            [
                ("Dish name", st.session_state.food_result.get("dish", "").title()),
                ("Diet preference", diet_filter_food),
                ("State focus", st.session_state.food_result.get("state_label", "")),
            ],
            df_summary,
            filename_slug="best_restaurant_for_my_craving",
            key="food_leaderboard_snapshot",
        )

        eligible_restaurants = [row["key"] for row in rows if row["Orders"] > 0][:5]
        if eligible_restaurants:
            rest_choice = st.selectbox(
                "Dive deeper into a restaurant",
                eligible_restaurants,
                format_func=lambda key: key.replace("_", " ").title(),
            )
            if st.button("Generate pairings", key="food_pairs_btn"):
                table, error = build_pairing_table(
                    rest_key=rest_choice,
                    focus_item=st.session_state.food_result["dish"],
                    diet_filter=diet_filter_food,
                    state_prefix=st.session_state.food_result["state"],
                    support=min_support,
                )
                st.session_state.food_pairs = {
                    "table": table,
                    "error": error,
                    "meta": {
                        "restaurant": rest_choice.replace("_", " ").title(),
                        "dish": st.session_state.food_result["dish"].title(),
                        "diet": diet_filter_food,
                        "state": st.session_state.food_result.get("state_label", ""),
                    },
                }

        if st.session_state.food_pairs.get("error"):
            st.warning(st.session_state.food_pairs["error"])
        elif isinstance(st.session_state.food_pairs.get("table"), pd.DataFrame):
            table = st.session_state.food_pairs["table"]
            styled_dataframe(
                table,
                caption="🔹 = top-3 item inside the restaurant, 🔸 = top-5 across the state.",
                column_config={
                    "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                },
                height=min(500, 80 + 36 * len(table)),
            )
            meta = st.session_state.food_pairs.get("meta", {})
            share_to_social_media(
                "CraveMap - Food Recommender",
                [
                    ("Restaurant", meta.get("restaurant", "")),
                    ("Dish name", meta.get("dish", "")),
                    ("Diet preference", meta.get("diet", "")),
                    ("State of Origin", meta.get("state", "")),
                ],
                table,
                filename_slug="best_combos_for_my_craving",
                key="food_pairings_snapshot",
            )

    explanation_block(
        "Explain this page:",
        [
            "We standardize the dish and state names so every comparison is apples-to-apples.",
            "For a chosen dish we count its orders across every state to auto-suggest where it sells best.",
            "We show a sortable ranking of restaurants by the actual number of orders for that dish.",
            "FP-Growth runs on the selected restaurant’s basket data to expose dishes that co-occur with the focus dish.",
            "Visual markers describe whether a suggested dish is locally or state-wise popular.",
        ],
    )


# ---------------------------------------------------------------------
# Tab 2 - Discover Top Picks
# ---------------------------------------------------------------------
with tab_state:
    st.subheader("📍 Food Map of India")

    diet_filter_state = st.selectbox(
        "Diet preference (state view)",
        ["Both", "Vegetarian", "Non Vegetarian"],
    )
    chosen_state_for_top = st.selectbox(
        "Which state's heritage dishes?",
        available_state_prefixes,
        format_func=lambda key: display_map.get(key, key),
        key="state_selector",
    )

    df_top, best_rest_map = aggregate_state_top_picks(chosen_state_for_top, diet_filter_state)
    if df_top.empty:
        st.warning("No state-origin dishes found in the selected files. Please try another state.")
    else:
        
        df_top_display = df_top.copy()
        if diet_filter_state.lower() != "both" and "Diet" in df_top_display.columns:
            df_top_display = df_top_display.drop(columns=["Diet"])
        styled_dataframe(
            df_top_display,
            caption="🔸 marks the overall state favourites. Click a column header to sort.",
            column_config={
                "Total Orders": st.column_config.NumberColumn("Total Orders", format="%d"),
            },
            height=min(420, 80 + 34 * len(df_top_display)),
        )
        share_to_social_media(
            "CraveMap - Food Recommender",
            [
                ("State focus", display_map.get(chosen_state_for_top, chosen_state_for_top)),
                ("Diet preference", diet_filter_state),
                
            ],
            df_top_display,
            filename_slug="food_map_of_india",
            key="state_map_snapshot",
        )

        state_restaurants = restaurants_by_state(chosen_state_for_top)
        if not state_restaurants:
            st.warning("No restaurant files available for this state.")
        else:
            st.markdown("---")
            st.subheader("Get Recommendations from a Top Pick")
            selected_top_pick = st.selectbox(
                "Pick a dish to derive recommendations",
                df_top_display["Dish"].tolist(),
            )
            raw_item = selected_top_pick.replace("🔸", "").strip().lower()
            
            diet_filter_pick = st.selectbox(
                "Diet preference",
                ["Both", "Vegetarian", "Non Vegetarian"],
                key="diet_filter_pick",
            )
            
            default_rest_key = best_rest_map.get(raw_item, state_restaurants[0])
            default_index = (
                state_restaurants.index(default_rest_key)
                if default_rest_key in state_restaurants
                else 0
            )
            rest_choice_state = st.selectbox(
                "Restaurant to analyse",
                state_restaurants,
                index=default_index,
                format_func=lambda key: key.replace("_", " ").title(),
                key="state_rest_selector",
            )

            if st.button("Generate state pairings", key="state_pairs_btn"):
                table, error = build_pairing_table(
                    rest_key=rest_choice_state,
                    focus_item=raw_item,
                    diet_filter=diet_filter_pick,
                    state_prefix=chosen_state_for_top,
                    support=min_support,
                )
                st.session_state.state_pairs = {
                    "table": table,
                    "error": error,
                    "meta": {
                        "restaurant": rest_choice_state.replace("_", " ").title(),
                        "diet": diet_filter_pick,
                        "state": display_map.get(chosen_state_for_top, chosen_state_for_top),
                    },
                }

            if st.session_state.state_pairs.get("error"):
                st.warning(st.session_state.state_pairs["error"])
            elif isinstance(st.session_state.state_pairs.get("table"), pd.DataFrame):
                table = st.session_state.state_pairs["table"]
                styled_dataframe(
                    table,
                    caption="Recommendations derived from FP-Growth on the selected restaurant.",
                    column_config={
                        "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                    },
                    height=min(500, 80 + 36 * len(table)),
                )
                meta = st.session_state.state_pairs.get("meta", {})
                share_to_social_media(
                    "CraveMap - Food Recommender",
                    [
                        ("Restaurant", meta.get("restaurant", "")),
                        ("State of Origin", meta.get("state", "")),
                        ("Diet preference", meta.get("diet", "")),
                        
                    ],
                    table,
                    filename_slug="state_pairings_snapshot",
                    key="state_pairings_snapshot",
                )

    explanation_block(
        "Explain this page:",
        [
            "Filter the master catalogue to dishes originally belonging to the chosen state.",
            "Aggregate their order counts across every restaurant from that state.",
            "Mark the best restaurant for each dish so tastings can be prioritised.",
            "Let FP-Growth reuse the same pipeline as Page 1 to surface meaningful pairings.",
            "All tables are searchable, sortable, zebra-striped, and have hover cues for readability.",
        ],
    )
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 8px 0;
    font-size: 13px;
    color: #d9e6e6;
    background: linear-gradient(90deg, #050505 0%, #0b1116 50%, #050505 100%);
    border-top: 1px solid rgba(255,255,255,0.04);
    box-shadow: 0 -1px 10px rgba(0,0,0,0.65);
    backdrop-filter: blur(6px) saturate(1.05);
    z-index: 9999;
}
.footer a {
    color: #f7a76c;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


footer_html = f"""
<div class="footer">
    <img src="data:image/png;base64,{title_base64}" alt="CraveMap" 
         style="height: 20px; vertical-align: middle; margin-right: 1px;">
    is charted with flavour, seasoned with data, and served with ❤️ by 
    <a href="https://www.linkedin.com/in/yash-jangid-7131162a0/" target="_blank">Yash Jangid</a> · 
    <a href="mailto:jangidworld59@gmail.com">jangidworld59@gmail.com</a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)




