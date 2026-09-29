"""Generative Artistic Poster - Streamlit app

Run with:
    pip install streamlit matplotlib numpy
    streamlit run app.py
"""

import io
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

SHAPES = ["Circle", "Heart", "Rectangle", "Star", "Triangle", "Shield", "Gun", "Trump"]
TITLES = ["ART WEEK", "CREATIVE", "MODERNISM", "EXHIBIT", "GALLERY", "GEOMETRIC"]
FONTS = ["sans-serif", "serif", "monospace"]
NUM_POINTS = 120


# ----------------------------------------------------------------------------
# Core generation logic (ported from the notebook)
# ----------------------------------------------------------------------------
def generate_random_palette(rng, num_colors, blend_with_white=0.3):
    """Random vibrant RGB tuples, optionally blended toward white."""
    palette = []
    for _ in range(num_colors):
        r, g, b = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)

        # Avoid muddy dark colors
        max_val = max(r, g, b)
        if max_val < 0.5:
            r, g, b = r / max_val * 0.7, g / max_val * 0.7, b / max_val * 0.7

        palette.append(
            (
                r * (1 - blend_with_white) + blend_with_white,
                g * (1 - blend_with_white) + blend_with_white,
                b * (1 - blend_with_white) + blend_with_white,
            )
        )
    return palette


def generate_blob_coords(radius, wobble_factor, cx, cy, noise_x, noise_y, shape_type="Circle"):
    """(x, y) coordinates of a blob: base shape + wobble.

    noise_x / noise_y are pre-generated values in [-1, 1] so the wobble stays
    stable between Streamlit reruns (e.g. when only the shape changes).
    """
    theta = np.linspace(0, 2 * np.pi, NUM_POINTS, endpoint=False)

    if shape_type == "Heart":
        x_base = 16 * np.sin(theta) ** 3
        y_base = (
            13 * np.cos(theta)
            - 5 * np.cos(2 * theta)
            - 2 * np.cos(3 * theta)
            - np.cos(4 * theta)
        )
        norm = np.sqrt(x_base**2 + y_base**2)
        max_norm = np.max(norm) if np.max(norm) > 0 else 1.0
        x_base = (x_base / max_norm) * radius
        y_base = (y_base / max_norm) * radius + (0.1 * radius)

    elif shape_type == "Rectangle":
        r_base = radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
        x_base = r_base * np.cos(theta)
        y_base = r_base * np.sin(theta)

    elif shape_type == "Star":
        star_pattern = 1.0 + 0.5 * np.sin(5 * theta)
        x_base = radius * star_pattern * np.cos(theta)
        y_base = radius * star_pattern * np.sin(theta)

    elif shape_type == "Triangle":
        with np.errstate(divide="ignore", invalid="ignore"):
            r_base = radius / np.cos(
                theta - (2 * np.pi / 3) * np.floor((3 * theta + np.pi) / (2 * np.pi))
            )
        r_base = np.clip(np.nan_to_num(r_base, nan=radius, posinf=radius * 2.0), 0, radius * 2.0)
        x_base = r_base * np.cos(theta)
        y_base = r_base * np.sin(theta)

    elif shape_type == "Shield":
        r_base = radius * (1.0 - 0.3 * np.abs(np.sin(theta)) * (theta > np.pi))
        x_base = r_base * np.cos(theta)
        y_base = r_base * np.sin(theta)

    elif shape_type == "Gun":
        raw_x, raw_y = np.cos(theta), np.sin(theta)
        for i, t in enumerate(theta):
            if 0 <= t < np.pi / 2:
                raw_x[i] *= 1.8  # barrel length
                raw_y[i] *= 0.6  # barrel height
            elif np.pi / 2 <= t < np.pi:
                raw_x[i] *= 0.8
                raw_y[i] *= 0.8
            elif np.pi <= t < 1.5 * np.pi:
                raw_x[i] *= 0.7  # grip width
                raw_y[i] *= 1.6  # grip length
        norm = np.sqrt(raw_x**2 + raw_y**2)
        max_norm = np.max(norm) if np.max(norm) > 0 else 1.0
        x_base = (raw_x / max_norm) * radius
        y_base = (raw_y / max_norm) * radius

    elif shape_type == "Trump":
        raw_x, raw_y = np.cos(theta), np.sin(theta)
        for i, t in enumerate(theta):
            if 0 <= t < np.pi / 3:
                raw_x[i] *= 1.7
                raw_y[i] *= 1.4
            elif np.pi / 3 <= t < np.pi / 2:
                raw_x[i] *= 1.3
            elif 1.5 * np.pi <= t < 2 * np.pi:
                raw_x[i] *= 1.2
        norm = np.sqrt(raw_x**2 + raw_y**2)
        max_norm = np.max(norm) if np.max(norm) > 0 else 1.0
        x_base = (raw_x / max_norm) * radius
        y_base = (raw_y / max_norm) * radius

    else:  # Circle
        x_base = radius * np.cos(theta)
        y_base = radius * np.sin(theta)

    x = x_base + noise_x * wobble_factor * radius + cx
    y = y_base + noise_y * wobble_factor * radius + cy
    return x, y


def build_poster_data(seed, n_layers, max_wobble, max_radius, palette_size, blend_with_white, base_alpha):
    """Palette, background color and blob parameters, all derived from `seed`."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    palette = generate_random_palette(rng, palette_size, blend_with_white)

    # Background must be clearly distinct from every blob color.
    bg_color = None
    for _ in range(1000):
        cand = generate_random_palette(rng, 1, blend_with_white=0.1)[0]
        if min(np.linalg.norm(np.array(cand) - np.array(c)) for c in palette) > 0.35:
            bg_color = cand
            break
    if bg_color is None:  # fallback so a big palette can never hang the app
        bg_color = (0.96, 0.96, 0.96)

    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    title_color = "#FFFFFF" if luminance < 0.5 else "#0A0A0A"

    blobs = []
    for _ in range(n_layers):
        blobs.append(
            {
                "radius": rng.uniform(max_radius * 0.4, max_radius),
                "wobble": rng.uniform(0.05, max(0.05, max_wobble)),
                "cx": rng.uniform(0.1, 0.9),
                "cy": rng.uniform(0.1, 0.9),
                "color": rng.choice(palette),
                "alpha": rng.uniform(base_alpha * 0.8, min(1.0, base_alpha * 1.2)),
                "noise_x": np_rng.uniform(-1, 1, NUM_POINTS),
                "noise_y": np_rng.uniform(-1, 1, NUM_POINTS),
            }
        )
    return bg_color, title_color, blobs


def draw_poster(bg_color, title_color, blobs, shape_type, title_text, title_fontsize, title_font):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for i, blob in enumerate(blobs):
        x, y = generate_blob_coords(
            blob["radius"], blob["wobble"], blob["cx"], blob["cy"],
            blob["noise_x"], blob["noise_y"], shape_type,
        )
        z = (i + 1) * 3

        # drop shadow
        ax.fill(x + 0.015, y - 0.015, facecolor="#111111", edgecolor="none",
                alpha=0.18 * blob["alpha"], zorder=z)
        # dark underlay
        ax.fill(x, y, facecolor="#222222", edgecolor="none",
                alpha=blob["alpha"], zorder=z + 1)
        # colored body, nudged for a pseudo-3D look
        ax.fill(x - 0.005, y + 0.005, facecolor=blob["color"], edgecolor=blob["color"],
                linewidth=1.0, alpha=blob["alpha"], zorder=z + 2)

    if title_text:
        ax.text(0.05, 0.95, title_text, transform=ax.transAxes, fontsize=title_fontsize,
                family=title_font, color=title_color, ha="left", va="top", zorder=1000)

    return fig


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Generative Poster", page_icon="🎨", layout="wide")
st.title("🎨 Generative Artistic Poster")

# Session state
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 2**31 - 1)
if "title_text" not in st.session_state:
    st.session_state.title_text = TITLES[0]


def on_preset_change():
    st.session_state.title_text = st.session_state.title_preset


with st.sidebar:
    st.header("Controls")

    if st.button("🎲 Regenerate", width="stretch"):
        st.session_state.seed = random.randint(0, 2**31 - 1)

    st.caption(f"Seed: {st.session_state.seed}")

    n_layers = st.slider("Layers", 3, 15, 7, 1)
    max_wobble = st.slider("Wobble", 0.0, 0.5, 0.25, 0.05)
    max_radius = st.slider("Blob size", 0.1, 0.6, 0.45, 0.05)
    palette_size = st.slider("Palette size", 2, 12, 7, 1)
    base_alpha = st.slider("Opacity", 0.1, 1.0, 0.75, 0.05)
    blend_with_white = st.slider("Pastel amount", 0.0, 0.8, 0.15, 0.05)

    st.divider()
    shape = st.selectbox("Shape", SHAPES)

    st.divider()
    st.selectbox("Title preset", TITLES, key="title_preset", on_change=on_preset_change)
    title_text = st.text_input("Title text", key="title_text")
    title_fontsize = st.slider("Font size", 12, 72, 32, 2)
    title_font = st.selectbox("Font family", FONTS)

bg_color, title_color, blobs = build_poster_data(
    st.session_state.seed, n_layers, max_wobble, max_radius,
    palette_size, blend_with_white, base_alpha,
)

fig = draw_poster(bg_color, title_color, blobs, shape, title_text, title_fontsize, title_font)

left, center, right = st.columns([1, 3, 1])
with center:
    st.pyplot(fig, width="stretch")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    st.download_button(
        "⬇️ Download PNG",
        data=buf.getvalue(),
        file_name=f"poster_{st.session_state.seed}.png",
        mime="image/png",
        width="stretch",
    )

plt.close(fig)
