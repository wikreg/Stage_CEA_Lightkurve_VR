"""
Catalog Plotting Utilities
---------------------------
Provides plotting functions & helpers for exoplanet catalog datasets.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter
from plotly.colors import DEFAULT_PLOTLY_COLORS

from .plotting import apply_style
from .label import label_map
from .presets import *
from .models import get_model_curve

# ===========================================================
# Constants
# ===========================================================

PRESET_GROUPS = [STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS]
ALL_PRESETS = {k: v for group in PRESET_GROUPS for k, v in group.items()}

# ===========================================================
# Data Helpers
# ===========================================================

def combine_samples(samples):
    """Combine multiple (label, DataFrame) samples into one DataFrame."""
    if isinstance(samples, dict):
        samples = samples.items()
    elif samples and not isinstance(samples[0], tuple):
        samples = [(f"Sample {i+1}", df) for i, df in enumerate(samples)]
    return pd.concat([df.assign(source=str(label)) for label, df in samples], ignore_index=True)


def prepare_labels(*keys):
    """Map keys to human-readable labels."""
    return {k: label_map.get(k, k) for k in keys}


def get_error_columns(df, x, y, show_error):
    """Return error column names for x and y if they exist and show_error is True."""
    return (
        f"{x}err1" if show_error and f"{x}err1" in df else None,
        f"{y}err1" if show_error and f"{y}err1" in df else None
    )


def clean_data(df, x, y=None, color_by=None, log_x=False, log_y=False, show_error=False):
    """Filter & clean DataFrame to keep necessary numeric columns and remove NaNs/infs."""
    cols = [x]
    if y: cols.append(y)
    if color_by: cols.append(color_by)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    if err_x: cols.append(err_x)
    if err_y: cols.append(err_y)
    cols += ['pl_name', 'source']
    df = df[cols].replace([np.inf, -np.inf], np.nan)
    for col in [x, y, color_by, err_x, err_y]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    if log_x:
        df = df[df[x] > 0].copy()
    if log_y and y:
        df = df[df[y] > 0].copy()
    return df

# ===========================================================
# Trace Helpers
# ===========================================================

def add_scatter_trace(fig, group, x, y, label_x, label_y,
                      name=None, color=None, err_x=None, err_y=None,
                      colorscale=None, cmin=None, cmax=None, colorbar=None, color_by=None):
    """Add a scatter trace to a figure."""
    marker = dict(opacity=0.8)
    if color is not None:
        marker['color'] = color
    if colorscale:
        marker['colorscale'] = colorscale
    if cmin is not None:
        marker['cmin'] = cmin
    if cmax is not None:
        marker['cmax'] = cmax
    if colorbar:
        marker['colorbar'] = colorbar

    fig.add_trace(go.Scatter(
        x=group[x], y=group[y], mode='markers',
        name=name or (group['source'].iloc[0] if 'source' in group and not group.empty else 'Sample'),
        text=group['pl_name'],
        marker=marker,
        error_x=dict(array=group[err_x]) if err_x else None,
        error_y=dict(array=group[err_y]) if err_y else None,
        hovertemplate=f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}" +
                      (f"<br>{color_by} = %{{marker.color}}" if colorscale else "") + "<extra></extra>",
        showlegend=True
    ))


def add_highlight_traces(fig, df, x, y, label_x, label_y, highlight):
    """Highlight specific planets in a figure."""
    if not highlight:
        return
    for planet in highlight:
        if planet not in df['pl_name'].values:
            continue
        hp = df[df['pl_name'] == planet]
        if not hp.empty:
            fig.add_trace(go.Scatter(
                x=hp[x], y=hp[y],
                mode='markers+text',
                text=[planet]*len(hp),
                textposition='top center',
                name=planet,
                marker=dict(symbol='star', size=14, color='red', line=dict(width=1, color='black')),
                hovertemplate=f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}<extra></extra>",
                showlegend=True
            ))


def add_model_overlay_traces(fig, x, y, overlay_models):
    """Overlay theoretical model curves if x/y match allowed axes."""
    if not overlay_models:
        return
    valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
    if (x, y) not in valid_axes and (y, x) not in valid_axes:
        return

    for i, model_key in enumerate(overlay_models):
        model_df = get_model_curve(model_key)
        x_model, y_model = model_df['mass'], model_df['radius']
        if x == "pl_rade":
            x_model, y_model = y_model, x_model

        fig.add_trace(go.Scatter(
            x=x_model, y=y_model, mode='lines',
            name=model_key.replace('_', ' ').title(),
            line=dict(dash='dash', width=2, color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]),
            hoverinfo='name', showlegend=True
        ))
    fig.update_layout(hovermode='closest')

# ===========================================================
# Plot Functions
# ===========================================================

def plot_scatter(df, x, y, highlight, log_x, log_y, show_error, overlay_models):
    """Basic scatter plot."""
    labels = prepare_labels(x, y)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()

    for src, group in base_df.groupby('source'):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, err_x=err_x, err_y=err_y)

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)
    apply_style(fig, labels, x, y, log_x, log_y)

    return fig


def plot_colored(df, x, y, color_by, highlight=None, log_x=False, log_y=False, show_error=False, colorscale_list=None, overlay_models=None):
    """Scatter plot with colormap."""
    labels = prepare_labels(x, y, color_by)
    palettes = colorscale_list or ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Viridis']
    vmin, vmax = df[color_by].min(), df[color_by].max()
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()
    sources = list(base_df['source'].unique())

    for i, src in enumerate(sources):
        group = base_df[base_df['source'] == src]
        add_scatter_trace(
            fig, group, x, y, labels[x], labels[y], name=src,
            color=group[color_by], err_x=err_x, err_y=err_y,
            colorscale=palettes[i % len(palettes)],
            cmin=vmin, cmax=vmax,
            colorbar=dict(x=1.02, y=0.5, len=0.7, thickness=12, xanchor='center', yanchor='middle'),
            color_by=labels[color_by]
        )

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)
    apply_style(fig, labels, x, y, log_x, log_y)

    return fig


def plot_density(df, x, y, highlight=None, log_x=False, log_y=False, show_error=False, cmap='YlOrRd', overlay_models=None):
    """Scatter plot with density heatmap."""
    labels = prepare_labels(x, y)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()

    x_data, y_data = df[x].to_numpy(), df[y].to_numpy()
    if log_x: x_data = np.log10(x_data)
    if log_y: y_data = np.log10(y_data)

    bins = 100
    x_bins = np.linspace(x_data.min(), x_data.max(), bins)
    y_bins = np.linspace(y_data.min(), y_data.max(), bins)
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
    H = gaussian_filter(H, sigma=8)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    fig.add_trace(go.Heatmap(x=x_centers, y=y_centers, z=H.T, colorscale=cmap, opacity=0.7, name='Density'))

    for i, (src, group) in enumerate(base_df.groupby('source')):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, err_x=err_x, err_y=err_y)

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)
    apply_style(fig, labels, x, y, log_x, log_y)

    return fig


def plot_histogram(df, column, bins=50, log_x=False, log_y=False):
    """Histogram of a single column, optionally grouped by source."""
    labels = {column: label_map.get(column, column), 'count': 'Count'}
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    if log_x: df = df[df[column] > 0]

    if 'source' not in df.columns:
        df['source'] = 'Total'

    min_val, max_val = df[column].min(), df[column].max()
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins+1) if log_x else np.linspace(min_val, max_val, bins+1)

    for i, (sample, df_sample) in enumerate(df.groupby('source')):
        counts, edges = np.histogram(df_sample[column], bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)

        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths, name=sample,
            marker=dict(color=palette[i % len(palette)], line=dict(color='black', width=1))
        ))

    apply_style(fig, labels, x=column, y='count', log_x=log_x, log_y=log_y)
    fig.update_layout(barmode='relative', title=f"Histogram of {labels[column]}")
    return fig

# ===========================================================
# Main Plot Tool
# ===========================================================

def main_plot(plot_type, preset_keys=None, df_full=None, pairs=None,
              x_axis=None, y_axis=None, highlight_planets=None, overlay_models=None,
              color_by=None, log_x=False, log_y=False, show_error=False, cmap='YlOrBr', bins=None):
    """
    Entry point to generate a plot based on a preset or dataset.
    """
    if pairs:
        df_list = []
        for preset_key, dataset_name in pairs:
            df_data = ALL_DATA.get(dataset_name)
            df_filtered = ALL_PRESETS[preset_key](df_data)
            df_list.append((f"{preset_key} ({dataset_name})", df_filtered))
        df = combine_samples(df_list)
    else:
        if isinstance(df_full, str):
            df_full = ALL_DATA.get(df_full)
        if not preset_keys:
            df = df_full.copy()
            if 'source' not in df.columns:
                df['source'] = 'NEA'
        else:
            df_list = [(key, ALL_PRESETS[key](df_full)) for key in preset_keys]
            df = combine_samples(df_list)

    if plot_type == 'histogram':
        df = clean_data(df, x_axis, None, log_x=log_x, log_y=log_y)
    else:
        df = clean_data(df, x_axis, y_axis, color_by=color_by, log_x=log_x, log_y=log_y, show_error=show_error)

    if plot_type == 'histogram':
        return plot_histogram(df, column=x_axis, bins=bins, log_x=log_x, log_y=log_y)
    elif plot_type == 'colored':
        return plot_colored(df, x_axis, y_axis, color_by, highlight_planets, log_x, log_y, show_error, overlay_models=overlay_models)
    elif plot_type == 'density':
        return plot_density(df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, cmap=cmap, overlay_models=overlay_models)
    else:
        return plot_scatter(df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, overlay_models)
