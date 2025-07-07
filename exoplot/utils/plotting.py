import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lightkurve as lk

# ===========================================================
# General Helpers
# ===========================================================

def fig_to_html(fig):
    """Convert a Plotly figure to embeddable HTML."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})


def apply_style(fig, labels, x, y, log_x=False, log_y=False,
                font_family="Inter, sans-serif", font_size=14, font_color="white"):
    """Apply consistent styling to Plotly figure."""
    fig.update_layout(
        font=dict(family=font_family, size=font_size, color=font_color),
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        margin=dict(l=80, r=80, t=80, b=80), template='plotly_dark',
        legend=dict(bgcolor='rgba(68,68,68,0.5)', bordercolor='white', borderwidth=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showline=True, linecolor='white', mirror=True)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showline=True, linecolor='white', mirror=True)
    return fig


# ===========================================================
# Lightcurve Plots
# ===========================================================

def plot_lightcurve(x, y, err=None, model_x=None, model_y=None,
                    title="Light Curve", style='scatter', bins=None,
                    xlabel="Time", ylabel="Flux"):
    """
    General purpose lightcurve (or folded) plot.
    Can overlay model curve & optionally bin data.
    """
    fig = go.Figure()

    # data
    if style == 'line':
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Data'))
    elif style == 'errorbar' and err is not None:
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 error_y=dict(type='data', array=err, visible=True),
                                 marker=dict(size=2, color='orange'), name='Data'))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2, color='orange'), name='Data'))

    # binned
    if bins:
        bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        fig.add_trace(go.Scatter(x=bin_centers, y=bin_means,
                                 mode='markers+lines', marker=dict(size=4, color='cyan'),
                                 line=dict(color='cyan'), name=f'Binned (bins={bins})'))

    # model
    if model_x is not None and model_y is not None:
        fig.add_trace(go.Scatter(x=model_x, y=model_y,
                                 mode='lines', line=dict(color='cyan', width=4),
                                 name='Model'))

    labels = {'x': xlabel, 'y': ylabel}
    fig = apply_style(fig, labels, x='x', y='y')
    fig.update_layout(title=title)

    return fig


def generate_folded_plot(folded, target, best_period, harmonic):
    """
    Convenience helper: produce folded lightcurve plot + HTML.
    """
    title = f"Folded Lightcurve: {target} @ P={harmonic}×{best_period:.4f} d"
    fig = plot_lightcurve(
        folded.time.value, folded.flux.value,
        folded.flux_err.value if folded.flux_err is not None else None,
        title=title, style='scatter', xlabel="Phase [days]", ylabel="Flux"
    )
    return fig_to_html(fig)


# ===========================================================
# Periodogram Plots
# ===========================================================

def plot_periodogram(x, y, title="Periodogram",
                     xaxis_type='period', style='line'):
    """
    Plot periodogram in either period or frequency space.
    """
    if xaxis_type == 'frequency':
        xlabel, logx, logy = "Frequency [1/day]", False, False
    else:
        xlabel, logx, logy = "Period [day]", True, True

    fig = go.Figure()

    trace_style = dict(color='orange')
    mode = 'markers' if style == 'scatter' else 'lines'
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, marker=trace_style, line=trace_style, name='Periodogram'))

    labels = {'x': xlabel, 'y': "Power"}
    fig = apply_style(fig, labels, x='x', y='y', log_x=logx, log_y=logy)
    fig.update_layout(title=title)

    return fig


# ===========================================================
# MCMC Diagnostic Plots
# ===========================================================

def plot_mcmc_traces(sampler, labels):
    """
    Plot MCMC traces of each parameter.
    """
    ndim = len(labels)
    nsteps, nwalkers, _ = sampler.get_chain().shape
    chain = sampler.get_chain()

    fig = make_subplots(
        rows=ndim, cols=1, shared_xaxes=True,
        subplot_titles=labels, vertical_spacing=0.02
    )

    for i in range(ndim):
        for w in range(nwalkers):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(nsteps),
                    y=chain[:, w, i],
                    line=dict(width=0.5, color="orange"),
                    opacity=0.3,
                    showlegend=False
                ),
                row=i+1, col=1
            )
        fig.update_yaxes(title_text=labels[i], row=i+1, col=1)

    fig.update_xaxes(title_text="Step number", row=ndim, col=1)
    fig.update_layout(autosize=True, template="plotly_dark", title_text="MCMC Trace Plots")

    return fig


def plot_mcmc_corner(flat_samples, labels, truths=None):
    """
    Corner-like scatter-matrix plot of posterior samples.
    """
    df = pd.DataFrame(flat_samples, columns=labels)

    fig = px.scatter_matrix(
        df,
        dimensions=labels,
        title="MCMC Posterior Distributions",
        template="plotly_dark"
    )

    fig.update_traces(
        diagonal_visible=True,
        marker=dict(opacity=0.3, size=2, color="orange")
    )

    if truths:
        for label, tval in zip(labels, truths):
            fig.add_shape(
                type="line",
                x0=tval, x1=tval,
                y0=df[label].min(), y1=df[label].max(),
                line=dict(color="cyan", dash="dot"),
                xref=f"x{labels.index(label)+1}",
                yref=f"y{labels.index(label)+1}"
            )

    fig.update_layout(autosize=True)
    return fig


# ===========================================================
# Compute Lightcurve + Stats
# ===========================================================

def compute_lightcurve(target, row, harmonic=1):
    """
    Download & preprocess lightcurve, compute periodogram & fold.
    """
    srch = lk.search_lightcurve(target)

    if row >= len(srch):
        raise ValueError("Invalid row")

    lc = srch[row].download().normalize().remove_nans()

    pg = lc.to_periodogram(method='bls')
    best_period = round(pg.period_at_max_power.value, 4)
    best_freq = round(pg.frequency_at_max_power.value, 4)
    best_power = round(pg.max_power.value, 4)
    epoch_time = round(lc.time[np.argmin(lc.flux)].value, 4)
    transit_time = round(pg.duration_at_max_power.value, 4)
    transit_depth = round(pg.depth_at_max_power.value, 4)

    if harmonic <= 0:
        harmonic = 1

    folded = lc.fold(period=harmonic * best_period, epoch_time=epoch_time)

    return lc, pg, folded, best_period, best_freq, best_power, epoch_time, transit_time, transit_depth
