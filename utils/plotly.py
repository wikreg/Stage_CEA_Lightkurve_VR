from IPython.display import display

import plotly.graph_objects as go
import lightkurve as lk
import numpy as np
from scipy import stats
import sys

import pandas as pd
pd.set_option("display.max_columns", None)

def apply_style(fig, labels, x, y, log_x=False, log_y=False, width=1500, height=500, font_family="Inter, sans-serif", font_size=14, font_color="white"):
    fig.update_layout(font=dict(family=font_family, size=font_size, color=font_color),
                      xaxis=dict(title=labels[x], type='log' if log_x else 'linear'), yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
                      margin=dict(l=80, r=80, t=80, b=80), template='plotly_dark', height=height, width=width,
                      legend=dict(bgcolor='rgba(68,68,68,0.5)', bordercolor='white', borderwidth=1))
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showline=True, linecolor='white', mirror=True, title_standoff=20)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', showline=True, linecolor='white', mirror=True, title_standoff=20)

    return fig



def plot_lc(lc, title="Light Curve", style='scatter', **kwargs):
    x, y, err = lc.time.value, lc.flux.value, lc.flux_err.value if lc.flux_err is not None else None

    fig = go.Figure()

    if style == 'line': fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Light Curve'))

    elif style == 'errorbar' and err is not None: fig.add_trace(go.Scatter(x=x, y=y, mode='markers', error_y=dict(type='data', array=err, visible=True), marker=dict(size=2, color='orange'), name='Light Curve'))

    else: fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2, color='orange'), name='Light Curve'))

    labels = {'time': f"Time [BKJD days]", 'flux': "Flux [e<sup>-</sup>.s<sup>-1</sup>]"}
    fig = apply_style(fig, labels, x='time', y='flux', log_x=False, log_y=False, width=kwargs.get("width",1500), height=kwargs.get("height",500))

    fig.update_layout(title=title)
    
    fig.show()



def plot_pgram(pgram, title="Periodogram", xaxis='period', style='line', **kwargs):
    if xaxis=='frequency': 
        x = pgram.frequency.value
        xlabel = f"Frequency [d<sup>-1</sup>]"
        xlog = False
        ylog = False
        xkey = 'frequency'
    else: 
        x = pgram.period.value
        xlabel = f"Period [d]"
        xlog = True
        ylog = True
        xkey = 'period'

    y, ylabel = pgram.power.value, f"Power [e<sup>-</sup>.s<sup>-1</sup>]]"

    fig = go.Figure()

    if style=='scatter': fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='orange'), name='Periodogram'))

    else: fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Periodogram'))

    labels = {xkey: xlabel, 'power': ylabel}
    fig = apply_style(fig, labels, x=xkey, y='power', log_x=xlog, log_y=ylog, width=kwargs.get("width",1500), height=kwargs.get("height",500))

    fig.update_layout(title=title)
    
    fig.show()



def plot_fold(lc, title="Folded Light Curve", style='scatter', bins=None, **kwargs):
    x, y, err = lc.phase.value, lc.flux.value, lc.flux_err.value if lc.flux_err is not None else None

    fig = go.Figure()

    if style == 'line': fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Folded', ))

    elif style == 'errorbar' and err is not None: fig.add_trace(go.Scatter(x=x, y=y, mode='markers', error_y=dict(type='data', array=err, visible=True), marker=dict(size=2, color='orange'), name='Folded'))

    else: fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2, color='orange'), name='Folded'))

    if bins:
        bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        fig.add_trace(go.Scatter(x=bin_centers, y=bin_means, mode='markers+lines', line=dict(color='cyan'), marker=dict(size=4, color='cyan'), name=f'Binned (bins={bins})'))

    labels = {'phase': "Phase", 'flux': "Flux [{lc.flux.unit}]"}
    fig = apply_style(fig, labels, x='phase', y='flux', log_x=False, log_y=False, width=kwargs.get("width",1500), height=kwargs.get("height",500))

    fig.update_layout(title=title)
    
    fig.show()



def explore(target, radius=None, exptime=None, cadence=None, mission=('Kepler', 'K2', 'TESS'), author=None, quarter=None, month=None, campaign=None, sector=None, limit=None, 
            normalized=False,
            plot_lc_style='line',
            plot_pgram_style='line', pgram_axis='frequency',
            plot_fold_style='scatter', bins=None
            ):
    print(f"Searching for lightcurves of '{target}'...")

    srch = lk.search_lightcurve(target, radius=radius, exptime=exptime, cadence=cadence, mission=mission, author=author, quarter=quarter, month=month, campaign=campaign, sector=sector, limit=limit)

    if len(srch) == 0:
        print("No lightcurves found.")
        return

    df = srch.table.to_pandas()
    cols_to_show = ['mission', 'year', 'author', 'exptime', 'target_name', 'distance']
    df_display = df[cols_to_show].copy()
    df_display.reset_index(inplace=True)
    df_display.rename(columns={'index': 'Choice'}, inplace=True)

    print("\nAvailable lightcurves:\n")
    display(df_display)

    sys.stdout.flush()

    idx = int(input(f"\nEnter the number of the row you want to download (0–{len(df_display)-1}): "))
    try:
        if not 0 <= idx < len(df_display):
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return

    print(f"\nDownloading {df_display.loc[idx, 'mission']}")
    lc = srch[idx].download()

    if normalized:
        lc = lc.remove_nans().normalize().flatten()
        print("Lightcurve downloaded and normalized.")

    print("\nPlotting lightcurve...")
    plot_lc(lc, title=f"Lightcurve: {target}", style=plot_lc_style)

    print("\nComputing periodogram...")
    pg = lc.to_periodogram()
    max_power_idx = pg.power.argmax()
    best_period = pg.period[max_power_idx]
    print(f"Highest power at period: {best_period:.4f}")
    plot_pgram(pg, title=f"Periodogram: {target}", xaxis=pgram_axis, style=plot_pgram_style)

    print(f"\nFolding lightcurve at period ≈ {best_period:.4f}...")
    t0 = lc.time[np.argmin(lc.flux)]
    folded = lc.fold(period=best_period, epoch_time=t0)
    plot_fold(folded, title=f"Folded Lightcurve: {target} @ P={best_period:.4f}", style=plot_fold_style, bins=bins)

    sys.stdout.flush()

    nb_harm = int(input(f"\nEnter the number of harmonics you want to refold with"))
    try:
        if nb_harm < 0:
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    folded_harm = lc.fold(period=nb_harm * best_period, epoch_time=t0)
    plot_fold(folded_harm, title=f"Refolded Lightcurve ({nb_harm} harmonics): {target} @ P={best_period:.4f}", style=plot_fold_style, bins=bins)




