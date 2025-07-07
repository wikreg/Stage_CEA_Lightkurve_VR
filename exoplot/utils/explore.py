from IPython.display import display
import pandas as pd
pd.set_option("display.max_rows", None)

import sys
import numpy as np
import lightkurve as lk
from exoplot.utils.plotting import (plot_lightcurve, plot_periodogram)
from exoplot.utils.calculation import run_mcmc
from exoplot.utils.mcmc import make_transit_model
import astropy.units as u


def explore(target, radius=None, exptime=None, cadence=None,
            mission=('Kepler', 'K2', 'TESS'), author=None,
            quarter=None, month=None, campaign=None, sector=None,
            limit=None):
    """
    User-friendly, interactive transit analysis of a target.
    """
    print(f"\n🔍 Searching for lightcurves of '{target}'…")
    srch = lk.search_lightcurve(
        target, radius=radius, exptime=exptime, cadence=cadence,
        mission=mission, author=author,
        quarter=quarter, month=month, campaign=campaign, sector=sector,
        limit=limit
    )

    if len(srch) == 0:
        print("❌ No lightcurves found.")
        return

    df = srch.table.to_pandas()

    cols_to_show = ['mission', 'year', 'author', 'exptime', 'target_name', 'distance']
    print("\n📋 Available lightcurves:\n")
    display(df[cols_to_show])
    sys.stdout.flush()

    idx = int(input(f"\n👉 Enter the number of the row you want to download (0–{len(df)-1}): "))
    lc = srch[idx].download().normalize().remove_nans()

    # raw lightcurve
    x = lc.time.value
    y = lc.flux.value
    err = lc.flux_err.value if lc.flux_err is not None else None

    print("\n📈 Plotting lightcurve…")
    plot_lightcurve(
        x, y, err,
        title=f"Lightcurve: {target}",
        style='line',
        xlabel="Time [days]",
        ylabel="Flux"
    )

    # periodogram
    print("\n⏳ Computing periodogram…")
    pg = lc.to_periodogram(method='bls')
    max_power_idx = np.argmax(pg.power)
    best_period = pg.period[max_power_idx].to_value(u.day)
    best_freq = pg.frequency[max_power_idx].to_value(1/u.day)
    best_power = pg.power[max_power_idx]

    print(f"\n⭐ Best period: {best_period:.4f} d")
    print(f"⭐ Best frequency: {best_freq:.4f} 1/d")
    print(f"⭐ Maximum power: {best_power:.4f}")

    xaxis_type = input("\n📊 Plot periodogram x-axis as 'period' or 'frequency'? (default=period): ").strip().lower()
    if xaxis_type not in ('frequency', 'period'):
        xaxis_type = 'period'

    if xaxis_type == 'frequency':
        x_pg = pg.frequency.value
    else:
        x_pg = pg.period.value
    y_pg = pg.power.value

    print("\n📊 Plotting periodogram…")
    plot_periodogram(
        x_pg, y_pg,
        title=f"Periodogram: {target}",
        xaxis_type=xaxis_type,
        style='line'
    )

    # fold lightcurve
    print(f"\n🔄 Folding lightcurve at best period ≈ {best_period:.4f} d …")
    t0 = lc.time[np.argmin(lc.flux)].value
    folded = lc.fold(period=best_period, epoch_time=t0)

    x_fold = folded.time.value
    y_fold = folded.flux.value
    err_fold = folded.flux_err.value if folded.flux_err is not None else None

    plot_lightcurve(
        x_fold, y_fold, err_fold,
        title=f"Folded Lightcurve: {target} @ P={best_period:.4f} d",
        style='scatter',
        xlabel="Phase [days]",
        ylabel="Flux"
    )

    # harmonics?
    harmonics = input("\n🎯 Enter number of harmonics to refold (or press Enter to skip): ").strip()
    if harmonics:
        try:
            nb_harm = int(harmonics)
            if nb_harm <= 0:
                print("⚠️ Invalid choice: must be > 0.")
            else:
                folded_harm = lc.fold(period=nb_harm * best_period, epoch_time=t0)
                plot_lightcurve(
                    folded_harm.time.value, folded_harm.flux.value,
                    title=f"Refolded Lightcurve ({nb_harm}×) @ P={best_period:.4f} d",
                    style='scatter',
                    xlabel="Phase [days]",
                    ylabel="Flux"
                )
        except ValueError:
            print("⚠️ Invalid input: must be integer.")

    # Run MCMC
    print("\n🚀 Running MCMC on folded lightcurve…")

    time_jd = folded.time.jd
    flux_val = folded.flux.value
    flux_err = folded.flux_err.value if folded.flux_err is not None else np.full_like(flux_val, np.median(flux_val)*0.01)

    results = run_mcmc(time_jd, flux_val, flux_err, period=best_period)

    # Compute best-fit model
    model_time = np.linspace(min(time_jd), max(time_jd), 5000)
    params = [results[l][0] for l in results]
    model_flux = make_transit_model(model_time, params, best_period)

    print("\n📐 Plotting MCMC fit on folded lightcurve…")
    plot_lightcurve(
        time_jd, flux_val, flux_err,
        model_x=model_time, model_y=model_flux,
        title=f"MCMC Fit: {target} @ P={best_period:.4f} d",
        style='scatter',
        xlabel="Phase [days]",
        ylabel="Flux"
    )

    print("\n✅ Analysis complete.")
