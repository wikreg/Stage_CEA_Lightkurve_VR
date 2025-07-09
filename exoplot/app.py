# ===========================================================
# Imports & Setup
# ===========================================================

from flask import (Flask, render_template, request, send_file, redirect, url_for, session, jsonify)
from io import StringIO
from io import BytesIO
import lightkurve as lk
import pandas as pd
import numpy as np
import tempfile, threading, uuid
import multiprocessing

from utils.plotting import *
from utils.calculation import run_mcmc
from utils.mcmc import make_transit_model


app = Flask(__name__)


# ===========================================================
# Helpers
# ===========================================================

def get_form_value(name, cast=str, default=None):
    val = request.form.get(name, default)
    if val in (None, ""):
        return default
    try:
        return cast(val)
    except Exception:
        return default


def render_search_results(df, target):
    cols_to_show = ['mission', 'year', 'author', 'exptime', 'target_name', 'distance']
    df = df[cols_to_show].copy()
    df["Analyse"] = [
        f'<a href="/lightcurve/{i}?target={target}">Analyse</a>' for i in range(len(df))
    ]
    return df.to_html(classes="table", escape=False, index=True)


def render_mcmc_results(result, time_jd, flux_val, flux_err, model_time, model_flux, best_period):
    lc_html = fig_to_html(
        plot_lightcurve(time_jd, flux_val, flux_err, model_x=model_time, model_y=model_flux,
                        title=f"MCMC Fit @ P={best_period:.4f} d", style='scatter')
    )

    trace_html = fig_to_html(
        plot_mcmc_traces(result['sampler'], result['labels'])
    )

    corner_html = fig_to_html(
        plot_mcmc_corner(result['flat_samples'], result['labels'],
                         truths=[result['results'][l][0] for l in result['labels']])
    )

    param_html = "<table class='table'><thead><tr><th>Parameter</th><th>Median</th><th>+1σ</th><th>-1σ</th></tr></thead><tbody>"
    for label, (med, plus, minus) in result['results'].items():
        param_html += f"<tr><td>{label}</td><td>{med:.4f}</td><td>+{plus:.4f}</td><td>-{minus:.4f}</td></tr>"
    param_html += "</tbody></table>"

    return {
        "lc_html": lc_html,
        "param_html": param_html,
        "trace_html": trace_html,
        "corner_html": corner_html,
    }

# ===========================================================
# Basic Pages
# ===========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/discover")
def discover():
    return render_template("discover.html")


@app.route("/analyse")
def analyse():
    return render_template("analyse.html")


@app.route("/single")
def single():
    return render_template("layouts/single.html")


@app.route("/side")
def side():
    return render_template("layouts/side.html")


@app.route("/stacked")
def stacked():
    return render_template("layouts/stacked.html")


@app.route("/grid2x2")
def grid2x2():
    return render_template("layouts/grid2x2.html")


# ===========================================================
# Discovery / Search
# ===========================================================

@app.route("/discover", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        target  = get_form_value("target")
        mission = get_form_value("mission")
        cadence = get_form_value("cadence")
        exptime = get_form_value("exptime", float)
        radius  = get_form_value("radius", float)
        author  = get_form_value("author")
        quarter = get_form_value("quarter", int)
        month   = get_form_value("month", int)
        campaign= get_form_value("campaign", int)
        sector  = get_form_value("sector", int)
        limit   = get_form_value("limit", int)

        search_args = {
            "target": target,
            "radius": radius,
            "exptime": exptime,
            "cadence": cadence,
            "author": author,
            "quarter": quarter,
            "month": month,
            "campaign": campaign,
            "sector": sector,
            "limit": limit,
            "mission": mission or ("Kepler", "K2", "TESS"),
        }
        search_args = {k: v for k, v in search_args.items() if v is not None}

        srch = lk.search_lightcurve(**search_args)

        if len(srch) == 0:
            return render_template("discover.html", target=target, results="<p>No lightcurves found.</p>")

        df = srch.table.to_pandas()
        html_table = render_search_results(df, target)

        return render_template("discover.html", target=target, results=html_table)

    return render_template("discover.html")


# ===========================================================
# Download Lightcurve File
# ===========================================================

@app.route("/download/<int:row>")
def download(row):
    target = request.args.get("target")
    fmt = request.args.get("format", "fits")  # default to FITS

    srch = lk.search_lightcurve(target)
    if row >= len(srch):
        return "Invalid row selected", 400

    lc = srch[row].download().normalize().remove_nans()

    if fmt == "csv":
        df = lc.to_pandas()
        csv_text = df.to_csv(index=False)

        csv_bytes = BytesIO(csv_text.encode('utf-8'))
        csv_bytes.seek(0)

        return send_file(
            csv_bytes,
            as_attachment=True,
            download_name=f"{target}_lc.csv",
            mimetype="text/csv"
        )

    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
        lc.to_fits(tmp.name)
        tmp.close()
        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=f"{target}_lc.fits"
        )


# ===========================================================
# Lightcurve Analysis
# ===========================================================

@app.route("/lightcurve/<int:row>")
def lightcurve_row(row):
    target = request.args.get("target")
    harmonic = request.args.get("harmonic", default=1, type=int)

    try:
        lc, pg, folded, best_period, best_freq, best_power, epoch_time, transit_time, transit_depth = compute_lightcurve(target, row, harmonic)
    except ValueError:
        return "Invalid row selected", 400

    raw_html = fig_to_html(
        plot_lightcurve(lc.time.value, lc.flux.value, lc.flux_err.value if lc.flux_err is not None else None,
                        title=f"Raw Lightcurve: {target}", style='line')
    )

    pg_period_html = fig_to_html(
        plot_periodogram(pg.period.value, pg.power.value, xaxis_type='period', title=f"Periodogram (Period): {target}")
    )

    pg_freq_html = fig_to_html(
        plot_periodogram(pg.frequency.value, pg.power.value, xaxis_type='frequency', title=f"Periodogram (Frequency): {target}")
    )

    folded_html = generate_folded_plot(folded, target, best_period, harmonic)

    return render_template(
        "lightcurve.html",
        target=target,
        row=row,
        raw_html=raw_html,
        pg_period_html=pg_period_html,
        pg_freq_html=pg_freq_html,
        folded_html=folded_html,
        best_period=best_period,
        best_freq=best_freq,
        best_power=best_power,
        epoch_time=epoch_time,
        transit_time=transit_time,
        transit_depth=transit_depth,
    )


@app.route("/refold")
def refold():
    target = request.args.get("target")
    row = int(request.args.get("row") or 0)
    harmonic = int(request.args.get("harmonic") or 1)

    try:
        _, pg, folded, best_period, *_ = compute_lightcurve(target, row, harmonic)
    except ValueError:
        return "Invalid row", 400

    return generate_folded_plot(folded, target, best_period, harmonic)


# ===========================================================
# MCMC Fit & Async Progress
# ===========================================================

progress_store = {}
result_store = {}

def update_progress(task_id, step, total):
    progress_store[task_id] = step / total

def mcmc_worker(task_id, time_jd, flux_val, flux_err, best_period):
    result = run_mcmc(
        time_jd, flux_val, flux_err, period=best_period,
        progress_callback=lambda step, total: update_progress(task_id, step, total)
    )

    model_time = np.linspace(min(time_jd), max(time_jd), 5000)
    params = [result['results'][l][0] for l in result['labels']]
    model_flux = make_transit_model(model_time, params, best_period)

    result_store[task_id] = render_mcmc_results(
        result, time_jd, flux_val, flux_err, model_time, model_flux, best_period
    )
    progress_store[task_id] = 1.0

@app.route("/mcmc")
def mcmc():
    target = request.args.get("target")
    row = int(request.args.get("row") or 0)
    harmonic = int(request.args.get("harmonic") or 1)

    try:
        _, _, folded, best_period, *_ = compute_lightcurve(target, row, harmonic)
    except ValueError:
        return "Invalid input", 400

    time_jd = folded.time.jd
    flux_val = folded.flux.value
    flux_err = folded.flux_err.value if folded.flux_err is not None else np.full_like(flux_val, np.median(flux_val) * 0.01)

    task_id = str(uuid.uuid4())
    progress_store[task_id] = 0
    result_store[task_id] = None

    # 🧠 Start in a separate process
    p = multiprocessing.Process(
        target=mcmc_worker,
        args=(task_id, time_jd, flux_val, flux_err, best_period)
    )
    p.start()

    return jsonify({"task_id": task_id})


@app.route("/progress/<task_id>")
def progress(task_id):
    if task_id not in progress_store:
        return jsonify({"error": "Invalid task_id"}), 404
    return jsonify({"progress": progress_store[task_id]})


@app.route("/result/<task_id>")
def result(task_id):
    result = result_store.get(task_id)
    if result is None:
        return jsonify({"error": "Result not ready"}), 202
    return jsonify(result)


# ===========================================================
# Run the App with Livereload
# ===========================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # 🧠 important for Windows/macOS!

    from livereload import Server
    server = Server(app.wsgi_app)
    server.watch('static/')
    server.watch('templates/')
    server.serve(port=5000, host='127.0.0.1', debug=True)
    
