"""
Flask server for CubeSat Analysis frontend.
Runs the same analysis as cubesat_analysis.py (by importing it) and exposes
results as JSON for the map UI. Does not modify cubesat_analysis.py.
Run from project root: python cubesat_server.py
"""

import glob
import os

# Ensure we run from project root so cubesat_analysis and data/ are found
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, send_from_directory

import cubesat_analysis

app = Flask(__name__, static_folder="static", static_url_path="")


def _extract_prediction(result: dict) -> dict:
    """Extract predicted lat, lon, altitude_km, datetime from analysis result."""
    calls = result.get("function_calls", [])
    # Support both "arguments" and "args" (some runtimes use different keys)
    by_name = {}
    for c in calls:
        name = c.get("name")
        if not name:
            continue
        payload = c.get("arguments") or c.get("args") or {}
        by_name[name] = payload

    geo = by_name.get("predict_geolocation", {})
    alt = by_name.get("predict_altitude", {})
    dt = by_name.get("predict_datetime", {})
    lat, lon = geo.get("latitude"), geo.get("longitude")
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon = None
    alt_km = alt.get("altitude_km")
    try:
        alt_km = float(alt_km) if alt_km is not None else None
    except (TypeError, ValueError):
        alt_km = None
    dt_val = dt.get("estimated_datetime") if dt else None
    return {
        "lat": lat,
        "lon": lon,
        "altitude_km": alt_km,
        "datetime": dt_val,
    }


def _telemetry_to_actual(telemetry: dict | None) -> dict:
    """Convert load_telemetry() output to actual lat, lon, altitude_km, datetime."""
    if not telemetry:
        return {"lat": None, "lon": None, "altitude_km": None, "datetime": None}
    lat, lon = telemetry.get("latitude"), telemetry.get("longitude")
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon = None
    alt = telemetry.get("altitude")
    try:
        alt = float(alt) if alt is not None else None
    except (TypeError, ValueError):
        alt = None
    return {
        "lat": lat,
        "lon": lon,
        "altitude_km": alt,
        "datetime": telemetry.get("datetime"),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _print_cloud_prediction(filename: str, cloud_pred: dict) -> None:
    """Print cloud (Gemini) prediction to console when cloud was run."""
    if not cloud_pred:
        return
    lat = cloud_pred.get("lat")
    lon = cloud_pred.get("lon")
    alt = cloud_pred.get("altitude_km")
    dt = cloud_pred.get("datetime")
    lat_s = f"{lat}°" if lat is not None else "—"
    lon_s = f"{lon}°" if lon is not None else "—"
    alt_s = f"{alt} km" if alt is not None else "—"
    print(f"[Cloud prediction] {filename}")
    print(f"  Lat: {lat_s}  Lon: {lon_s}  Alt: {alt_s}  Datetime: {dt or '—'}")


@app.route("/api/run-analysis", methods=["POST"])
def run_analysis():
    """Run cubesat_analysis on all data/cubesat-data*.json and return local, cloud, and actual per file."""
    json_files = sorted(glob.glob("data/cubesat-data*.json"))
    if not json_files:
        return jsonify({"error": "No cubesat-data*.json found in data/", "results": []}), 200

    results = []
    for json_path in json_files:
        filename = os.path.basename(json_path)
        try:
            images = cubesat_analysis.load_images_only(json_path)
            result = cubesat_analysis.analyse_cubesat_hybrid(images)
            ground_truth = cubesat_analysis.load_telemetry(json_path)
        except Exception as e:
            results.append({
                "filename": filename,
                "error": str(e),
                "local_predicted": None,
                "cloud_predicted": None,
                "actual": None,
            })
            continue

        # result may be the local dict (when we kept local) or the cloud dict (when we deferred)
        # local_result is always the FunctionGemma result; cloud_result is set only when we ran Gemini
        local_result = result.get("local_result")
        if local_result is None:
            local_result = result
        cloud_result = result.get("cloud_result")  # same as result when we deferred; None otherwise
        local_predicted = _extract_prediction(local_result)
        cloud_predicted = _extract_prediction(cloud_result) if cloud_result else None
        actual = _telemetry_to_actual(ground_truth)

        # Log to server console for debugging
        n_calls = len(local_result.get("function_calls", []))
        cloud_str = f"({cloud_predicted.get('lat')}, {cloud_predicted.get('lon')})" if cloud_predicted else None
        print(f"[Server] {filename}: local calls={n_calls}  local_predicted=({local_predicted.get('lat')}, {local_predicted.get('lon')})  cloud_predicted={cloud_str}")

        if cloud_predicted is not None:
            _print_cloud_prediction(filename, cloud_predicted)

        results.append({
            "filename": filename,
            "local_predicted": local_predicted,
            "cloud_predicted": cloud_predicted,
            "actual": actual,
        })
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
