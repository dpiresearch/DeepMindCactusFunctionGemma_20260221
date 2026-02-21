"""
CubeSat Imagery Analysis
========================
Reads image data from a cubesat JSON file and uses FunctionGemma (on-device via Cactus)
to predict geolocation, altitude, and datetime from the imagery context.

Hybrid routing strategy:
  - FunctionGemma is queried once per prediction (3 focused calls).
  - Confidence is evaluated per-prediction; any low-confidence result triggers a cloud fallback.
  - Gemini (Google DeepMind) cloud receives the actual multimodal images for high-quality analysis.
"""

import sys, os, json, base64, time, glob as _glob

# Locate the cactus python bindings
for _p in ["cactus/python/src", "/Users/dpang/dev/cactus/python/src"]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

FUNCTIONGEMMA_PATH = next(
    (p for p in [
        "cactus/weights/functiongemma-270m-it",
        "/Users/dpang/dev/cactus/weights/functiongemma-270m-it",
    ] if os.path.isdir(p)),
    None,
)

import re

from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ─── JSON repair helper ───────────────────────────────────────────────────────

def _try_parse_json(raw_str: str) -> dict | None:
    """
    Attempt to parse a (possibly malformed) JSON string produced by a small LLM.
    Common failure modes handled:
      • "field":,   → "field":null,
      • "field"::<1 → "field":0.5
      • bare-word non-numeric values for number fields
    Returns parsed dict or None on unrecoverable failure.
    """
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        pass

    # Fix "key":,   → "key":null,
    repaired = re.sub(r'("[\w]+")\s*:\s*,', r'\1:null,', raw_str)
    # Fix "key":<value  (XML-style) → remove the broken token
    repaired = re.sub(r'("[\w]+")\s*:\s*<[^,}\]]+', r'\1:null', repaired)
    # Fix "key":bare_word  (non-quoted string in number position)
    repaired = re.sub(r'("(?:confidence|latitude|longitude|altitude_km)"\s*:\s*)([a-zA-Z][^,}\]]*)',
                      r'\g<1>0.5', repaired)
    # Trim any trailing garbage after the last closing brace
    last_brace = repaired.rfind("}")
    if last_brace != -1:
        repaired = repaired[: last_brace + 1]
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


# ─── Tool definitions ─────────────────────────────────────────────────────────

TOOL_PREDICT_GEOLOCATION = {
    "name": "predict_geolocation",
    "description": (
        "Predict the geographic coordinates (latitude and longitude) of a CubeSat "
        "in low Earth orbit. Base predictions on visible land/ocean boundaries, "
        "coastline shapes, cloud-band patterns, and terminator position."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Predicted latitude in decimal degrees (-90 South Pole to +90 North Pole)",
            },
            "longitude": {
                "type": "number",
                "description": "Predicted longitude in decimal degrees (-180 to +180, East positive)",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence of the location prediction from 0.0 (none) to 1.0 (certain)",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of which visual features drove this estimate",
            },
        },
        "required": ["latitude", "longitude", "confidence", "reasoning"],
    },
}

TOOL_PREDICT_ALTITUDE = {
    "name": "predict_altitude",
    "description": (
        "Predict the orbital altitude in kilometres of a CubeSat above Earth's surface. "
        "Use Earth-limb curvature radius, atmospheric-haze layer thickness, and the "
        "angular size of features on the surface as cues."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "altitude_km": {
                "type": "number",
                "description": "Predicted orbital altitude in kilometres above Earth's surface",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence of the altitude prediction from 0.0 to 1.0",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of which visual cues drove this estimate",
            },
        },
        "required": ["altitude_km", "confidence", "reasoning"],
    },
}

TOOL_PREDICT_DATETIME = {
    "name": "predict_datetime",
    "description": (
        "Predict the UTC date and time when a set of CubeSat images were captured. "
        "Use solar illumination angle on Earth's surface, terminator line position, "
        "shadow directions, star-field orientation in the zenith camera, and seasonal "
        "vegetation or ice-coverage cues for the estimate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "estimated_datetime": {
                "type": "string",
                "description": "Best-estimate capture time in ISO 8601 UTC, e.g. '2026-02-22T08:40:00Z'",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence of the datetime prediction from 0.0 to 1.0",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the lighting, shadow, or seasonal cues used",
            },
        },
        "required": ["estimated_datetime", "confidence", "reasoning"],
    },
}

ALL_TOOLS = [TOOL_PREDICT_GEOLOCATION, TOOL_PREDICT_ALTITUDE, TOOL_PREDICT_DATETIME]

# ─── Simplified tool schemas for the local model ─────────────────────────────
# The full reasoning field causes FunctionGemma-270m to generate malformed JSON.
# These stripped-down versions omit reasoning and use short descriptions.

_LOCAL_TOOL_PREDICT_GEOLOCATION = {
    "name": "predict_geolocation",
    "description": "Predict satellite lat/lon from imagery. Use best estimate.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "Latitude degrees"},
            "longitude": {"type": "number", "description": "Longitude degrees"},
            "confidence": {"type": "number", "description": "Confidence 0.0 to 1.0"},
        },
        "required": ["latitude", "longitude", "confidence"],
    },
}

_LOCAL_TOOL_PREDICT_ALTITUDE = {
    "name": "predict_altitude",
    "description": "Predict satellite orbital altitude in km. Use best estimate.",
    "parameters": {
        "type": "object",
        "properties": {
            "altitude_km": {"type": "number", "description": "Altitude in km"},
            "confidence": {"type": "number", "description": "Confidence 0.0 to 1.0"},
        },
        "required": ["altitude_km", "confidence"],
    },
}

_LOCAL_TOOL_PREDICT_DATETIME = {
    "name": "predict_datetime",
    "description": "Predict UTC capture datetime. Use best estimate.",
    "parameters": {
        "type": "object",
        "properties": {
            "estimated_datetime": {"type": "string", "description": "ISO 8601 UTC datetime"},
            "confidence": {"type": "number", "description": "Confidence 0.0 to 1.0"},
        },
        "required": ["estimated_datetime", "confidence"],
    },
}

_LOCAL_TOOLS = [_LOCAL_TOOL_PREDICT_GEOLOCATION, _LOCAL_TOOL_PREDICT_ALTITUDE, _LOCAL_TOOL_PREDICT_DATETIME]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_images_only(json_path: str) -> list[dict]:
    """Return only the images list from the cubesat JSON. Telemetry is NOT loaded."""
    with open(json_path) as f:
        data = json.load(f)
    return data["images"]  # [{url: "data:image/png;base64,...", label: "..."}, ...]


def load_telemetry(json_path: str) -> dict | None:
    """
    Load ground-truth telemetry and timestamp from a cubesat JSON file.
    Returns dict with latitude, longitude, altitude, datetime (time part only, e.g. 08:40:27Z),
    or None if the file has no telemetry.
    """
    with open(json_path) as f:
        data = json.load(f)
    telemetry = data.get("telemetry")
    if not telemetry:
        return None
    ts = data.get("timestamp") or ""
    # Use time part only for display, e.g. 2026-02-22T08:40:27.229Z -> 08:40:27Z
    if "T" in ts:
        time_part = ts.split("T", 1)[1]
        datetime_str = time_part.replace(".000Z", "Z").split(".")[0]
        if not datetime_str.endswith("Z"):
            datetime_str += "Z"
    else:
        datetime_str = ts or "—"
    return {
        "latitude": telemetry.get("latitude"),
        "longitude": telemetry.get("longitude"),
        "altitude": telemetry.get("altitude"),
        "datetime": datetime_str,
    }


def decode_b64_image(data_url: str) -> bytes:
    """Strip the data-URL header and return raw image bytes."""
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


# ─── Gemini schema helper ─────────────────────────────────────────────────────

_TYPE_MAP = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN"}

def _to_gemini_decl(tool: dict) -> types.FunctionDeclaration:
    props = {
        k: types.Schema(type=_TYPE_MAP.get(v["type"].lower(), "STRING"), description=v.get("description", ""))
        for k, v in tool["parameters"]["properties"].items()
    }
    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool["description"],
        parameters=types.Schema(
            type="OBJECT",
            properties=props,
            required=tool["parameters"].get("required", []),
        ),
    )


# ─── On-device inference (Cactus / FunctionGemma) ────────────────────────────

_SCENE_BRIEF = (
    "A CubeSat in low Earth orbit has 6 cameras. "
    "Nadir (Earth-facing): open ocean, partial cloud cover, no coastlines or land visible. "
    "Zenith (space-facing): dark star field. "
    "Forward/Backward (along orbital track): Earth limb with thin atmospheric haze. "
    "Port/Starboard (lateral): curved Earth horizon."
)

_PER_TOOL_PROMPTS = {
    "predict_geolocation": (
        f"{_SCENE_BRIEF} "
        "Based solely on this description, call predict_geolocation with your best "
        "lat/lon estimate and an honest confidence reflecting the limited visual info."
    ),
    "predict_altitude": (
        f"{_SCENE_BRIEF} "
        "Based solely on this description, call predict_altitude with your best "
        "altitude_km estimate and an honest confidence reflecting the limited visual info."
    ),
    "predict_datetime": (
        f"{_SCENE_BRIEF} "
        "Based solely on this description, call predict_datetime with your best "
        "ISO 8601 UTC datetime estimate and an honest confidence reflecting limited info."
    ),
}


def _run_cactus_single(tool: dict, user_prompt: str, max_retries: int = 3) -> dict:
    """
    Run one focused Cactus inference for a single tool.
    Uses simplified (no-reasoning) tool schemas to avoid JSON malformation by the
    small model. Retries up to max_retries times; on each attempt tries both direct
    JSON parsing and a lightweight repair pass.
    """
    cactus_tool = [{"type": "function", "function": tool}]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a satellite data analyst. Call the provided function with a concrete "
                "numerical estimate. Never ask for more data."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        model = cactus_init(FUNCTIONGEMMA_PATH)
        raw_str = cactus_complete(
            model,
            messages,
            tools=cactus_tool,
            force_tools=True,
            max_tokens=128,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        cactus_destroy(model)
        parsed = _try_parse_json(raw_str)
        if parsed is not None:
            return parsed
        # All repair attempts failed; retry with a fresh model init

    return {"function_calls": [], "total_time_ms": 0, "confidence": 0.0}


def generate_cactus_all(images: list[dict]) -> dict:
    """
    Run FunctionGemma once per prediction tool (3 focused calls).
    Uses simplified tool schemas to avoid JSON malformation by the small model.
    The full (reasoning-bearing) tool schemas are used for cloud inference only.
    Returns aggregated result with per-call confidence and combined average.
    """
    results = []
    total_time_ms = 0.0

    for local_tool, full_tool in zip(_LOCAL_TOOLS, ALL_TOOLS):
        assert local_tool["name"] == full_tool["name"]
        prompt = _PER_TOOL_PROMPTS[local_tool["name"]]
        raw = _run_cactus_single(local_tool, prompt)

        # Rename back to full tool name for consistent downstream handling
        tool = full_tool

        calls = raw.get("function_calls", [])
        confidence = raw.get("confidence", 0.0)
        t = raw.get("total_time_ms", 0.0)
        total_time_ms += t

        # Keep the best matching call for this tool (if any)
        matched = next((c for c in calls if c.get("name") == tool["name"]), None)
        results.append({
            "tool": tool["name"],
            "call": matched,
            "confidence": confidence,
            "time_ms": t,
        })
        # Show both cactus outer confidence and the in-argument prediction confidence
        arg_conf = matched["arguments"].get("confidence", "?") if matched else "?"
        try:
            arg_conf_str = f"pred_conf={float(arg_conf):.2f}"
        except (TypeError, ValueError):
            arg_conf_str = f"pred_conf={arg_conf}"
        print(f"      [{tool['name']}] format_conf={confidence:.4f}  {arg_conf_str}  "
              f"{'✓ got call' if matched else '✗ no call'}  ({t:.0f} ms)")

    valid_calls = [r["call"] for r in results if r["call"] is not None]

    # Use the prediction confidence from each tool's argument payload as the
    # routing signal — this is what the model says about its OWN prediction
    # quality, which is more meaningful than the outer cactus JSON confidence
    # (which only reflects tool-call formatting correctness).
    pred_confs = []
    for r in results:
        if r["call"] is not None:
            arg_conf = r["call"].get("arguments", {}).get("confidence")
            try:
                pred_confs.append(float(arg_conf))
            except (TypeError, ValueError):
                pass  # ignore non-numeric confidence args
    if not pred_confs:
        pred_confs = [r["confidence"] for r in results]  # fallback to outer conf

    avg_confidence = sum(pred_confs) / len(pred_confs) if pred_confs else 0.0
    min_confidence = min(pred_confs) if pred_confs else 0.0

    # Also track outer (format) confidence for diagnostics
    outer_confs = [r["confidence"] for r in results]
    outer_avg = sum(outer_confs) / len(outer_confs) if outer_confs else 0.0

    return {
        "function_calls": valid_calls,
        "per_tool": results,
        "confidence": avg_confidence,       # prediction quality confidence
        "min_confidence": min_confidence,
        "outer_confidence": outer_avg,       # tool-call format confidence (diagnostic)
        "total_time_ms": total_time_ms,
        "cloud_handoff": False,
    }


# ─── Cloud inference (Gemini / Google DeepMind) ───────────────────────────────

def generate_gemini_multimodal(images: list[dict]) -> dict | None:
    """
    Multimodal analysis via Gemini (Google DeepMind) cloud.
    All 6 camera images are sent as inline PNG blobs alongside the task description.
    All three prediction tools are requested in a single call.
    Returns None if GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠  GEMINI_API_KEY not set — skipping cloud inference.")
        print("     Set it with: export GEMINI_API_KEY=your_key")
        return None

    client = genai.Client(api_key=api_key)
    gemini_tools = [types.Tool(function_declarations=[_to_gemini_decl(t) for t in ALL_TOOLS])]

    parts = [
        types.Part.from_text(text=(
            "You are a satellite imagery expert. Below are 6 camera images from a CubeSat "
            "in low Earth orbit. Each image is labelled by its camera orientation.\n\n"
            "Based ONLY on visual evidence in these images, call ALL THREE tools:\n"
            "  • predict_geolocation – lat/lon from coastlines, land/ocean, cloud patterns, terminator.\n"
            "  • predict_altitude    – altitude from Earth-limb curvature and atmospheric haze.\n"
            "  • predict_datetime    – UTC time from solar angle, shadows, terminator, seasonal cues.\n\n"
            "Provide a confidence (0–1) and concise reasoning for each prediction."
        ))
    ]

    for img in images:
        parts.append(types.Part.from_text(text=f"\n--- Camera: {img['label']} ---"))
        parts.append(types.Part.from_bytes(data=decode_b64_image(img["url"]), mime_type="image/png"))

    start = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Content(parts=parts, role="user"),
        config=types.GenerateContentConfig(tools=gemini_tools),
    )
    total_time_ms = (time.time() - start) * 1000

    function_calls = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ─── Hybrid routing ───────────────────────────────────────────────────────────

def analyse_cubesat_hybrid(
    images: list[dict],
    confidence_threshold: float = 0.75,
) -> dict:
    """
    Hybrid inference strategy for CubeSat telemetry prediction:

    1. Run FunctionGemma (Cactus) with one focused call per tool (3 total).
    2. Evaluate per-tool confidence. Accept local results only when:
       - The average confidence across all 3 predictions >= threshold AND
       - All 3 tools produced a valid call.
    3. Otherwise defer to Gemini cloud with the actual satellite images for
       multimodal, high-quality analysis.
    """
    print("\n[Step 1/2] On-device FunctionGemma (Cactus) – 3 focused inferences")
    local = generate_cactus_all(images)

    avg_conf = local["confidence"]
    min_conf = local["min_confidence"]
    n_calls = len(local["function_calls"])

    print(f"\n  Avg confidence : {avg_conf:.4f}  |  Min confidence: {min_conf:.4f}")
    print(f"  Valid tool calls: {n_calls}/3  |  Threshold: {confidence_threshold}")

    # Accept local results if confidence is sufficient, even with partial tool coverage.
    # FunctionGemma-270m is text-only and may miss the datetime call — still useful for
    # geo + altitude. Require all 3 only when cloud is available for a clean comparison.
    enough_calls = n_calls >= 2  # at least geo + altitude
    if avg_conf >= confidence_threshold and enough_calls:
        local["source"] = "on-device (FunctionGemma / Cactus)"
        local["local_result"] = local
        local["cloud_result"] = None
        print(f"  → Using on-device result ({n_calls}/3 tools, avg conf {avg_conf:.4f})\n")
        return local

    # Determine why we fell back
    reasons = []
    if avg_conf < confidence_threshold:
        reasons.append(f"avg confidence {avg_conf:.4f} < {confidence_threshold}")
    if not enough_calls:
        reasons.append(f"only {n_calls}/3 tool calls succeeded")
    reason_str = "; ".join(reasons)

    print(f"  → Deferring to Gemini cloud ({reason_str})\n")
    print("[Step 2/2] Cloud inference – Gemini (Google DeepMind) multimodal")

    cloud = generate_gemini_multimodal(images)
    if cloud is None:
        # API key missing; return the best local result we have
        print("  → Returning best available on-device results (cloud unavailable)\n")
        local["source"] = "on-device (FunctionGemma / Cactus) — cloud unavailable"
        local["local_result"] = local
        local["cloud_result"] = None
        return local

    cloud_time_ms = cloud["total_time_ms"]
    cloud["source"] = "cloud (Gemini 2.5 Flash / Google DeepMind)"
    cloud["local_confidence"] = avg_conf
    cloud["local_min_confidence"] = min_conf
    cloud["local_calls"] = n_calls
    cloud["total_time_ms"] += local["total_time_ms"]
    cloud["local_result"] = local
    cloud["cloud_result"] = cloud
    cloud["cloud_time_ms"] = cloud_time_ms
    cloud["local_time_ms"] = local["total_time_ms"]
    return cloud


# ─── Output formatting ────────────────────────────────────────────────────────

def _fmt_conf(v) -> str:
    """Format a confidence value, clamping to [0,1] and flagging invalid values."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
        if f < 0 or f > 1:
            return f"{f:.3f}  ⚠ (out of range)"
        return f"{f:.3f}"
    except (TypeError, ValueError):
        return f"? ({v!r})"


def print_analysis(result: dict) -> None:
    w = 62
    print(f"\n{'═'*w}")
    print(f"  CubeSat Telemetry Prediction")
    print(f"{'═'*w}")
    print(f"  Source         : {result.get('source', 'unknown')}")
    print(f"  Total time     : {result['total_time_ms']:.0f} ms")

    if "local_confidence" in result:
        print(f"  Local avg conf : {result['local_confidence']:.4f}  (deferred to cloud)")
        print(f"  Local min conf : {result['local_min_confidence']:.4f}")
        print(f"  Local calls    : {result['local_calls']}/3")
    elif "confidence" in result:
        print(f"  Pred. conf avg : {result['confidence']:.4f}  (model's self-reported prediction quality)")
        mc = result.get('min_confidence')
        if mc is not None:
            print(f"  Pred. conf min : {mc:.4f}")
        oc = result.get('outer_confidence')
        if oc is not None:
            print(f"  Format conf    : {oc:.4f}  (tool-call JSON formatting confidence)")

    print()
    calls = result.get("function_calls", [])
    if not calls:
        print("  ⚠  No prediction calls returned.")
        print(f"{'═'*w}\n")
        return

    call_by_name = {c["name"]: c.get("arguments", {}) for c in calls}

    # Geolocation
    if "predict_geolocation" in call_by_name:
        a = call_by_name["predict_geolocation"]
        lat = a.get("latitude")
        lon = a.get("longitude")
        try:
            lat_str = f"{float(lat):+.4f}°" if lat is not None else "N/A"
            lon_str = f"{float(lon):+.4f}°" if lon is not None else "N/A"
        except (TypeError, ValueError):
            lat_str, lon_str = str(lat), str(lon)
        print(f"  📍 Geolocation")
        print(f"     Latitude   : {lat_str}")
        print(f"     Longitude  : {lon_str}")
        print(f"     Confidence : {_fmt_conf(a.get('confidence'))}")
        if a.get("reasoning"):
            print(f"     Reasoning  : {a['reasoning']}")
        print()

    # Altitude
    if "predict_altitude" in call_by_name:
        a = call_by_name["predict_altitude"]
        alt = a.get("altitude_km")
        try:
            alt_str = f"{float(alt):.0f} km" if alt is not None else "N/A"
        except (TypeError, ValueError):
            alt_str = str(alt)
        print(f"  🛰  Altitude")
        print(f"     Altitude   : {alt_str}")
        print(f"     Confidence : {_fmt_conf(a.get('confidence'))}")
        if a.get("reasoning"):
            print(f"     Reasoning  : {a['reasoning']}")
        print()

    # Datetime
    if "predict_datetime" in call_by_name:
        a = call_by_name["predict_datetime"]
        dt = a.get("estimated_datetime", "N/A")
        print(f"  🕒 Datetime (UTC)")
        print(f"     Estimated  : {dt}")
        print(f"     Confidence : {_fmt_conf(a.get('confidence'))}")
        if a.get("reasoning"):
            print(f"     Reasoning  : {a['reasoning']}")
        print()

    print(f"{'═'*w}\n")


def _cell(val: str, w: int) -> str:
    """Pad or truncate val to width w for table cell."""
    if len(val) > w:
        return val[: w - 1] + "…"
    return val + " " * (w - len(val))


def _format_lat(v) -> str:
    if v is None:
        return "failed"
    try:
        return f"{float(v):+.1f}°"
    except (TypeError, ValueError):
        return str(v)


def _format_lon(v) -> str:
    return _format_lat(v)


def _format_alt(v) -> str:
    if v is None:
        return "failed"
    try:
        f = float(v)
        if f < 0 or f > 2000:
            return f"{f:.0f} km (invalid)"
        return f"{f:.0f} km"
    except (TypeError, ValueError):
        return str(v)


def _format_dt(v) -> str:
    if v is None or v == "N/A":
        return "failed"
    return str(v)


def print_comparison_report(result: dict, ground_truth: dict | None) -> None:
    """
    Print a comparison table: FunctionGemma (local), Gemini (cloud), Ground Truth.
    Requires result to contain local_result; cloud_result and cloud_time_ms when cloud was used.
    """
    W0, W1, W2, W3 = 16, 34, 28, 14
    sep = "├" + "─" * W0 + "┼" + "─" * W1 + "┼" + "─" * W2 + "┼" + "─" * W3 + "┤"
    top = "┌" + "─" * W0 + "┬" + "─" * W1 + "┬" + "─" * W2 + "┬" + "─" * W3 + "┐"
    bot = "└" + "─" * W0 + "┴" + "─" * W1 + "┴" + "─" * W2 + "┴" + "─" * W3 + "┘"

    local = result.get("local_result") or result
    cloud = result.get("cloud_result")
    local_calls = {c["name"]: c.get("arguments", {}) for c in local.get("function_calls", [])}
    cloud_calls = {c["name"]: c.get("arguments", {}) for c in (cloud or {}).get("function_calls", [])} if cloud else {}
    local_time_s = local.get("total_time_ms", 0) / 1000
    cloud_time_s = result.get("cloud_time_ms") / 1000 if result.get("cloud_time_ms") is not None else None
    avg_conf = local.get("confidence")
    threshold = 0.75
    routing_str = "—"
    if avg_conf is not None:
        try:
            c = float(avg_conf)
            if c < threshold:
                routing_str = f"{c:.2f} (below threshold)"
            else:
                routing_str = f"{c:.2f}"
        except (TypeError, ValueError):
            routing_str = str(avg_conf)

    def gt(key: str, fmt=str) -> str:
        if not ground_truth or ground_truth.get(key) is None:
            return "—"
        return fmt(ground_truth[key])

    # Row values
    geo = local_calls.get("predict_geolocation", {})
    alt = local_calls.get("predict_altitude", {})
    dt = local_calls.get("predict_datetime", {})
    local_lat = _format_lat(geo.get("latitude"))
    local_lon = _format_lon(geo.get("longitude"))
    local_alt = _format_alt(alt.get("altitude_km"))
    local_dt = _format_dt(dt.get("estimated_datetime"))

    c_geo = cloud_calls.get("predict_geolocation", {}) if cloud else {}
    c_alt = cloud_calls.get("predict_altitude", {}) if cloud else {}
    c_dt = cloud_calls.get("predict_datetime", {}) if cloud else {}
    cloud_lat = _format_lat(c_geo.get("latitude")) if cloud else "—"
    cloud_lon = _format_lon(c_geo.get("longitude")) if cloud else "—"
    cloud_alt = _format_alt(c_alt.get("altitude_km")) if cloud else "—"
    cloud_dt = _format_dt(c_dt.get("estimated_datetime")) if cloud else "—"

    gt_lat = gt("latitude", lambda x: f"{float(x):+.1f}°")
    gt_lon = gt("longitude", lambda x: f"{float(x):+.1f}°")
    gt_alt = gt("altitude", lambda x: f"{float(x):.0f} km")
    gt_dt = gt("datetime")

    time_local = f"{local_time_s:.1f}s"
    time_cloud = f"{cloud_time_s:.1f}s total" if cloud_time_s is not None else "—"

    print(top)
    print("│" + _cell("", W0) + "│" + _cell("FunctionGemma (local, text-only)", W1) + "│" + _cell("Gemini (cloud, multimodal)", W2) + "│" + _cell("Ground Truth", W3) + "│")
    print(sep)
    print("│" + _cell("Latitude", W0) + "│" + _cell(local_lat, W1) + "│" + _cell(cloud_lat, W2) + "│" + _cell(gt_lat, W3) + "│")
    print(sep)
    print("│" + _cell("Longitude", W0) + "│" + _cell(local_lon, W1) + "│" + _cell(cloud_lon, W2) + "│" + _cell(gt_lon, W3) + "│")
    print(sep)
    print("│" + _cell("Altitude", W0) + "│" + _cell(local_alt, W1) + "│" + _cell(cloud_alt, W2) + "│" + _cell(gt_alt, W3) + "│")
    print(sep)
    print("│" + _cell("Datetime", W0) + "│" + _cell(local_dt, W1) + "│" + _cell(cloud_dt, W2) + "│" + _cell(gt_dt, W3) + "│")
    print(sep)
    print("│" + _cell("Routing conf", W0) + "│" + _cell(routing_str, W1) + "│" + _cell("—", W2) + "│" + _cell("—", W3) + "│")
    print(sep)
    print("│" + _cell("Inference time", W0) + "│" + _cell(time_local, W1) + "│" + _cell(time_cloud, W2) + "│" + _cell("—", W3) + "│")
    print(bot)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    json_files = _glob.glob("data/cubesat-data*.json")
    if not json_files:
        print("ERROR: No cubesat-data*.json found in data/")
        sys.exit(1)

    for json_path in sorted(json_files):
        print(f"\n{'═'*60}")
        print(f"CubeSat data file : {json_path}")
        images = load_images_only(json_path)
        print(f"Images loaded     : {len(images)}  ({', '.join(img['label'] for img in images)})")
        result = analyse_cubesat_hybrid(images)
        print_analysis(result)
        ground_truth = load_telemetry(json_path)
        print_comparison_report(result, ground_truth)
