"""EARTH PULSE - Phase 2 app. Shows ONLY observations retrieved from NASA products and cached in
data/processed. No data => empty state. Nothing here is generated or simulated."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.pulse_analysis import (DEFAULTS, RULES_TEXT, change_story, classify_signature, evidence_panel,
                                    series_from_arrays, to_db)
from modules.forecast import forecast_explanation, run_forecast
from modules.pulse_compare import compare_stacks

st.set_page_config(page_title="EARTH PULSE", layout="wide")

def mission_chart(fig, **kwargs):
    """Visual-only Plotly theme; keeps input data, plot types, and scientific axes intact."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#DDEAF7", size=12),
        title_font=dict(family="Orbitron, sans-serif", color="#F1FAFF"),
        xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332", linecolor="#31516b", tickfont=dict(family="JetBrains Mono, monospace")),
        yaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332", linecolor="#31516b", tickfont=dict(family="JetBrains Mono, monospace")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(family="JetBrains Mono, monospace")),
    )
    # Do not recolor heatmaps, raster imagery, difference maps or category-coded traces.
    for trace in fig.data:
        if trace.type == "scatter" and getattr(trace, "mode", None) and "lines" in trace.mode:
            if getattr(trace, "line", None) is not None and not trace.line.color:
                trace.line.color = "#00D4FF"
    return st.plotly_chart(fig, **kwargs)

BLUE, GREEN, NAVY = "#4aa3df", "#2fbf8f", "#0b1a2e"
# NASA Blue Marble is decorative context only, never a NISAR measurement.
# Source: https://science.nasa.gov/resource/blue-marble/
NASA_EARTH_PHOTO = "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/climate/2023/12/ImageWall5_1920x1200-80.jpg"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Exo+2:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {color-scheme:dark; --ep-cyan:#00D4FF; --ep-blue:#0B3D91; --ep-ink:#050810; --ep-border:rgba(0,212,255,.20);}
html,body,[data-testid="stAppViewContainer"] {background:#050810 !important;color:#EAF2FC;}
[data-testid="stAppViewContainer"] {background-image:linear-gradient(100deg,rgba(5,8,16,.97),rgba(5,8,16,.94)),url('https://assets.science.nasa.gov/dynamicimage/assets/science/esd/climate/2023/12/ImageWall5_1920x1200-80.jpg') !important;background-size:cover !important;background-position:center !important;background-attachment:fixed !important;}
[data-testid="stAppViewContainer"]::before {content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.22;background-image:radial-gradient(1px 1px at 13% 18%,#d4efff 98%,transparent),radial-gradient(1px 1px at 76% 32%,#d4efff 98%,transparent),radial-gradient(1px 1px at 40% 81%,#d4efff 98%,transparent),radial-gradient(1px 1px at 91% 71%,#d4efff 98%,transparent);background-size:480px 350px;animation:ep-stars 75s linear infinite;}
@keyframes ep-stars {to {background-position:480px 350px;}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
[data-testid="stSidebar"] {background:rgba(6,12,26,.97) !important;border-right:1px solid var(--ep-border) !important;}
[data-testid="stMainBlockContainer"] {max-width:1460px !important;padding-top:1.1rem !important;padding-bottom:2rem !important;}
h1,h2,h3,.ep-hero h2 {font-family:'Orbitron','Exo 2',sans-serif !important;letter-spacing:.035em;color:#F3FAFF !important;text-shadow:0 0 14px rgba(0,212,255,.13);}
h1 {font-size:clamp(1.45rem,2.7vw,2.3rem) !important;}
p,li,label,span {line-height:1.52;}
[data-testid="stMetric"],[data-testid="stAlert"],[data-testid="stExpander"],[data-testid="stForm"],[data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stPlotlyChart"],[data-testid="stImage"],[data-testid="stPyplotGlobal"] {background:rgba(255,255,255,.055) !important;border:1px solid var(--ep-border) !important;border-radius:12px !important;box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 0 22px rgba(0,212,255,.035);}
[data-testid="stMetric"] {padding:14px 18px;transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;}
[data-testid="stMetric"]:hover {transform:translateY(-3px);border-color:rgba(0,212,255,.45) !important;box-shadow:0 0 22px rgba(0,212,255,.13);}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stDataFrame"] td, [data-testid="stTable"] td, code, pre, time, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {font-family:'JetBrains Mono',monospace !important;font-variant-numeric:tabular-nums;}
[data-testid="stMetricValue"] {color:#EAFBFF !important;text-shadow:0 0 13px rgba(0,212,255,.22);}
[data-testid="stMetricLabel"] {color:#BDCCDC !important;}
[data-testid="stMarkdownContainer"] p {color:#E5EFFB;}
[data-testid="stTextInput"] input,[data-testid="stDateInput"] input,[data-testid="stNumberInput"] input,textarea,[data-baseweb="select"]>div {background:#0B1628 !important;color:#F3FAFF !important;border:1px solid rgba(0,212,255,.25) !important;}
[data-testid="stTabs"] [role="tablist"] {background:rgba(255,255,255,.05);border-radius:10px;}
.stButton>button,[data-testid="stFormSubmitButton"] button {border:1px solid rgba(0,212,255,.34) !important;background:linear-gradient(135deg,#0B3D91,#082A61) !important;color:#F7FBFF !important;transition:box-shadow .2s ease,transform .2s ease;}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover {box-shadow:0 0 15px rgba(0,212,255,.22);transform:translateY(-1px);}
.ep-hero {display:flex;align-items:center;justify-content:space-between;gap:24px;min-height:175px;padding:24px 30px;margin:8px 0 22px;border:1px solid rgba(0,212,255,.27);border-radius:16px;background:linear-gradient(90deg,rgba(5,8,16,.96),rgba(7,20,39,.92) 70%,rgba(5,8,16,.73)),url('https://assets.science.nasa.gov/dynamicimage/assets/science/esd/climate/2023/12/ImageWall5_1920x1200-80.jpg') right 43%/auto 135% no-repeat;box-shadow:0 0 25px rgba(0,212,255,.06);}
.ep-kicker {font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.17em;color:#00D4FF;font-weight:600;}
.ep-kicker::before {content:"";display:inline-block;width:8px;height:8px;margin-right:11px;border-radius:50%;background:#00D4FF;box-shadow:0 0 11px #00D4FF;animation:ep-pulse 2.4s ease-in-out infinite;}
@keyframes ep-pulse {50% {opacity:.35;box-shadow:0 0 3px #00D4FF;}}
.ep-hero h2 {font-size:clamp(1.25rem,2.3vw,2rem);margin:.35rem 0;}
.ep-hero p {max-width:650px;margin:0;color:#DFEBF7;}
.ep-photo-credit {font-family:'JetBrains Mono',monospace;font-size:.66rem;color:#A7BCD1;margin:-12px 0 20px;}
.src {font-family:'JetBrains Mono',monospace;color:#B6D6E8;font-size:.78rem;background:rgba(255,255,255,.055);border:1px solid var(--ep-border);padding:8px 12px;border-radius:8px;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {font-family:'JetBrains Mono',monospace;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:#E8F3FF !important;}
@media(max-width:800px){.ep-hero{padding:18px;min-height:140px;}[data-testid="stMainBlockContainer"]{padding-left:.8rem !important;padding-right:.8rem !important;}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important;}}
</style>
""", unsafe_allow_html=True)



# All data lives in st.session_state (memory). Nothing is written to disk.
def load(name):
    return st.session_state["streams"][name]


def list_stacks():
    return list(st.session_state.get("streams", {}))


def state():
    d = st.session_state.get("discovery")
    if not d:
        return None
    return {"origin": d["origin"], "last_successful_refresh_utc": d["retrieved_at_utc"],
            "n_observations": len(d["observations"]), "n_stacks": len(d["stacks"]),
            "selection_reason": d["selection_reason"], "search_complete": d["report"]["complete"]}


def remember_search_context(aoi, label, start, end):
    """One shared area and period for all Earth Pulse investigation stages."""
    from modules.aoi import center_lonlat
    lon, lat = center_lonlat(aoi)
    new = {"lon": float(lon), "lat": float(lat), "label": label,
           "start": str(start), "end": str(end), "aoi": aoi}
    old = st.session_state.get("active_location")
    if old != new:
        st.session_state.pop("cause_result", None)
        st.session_state.pop("cause_context_key", None)
    st.session_state["active_location"] = new


def source_bar(m=None, e=None):
    s = state()
    bits = ["Source: NASA-ISRO NISAR via ASF DAAC (asf_search)"]
    if s: bits.append(f"archive records: {s['origin']}; last successful live refresh (UTC): {s['last_successful_refresh_utc']}")
    if e: bits.append(f"product: {e['product_id']}; retrieval: {e['retrieval_status']}")
    st.markdown(f"<div class='src'>{' | '.join(bits)}</div>", unsafe_allow_html=True)


def meta_of(m, ok):
    s = m["stack"]; e0 = ok[0]
    return {**s, "layer_path_leaf": e0.get("layer_term"), "units": e0.get("units"), "bbox": m.get("bbox"),
            "first_acquisition": ok[0]["start_time"], "last_acquisition": ok[-1]["start_time"]}


def pick_stack(key):
    names = list_stacks()
    if not names:
        st.warning("No NISAR observations have been streamed in this session. Open LIVE DATA, search the archive "
                   "and stream a stack. Nothing is shown because no real observations are loaded.")
        st.stop()
    name = st.selectbox("Observation stack", names, key=key)
    m, ok, arrays, skipped = load(name)
    for s in skipped: st.error(f"Skipped {s['product_id']}: {s['reason']}")
    if not ok:
        st.error("This stack has no usable verified observations."); st.stop()
    return name, m, ok, arrays


# Streamlit forgets widget values when their page is not shown; re-assigning keeps them.
for _k in ("aoi", "d0", "d1", "ign"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

# Friendly navigation changes labels only; every original page and calculation is retained.
PAGES = {
    "Start here": "HOME", "Get NASA images": "LIVE DATA",
    "Watch Earth change": "WATCH", "Understand a change": "INVESTIGATE",
    "Compare two places": "COMPARE", "What might happen next?": "FORECAST",
    "Satellite image dates": "OBSERVATION CATALOG", "Can we trust this image?": "DATA QUALITY",
    "Save the evidence": "EXPORT / PROVENANCE", "Explore one spot": "PIXEL EXPLORER",
    "Find the biggest changes": "CHANGE HOTSPOTS", "Replay NASA observations": "MISSION CONTROL",
    "Explain it simply": "AI EARTH SCIENTIST", "Explore the world": "3D EARTH EXPLORER",
    "My watched places": "ENVIRONMENTAL WATCHLIST", "Why did this happen?": "CAUSE EXPLORER", "About the NASA data": "ABOUT / DATA SOURCES",
}
INTRO = {
    "HOME": ("Welcome! See how our planet changes over time.", "Start with 'Get NASA images'. Once real observations load, explore them using the menu."),
    "LIVE DATA": ("Find real satellite pictures of a place you care about.", "Choose a place and dates, search NASA, then load the available observations. A free NASA Earthdata login may be needed."),
    "WATCH": ("See the same place on different days.", "Move between real satellite visits. Different shades show radar reflections, not ordinary camera colours."),
    "INVESTIGATE": ("Understand what changed and whether it stayed changed.", "The graph follows satellite readings through time. A change in the reading does not automatically mean a disaster."),
    "COMPARE": ("See how two places changed differently.", "Compare the pattern of change, not just the total difference."),
    "FORECAST": ("See a possible future trend — not a fact.", "Forecasts are computer estimates. They are NOT future NASA observations and may be wrong."),
    "OBSERVATION CATALOG": ("See when NASA actually observed your area.", "Each entry is a real satellite product, not an invented picture."),
    "DATA QUALITY": ("Check how complete the satellite information is.", "Missing observations mean we cannot know exactly what happened between visits."),
    "EXPORT / PROVENANCE": ("Save the results and where they came from.", "Keep the NASA product details so others can check the evidence."),
    "PIXEL EXPLORER": ("Pick a tiny spot and follow its readings.", "One pixel is a small part of the image. The chart shows the readings for that spot across dates."),
    "CHANGE HOTSPOTS": ("Find spots with the biggest measured differences.", "A hotspot means a radar reading changed. It does not prove fire, flooding, or any specific cause."),
    "MISSION CONTROL": ("Replay the satellite visits like a story.", "Each step represents an actual observation date; missing dates are not filled with made-up data."),
    "AI EARTH SCIENTIST": ("Read what the satellite observations mean in everyday English.", "This uses clear rules based on real measurements, not a chatbot making guesses."),
    "3D EARTH EXPLORER": ("See where your loaded NASA images come from.", "The map shows actual observation locations. The background map is not itself a NASA radar image."),
    "ENVIRONMENTAL WATCHLIST": ("Keep an eye on places that matter to you.", "Save a place for this session, then search NASA again to check for new images. No automatic alerts yet."),
    "CAUSE EXPLORER": ("Investigate what might explain an observed change.", "Look for real rainfall, earthquake and event records near a place and time. These clues do not prove the cause."),
    "ABOUT / DATA SOURCES": ("See how Earth Pulse works and where its information comes from.", "We use real NASA NISAR products. We explain observations without guessing their cause."),
}
st.sidebar.markdown("### Earth Pulse")
st.sidebar.caption("Discover Earth's changes with real NASA information")
beginner = st.sidebar.toggle("Simple explanations", value=True, help="Show easy explanations throughout the app. All scientific features remain available.")
if "navigation_page" not in st.session_state or st.session_state["navigation_page"] not in PAGES:
    st.session_state["navigation_page"] = "Start here"
selected_page = st.sidebar.radio("Where would you like to go?", list(PAGES), key="navigation_page")
page = PAGES[selected_page]
# Image is an authentic NASA Blue Marble composite (not a live NISAR scene).
# It is only used in the header and behind opaque surfaces; all analysis is unchanged.
st.markdown(f"""<section class='ep-hero'><div>
<div class='ep-kicker'>EARTH PULSE / EARTH OBSERVATION</div>
<h2>{selected_page.split(' ', 1)[-1]}</h2>
<p>Explore Earth's changing surface through real NASA observations, with clear explanations and traceable evidence.</p>
</div></section>
<div class='ep-photo-credit'>Background image: NASA Blue Marble (illustrative Earth imagery, not a NISAR observation). Credit: NASA Goddard / Reto Stöckli / Robert Simmon.</div>""", unsafe_allow_html=True)
if beginner:
    headline, explanation = INTRO[page]
    st.info(f"**{headline}**\n\n{explanation}")
    with st.sidebar.expander("Simple word guide"):
        st.markdown("**Satellite pass:** one visit by the satellite.\n\n**Radar:** a sensor that uses radio waves to observe Earth's surface.\n\n**Pixel:** one small spot in an image.\n\n**dB:** a way to measure radar signal strength.\n\n**Backscatter:** radar energy reflected back from the ground.\n\n**Uncertainty:** what the available information cannot tell us.\n\n**Forecast:** a prediction, not an observation.")


if page == "HOME":
    st.title("EARTH PULSE")
    if beginner:
        st.markdown("### Three easy steps")
        a,b,c = st.columns(3)
        a.info("**1. Pick a place**\n\nGo to Get NASA images.")
        b.info("**2. Watch it change**\n\nOpen Watch Earth change.")
        c.info("**3. Understand it**\n\nOpen Explain it simply.")
    st.subheader("A satellite image shows a moment. Earth Pulse helps us understand the process.")
    st.write("Earth Pulse follows how NISAR radar backscatter evolves across repeated, compatible acquisitions "
             "and describes the temporal pattern - sudden, gradual, persistent, reversing or repeated - "
             "without claiming to know the cause.")
    s = state(); names = list_stacks()
    c = st.columns(4)
    c[0].metric("Archive search", "done" if s else "not run")
    c[1].metric("Observations found", s["n_observations"] if s else "unavailable")
    c[2].metric("Candidate stacks", s["n_stacks"] if s else "unavailable")
    c[3].metric("Retrieved stacks", len(names))
    if s: st.caption(f"Last successful live refresh (UTC): {s['last_successful_refresh_utc']} | {s['selection_reason']}")
    for n in names:
        m, ok, _, _ = load(n)
        st.write(f"**{n}**: {len(ok)} verified observations, {ok[0]['start_time'][:10]} to {ok[-1]['start_time'][:10]}")
    if not names: st.info("No stack has been retrieved yet, so no observation locations or periods are listed.")
    source_bar()

elif page == "LIVE DATA":
    from datetime import date, datetime, timezone
    from modules.earthdata_auth import credentials_available
    from modules.live_stream import discover, stream_stack
    st.title("LIVE DATA - stream NISAR straight from NASA")
    st.caption("Search results and image windows are held in memory for this session only. Nothing is saved to your computer.")
    if not credentials_available():
        st.error("No Earthdata credentials found. Set EARTHDATA_TOKEN (or EARTHDATA_USERNAME + EARTHDATA_PASSWORD) "
                 "as environment variables or in .streamlit/secrets.toml. Searching may work without them; streaming needs them.")
    st.subheader("Find a place on Earth")
    st.caption("Search by a familiar place name. We'll find its real location, then search NASA's archive there.")
    mode = st.radio("How would you like to search?", ["Place name (easy)", "Coordinates (advanced)"], horizontal=True)
    if mode == "Place name (easy)":
        search_col, find_col = st.columns([4, 1])
        query = search_col.text_input("City, town, region, or landmark", placeholder="e.g. Durban, South Africa; Los Angeles; Mount Etna", key="place_query")
        if find_col.button("Find place", type="secondary"):
            try:
                from modules.place_search import find_places
                st.session_state["place_matches"] = find_places(query)
                st.session_state["place_query_resolved"] = query
                if not st.session_state["place_matches"]:
                    st.warning("No places found. Try adding the country or a nearby city.")
            except Exception as ex:
                st.session_state["place_matches"] = []
                st.error(f"Place lookup failed: {ex}. You can use the advanced coordinates option instead.")
        matches = st.session_state.get("place_matches", []) if st.session_state.get("place_query_resolved") == query else []
        if matches:
            selected = st.selectbox("Choose the correct place", range(len(matches)),
                                    format_func=lambda i: matches[i]["name"], key="place_choice")
            chosen = matches[selected]
            aoi = f'{chosen["lon"]:.6f},{chosen["lat"]:.6f}'
            st.success(f'Selected: {chosen["name"]}')
            st.caption("NASA search centre (longitude, latitude): " + aoi + ". This is a geographic lookup, not a satellite observation.")
        else:
            aoi = None
            st.info("Type a place and click **Find place**, then select the correct result before searching NASA.")
    else:
        aoi = st.text_input("Coordinates: longitude,latitude (or minlon,minlat,maxlon,maxlat)",
                            key="aoi", placeholder="e.g. -122.4,37.8")
    c2, c3 = st.columns(2)
    d0 = c2.date_input("From", date(2025, 10, 1), key="d0")
    d1 = c3.date_input("To", date.today(), key="d1")
    ign = st.checkbox("The API does not provide data maturity: group without it (UNVERIFIED)", value=False, key="ign")
    if st.button("1. Search NASA archive", type="primary", disabled=not bool(aoi)):
        try:
            if d1 < d0:
                st.error("End date must not be before the start date.")
                st.stop()
            location_label = chosen["name"] if mode == "Place name (easy)" and matches else "Selected coordinates"
            remember_search_context(aoi, location_label, d0, d1)
            bar = st.progress(0.0, text="Contacting ASF Search API...")
            def on_w(w):
                bar.progress(min(1.0, w["call"] / w["max_calls"]), text=f"request {w['call']}: {w['returned']} records")
            day = lambda d: datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
            st.session_state["discovery"] = discover(aoi, day(d0), day(d1), ("maturity",) if ign else (), on_window=on_w)
            bar.empty()
        except Exception as ex:
            st.error(f"Archive search failed (nothing was fabricated): {type(ex).__name__}: {ex}")
    d = st.session_state.get("discovery")
    if d:
        c = st.columns(4)
        c[0].metric("Observations found", len(d["observations"])); c[1].metric("Stacks", len(d["stacks"]))
        c[2].metric("Eligible stacks", sum(x["eligible"] for x in d["stacks"])); c[3].metric("Search complete", str(d["report"]["complete"]))
        st.caption(f"Source: ASF Search API | retrieved (UTC): {d['retrieved_at_utc']} | {d['origin']}")
        st.info(d["selection_reason"])
        if d.get("auto_ignored_fields"):
            st.warning("The API leaves these fields empty for every product: " + ", ".join(d["auto_ignored_fields"]) +
                       ". They were not used for grouping; the polarization layer is verified from each file when streaming.")
        if d["stacks"]:
            st.dataframe(pd.DataFrame(d["stacks"])[["rank", "track", "frame", "direction", "polarization", "maturity", "release",
                                                    "n_observations", "first_acquisition", "last_acquisition", "max_gap_days", "eligible"]],
                         use_container_width=True)
        with st.expander("Metadata diagnostics (what the API did and did not provide)"):
            st.write({"missing field counts": d["missing_field_counts"], "unstackable records": d["unstackable"],
                      "QA/auxiliary excluded": d["excluded_qa_or_auxiliary"], "duplicates": d["dedupe"]})
            st.write("Property names returned by the API:", d["sample_property_keys"])
            st.write("Raw metadata of the first product (public archive metadata):")
            st.json(d.get("sample_properties", {}))
            st.write("How Earth Pulse interpreted it (None = not found):")
            st.json(d.get("sample_normalized", {}))
        if d["stacks"]:
            r1, r2, r3, r4 = st.columns(4)
            rank = r1.number_input("Stack rank to stream", 1, len(d["stacks"]), 1)
            half = r2.slider("Window half-size (pixels)", 50, 1000, 300, 50)
            stride = r3.selectbox("Pixel stride", [1, 2, 4], index=0)
            maxp = r4.slider("Max observations", 4, 40, 16)
            if st.button("2. Stream observations from NASA", type="primary"):
                row = d["stacks"][int(rank) - 1]
                bar = st.progress(0.0)
                def prog(i, n, o):
                    bar.progress(i / max(n, 1), text=f"Streaming {i + 1}/{n}: {o['product_id'][:70]}")
                try:
                    from modules.aoi import center_lonlat
                    lon, lat = center_lonlat(d["aoi_text"])      # the area that was actually searched
                    st.caption(f"Streaming around longitude {lon}, latitude {lat} (the searched area).")
                    m, ok, arrays, skipped = stream_stack(row, d["observations"], lon, lat, half, stride, maxp, progress=prog)
                    bar.empty()
                    for sk in skipped: st.warning(f"Skipped {sk['product_id']}: {sk['reason']}")
                    if ok:
                        st.session_state.setdefault("streams", {})[row["stack_id"][:90]] = (m, ok, arrays, skipped)
                        st.success(f"Streamed {len(ok)} observations into memory. Your location and dates are ready for every investigation step.")
                        st.info("Next: open Watch Earth change, then Understand a change, then Why did this happen? No second location search is needed.")
                    else:
                        st.error("No observation could be streamed. See the reasons above; nothing was fabricated.")
                except Exception as ex:
                    st.error(f"Streaming failed: {type(ex).__name__}: {ex}")

elif page == "WATCH":
    st.title("WATCH - see the surface change")
    name, m, ok, arrays = pick_stack("w")
    dates = [e["start_time"] for e in ok]
    i = st.select_slider("Acquisition", options=list(range(len(ok))), format_func=lambda k: dates[k][:19])
    e = ok[i]; db = to_db(arrays[i])
    fig = go.Figure(go.Heatmap(z=db, colorscale="Viridis", colorbar=dict(title="dB")))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                                                              title=f"{e['layer_term']} backscatter (dB), {dates[i][:19]}")
    mission_chart(fig, use_container_width=True)
    q = e["quality"]
    st.write(f"**Product ID:** {e['product_id']}  \n**Acquired:** {dates[i]}  \n**Polarization/layer:** "
             f"{e['layer_term']} ({e['layer_path']})  \n**Units:** {e.get('units') or 'unavailable'}  \n"
             f"**Valid pixels:** {q['valid_pixel_pct']}%  |  **Missing:** {q['missing_pct']}%  |  "
             f"sampled every {e['stride']}th pixel")
    if len(ok) > 1:
        j = st.selectbox("Compare with", [k for k in range(len(ok)) if k != i], format_func=lambda k: dates[k][:19])
        diff = to_db(arrays[j]) - db
        f2 = go.Figure(go.Heatmap(z=diff, colorscale="RdBu", zmid=0, colorbar=dict(title="dB")))
        f2.update_yaxes(autorange="reversed")
        f2.update_layout(height=450, title=f"Difference: {dates[j][:10]} minus {dates[i][:10]} (dB)")
        mission_chart(f2, use_container_width=True)
    source_bar(m, e)

elif page == "INVESTIGATE":
    st.title("INVESTIGATE - the Pulse Lab")
    name, m, ok, arrays = pick_stack("i")
    thr = st.slider("Minimum meaningful change (dB) - analyst parameter, not a confidence level", 0.1, 3.0,
                    DEFAULTS["threshold_db"], 0.1)
    try:
        ser = series_from_arrays(arrays)
    except ValueError as ex:
        st.error(str(ex)); st.stop()
    dates = [e["start_time"][:19] for e in ok]
    sig = classify_signature(ser["mean_db"], ser["common_valid_pct"], threshold_db=thr)
    units = ok[0].get("units")
    st.header(f"EARTH PULSE SIGNATURE: {sig['label']}" + (f" - {', '.join(sig['tags'])}" if sig["tags"] else ""))
    st.caption("Descriptive analytical label, not a confirmed natural-event classification. Rule applied: " + sig["rule"])
    fig = go.Figure(go.Scatter(x=dates, y=ser["mean_db"], mode="lines+markers", name="measured (NISAR)",
                               line=dict(color=GREEN)))
    fig.update_layout(height=380, yaxis_title="mean backscatter (dB)", xaxis_title="acquisition")
    mission_chart(fig, use_container_width=True)
    if sig["metrics"]:
        mt = sig["metrics"]
        c = st.columns(4)
        c[0].metric("Net change (dB)", f"{mt['net_change_db']:+.2f}")
        c[1].metric("Significant steps", mt["n_significant_steps"])
        c[2].metric("Largest step (dB)", f"{mt['largest_step_db']:+.2f}")
        c[3].metric("Common valid pixels", f"{ser['common_valid_pct']:.1f}%")
    st.subheader("Change story")
    st.write(change_story(dates, sig, ser, units))
    with st.expander("Evidence panel"):
        st.json(evidence_panel(ok, m["stack"], ok[0]["layer_path"], units, ok[0]["stride"]))
    with st.expander("Classification rules"):
        st.code(RULES_TEXT)
    source_bar(m)

elif page == "COMPARE":
    st.title("COMPARE - two Earths")
    names = list_stacks()
    if len(names) < 2:
        st.warning("Comparison needs two retrieved stacks; " f"{len(names)} available. Nothing is compared."); st.stop()
    a_name = st.selectbox("Stack A", names, key="ca"); b_name = st.selectbox("Stack B", [n for n in names if n != a_name], key="cb")
    (ma, oka, ara, _), (mb, okb, arb, _) = load(a_name), load(b_name)
    if not oka or not okb:
        st.error("A selected stack has no usable observations."); st.stop()
    res = compare_stacks(meta_of(ma, oka), meta_of(mb, okb))
    for b in res["blocking"]: st.error("Comparison blocked: " + b)
    for w in res["warnings"]: st.warning(w)
    if res["allowed"]:
        sa, sb = series_from_arrays(ara), series_from_arrays(arb)
        rel = res["mode"].startswith("relative")
        fa = np.array(sa["mean_db"]); fb = np.array(sb["mean_db"])
        if rel: fa, fb = fa - fa[0], fb - fb[0]
        fig = go.Figure()
        fig.add_scatter(x=[e["start_time"][:19] for e in oka], y=fa, name=a_name[:40], line=dict(color=GREEN))
        fig.add_scatter(x=[e["start_time"][:19] for e in okb], y=fb, name=b_name[:40], line=dict(color=BLUE))
        fig.update_layout(height=400, yaxis_title="dB relative to first observation" if rel else "mean backscatter (dB)")
        mission_chart(fig, use_container_width=True)
        st.write(f"Mode: {res['mode']}. Spatial overlap: {res['spatial_overlap']}. Temporal overlap (days): "
                 f"{res['temporal_overlap_days']}.")
        for n, s, sg in ((a_name, sa, classify_signature(sa["mean_db"], sa["common_valid_pct"])),
                         (b_name, sb, classify_signature(sb["mean_db"], sb["common_valid_pct"]))):
            st.write(f"**{n[:50]}** - {sg['label']}; common valid pixels {s['common_valid_pct']:.1f}%")
    source_bar()

elif page == "FORECAST":
    st.title("FORECAST - what might the next measurement look like?")
    st.caption("Forecasts the measured radar signal of this area at a future acquisition. It does not predict "
               "floods, earthquakes or any other event, and implies no cause.")
    name, m, ok, arrays = pick_stack("f")
    try:
        ser = series_from_arrays(arrays)
    except ValueError as ex:
        st.error(str(ex)); st.stop()
    dates = [e["start_time"] for e in ok]
    with st.spinner("Training models chronologically..."):
        res = run_forecast(dates, ser["mean_db"], ser["per_obs_valid_pct"], [e["product_id"] for e in ok])
    if res["status"] != "ok":
        st.warning(f"{res['message']}  ({res['reason']})")
        st.info("Historical measurements are still shown on INVESTIGATE. No model was trained.")
        source_bar(m); st.stop()
    f = res["forecast"]
    fig = go.Figure()
    fig.add_scatter(x=[d[:10] for d in dates], y=ser["mean_db"], mode="lines+markers", name="MEASURED (NISAR)",
                    line=dict(color=GREEN, width=3), marker=dict(size=8))
    fig.add_scatter(x=[res["last_observation"]["date"], f["forecast_date"]], y=[res["last_observation"]["value_db"], f["value_db"]],
                    mode="lines+markers", name="MODEL ESTIMATE (not an observation)",
                    line=dict(color="#f0a030", dash="dash"), marker=dict(symbol="diamond-open", size=12))
    if f["interval"]:
        iv = f["interval"]
        fig.add_scatter(x=[f["forecast_date"]] * 2, y=[iv["lower"], iv["upper"]], mode="lines", name=f"~{iv['level']:.0%} empirical interval",
                        line=dict(color="#f0a030", width=6), opacity=.35)
    fig.add_vrect(x0=res["test_period"][0], x1=res["test_period"][1], fillcolor=BLUE, opacity=.12, line_width=0,
                  annotation_text="held-out test period")
    fig.update_layout(height=430, yaxis_title="mean backscatter (dB)", xaxis_title="acquisition date")
    mission_chart(fig, use_container_width=True)
    c = st.columns(4)
    c[0].metric("Selected model", res["selected_model"].replace("_", " "))
    c[1].metric("Forecast (model estimate)", f"{f['value_db']:.2f} dB")
    c[2].metric("Horizon", f"{f['horizon_days']:.0f} d -> {f['forecast_date']}")
    c[3].metric("Observations", res["n_observations"])
    st.write(f"**Training period:** {res['training_period'][0]} to {res['training_period'][1]} ({res['n_train_rows']} rows)  \n"
             f"**Test period:** {res['test_period'][0]} to {res['test_period'][1]} ({res['n_test_rows']} rows)")
    tm, vm = res["test_metrics"], res["validation_metrics"]
    st.subheader("Evaluation vs persistence baseline")
    st.dataframe(pd.DataFrame([{"model": k, "test MAE (dB)": round(tm[k]["mae"], 3), "test RMSE (dB)": round(tm[k]["rmse"], 3),
                                "validation RMSE (dB)": round(vm[k]["rmse"], 3),
                                "beats persistence on test": res["beats_persistence_on_test"].get(k, "baseline")} for k in tm]),
                 use_container_width=True)
    st.subheader("Explanation")
    st.write(forecast_explanation(res))
    with st.expander("Limitations"):
        for l in res["limitations"]: st.write("- " + l)
    with st.expander("Evidence and reproducibility"):
        st.json({"product_ids": res["product_ids"], "features": res["features"], "seed": res["seed"],
                 "data_fingerprint": res["data_fingerprint"], "selection_basis": res["selection_basis"],
                 "horizon_assumption": f["horizon_assumption"]})
    source_bar(m)

elif page == "OBSERVATION CATALOG":
    st.title("OBSERVATION CATALOG - actual NASA acquisitions")
    st.caption("Every row comes from metadata returned by NASA/ASF or a product streamed in this session. No example records are inserted.")
    d = st.session_state.get("discovery")
    if not d:
        st.info("Run an archive search on LIVE DATA to populate this catalog.")
    else:
        rows = d.get("observations", [])
        st.metric("Archive observations", len(rows))
        if rows:
            frame = pd.json_normalize(rows, sep=".")
            terms = st.text_input("Filter product metadata (case insensitive)", placeholder="Product ID, track, date, polarization...")
            if terms:
                frame = frame[frame.astype(str).apply(lambda col: col.str.contains(terms, case=False, regex=False)).any(axis=1)]
            st.dataframe(frame, use_container_width=True, height=450)
            st.download_button("Download filtered NASA metadata (CSV)", frame.to_csv(index=False).encode("utf-8"),
                               file_name="earth_pulse_nasa_catalog.csv", mime="text/csv")
        st.caption("Discovery metadata alone is not proof that image pixels were successfully downloaded or validated.")
    source_bar()

elif page == "DATA QUALITY":
    st.title("DATA QUALITY - what can we trust?")
    name, m, ok, arrays = pick_stack("quality")
    rows = []
    for e, arr in zip(ok, arrays):
        q = e.get("quality", {})
        rows.append({"acquired_utc": e.get("start_time"), "product_id": e.get("product_id"),
                     "layer": e.get("layer_term"), "valid_pixel_pct": q.get("valid_pixel_pct"),
                     "missing_pct": q.get("missing_pct"), "stride": e.get("stride"),
                     "retrieval_status": e.get("retrieval_status")})
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True)
    st.subheader("Per-pass valid pixel coverage")
    fig = go.Figure(go.Bar(x=frame["acquired_utc"], y=frame["valid_pixel_pct"], marker_color=GREEN))
    fig.update_layout(yaxis_title="Valid pixels (%)", xaxis_title="Acquisition UTC", yaxis_range=[0, 100], height=340)
    mission_chart(fig, use_container_width=True)
    if len(ok) > 1:
        from datetime import datetime
        times = [datetime.fromisoformat(str(e["start_time"]).replace("Z", "+00:00")) for e in ok]
        gaps = [(times[i] - times[i-1]).total_seconds()/86400 for i in range(1, len(times))]
        st.metric("Largest observed acquisition gap", f"{max(gaps):.1f} days")
        st.caption("Actual intervals are shown; missing acquisitions are not interpolated or presented as measurements. The nominal revisit cadence does not guarantee every pass is available here.")
        st.dataframe(pd.DataFrame({"from_utc": frame["acquired_utc"].iloc[:-1].tolist(),
                                   "to_utc": frame["acquired_utc"].iloc[1:].tolist(), "gap_days": gaps}), use_container_width=True)
    try:
        ser = series_from_arrays(arrays)
        st.metric("Pixels valid on every pass", f"{ser['common_valid_pct']:.1f}%")
    except ValueError as ex:
        st.warning(f"Cannot calculate a shared valid-pixel mask: {ex}")
    st.warning("Quality metrics describe sampling and data completeness. They are not statistical confidence probabilities or proof of an environmental event.")
    source_bar(m)

elif page == "EXPORT / PROVENANCE":
    st.title("EXPORT / PROVENANCE - reproducible evidence")
    name, m, ok, arrays = pick_stack("export")
    try:
        ser = series_from_arrays(arrays)
    except ValueError as ex:
        st.error(str(ex)); st.stop()
    rows = []
    for i, e in enumerate(ok):
        rows.append({"acquisition_utc": e["start_time"], "product_id": e["product_id"],
                     "layer_path": e["layer_path"], "units": e.get("units"),
                     "mean_backscatter_db": float(ser["mean_db"][i]),
                     "valid_pixel_pct": e.get("quality", {}).get("valid_pixel_pct"),
                     "stride": e.get("stride"), "retrieval_status": e.get("retrieval_status")})
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True)
    st.download_button("Download measured NISAR time series (CSV)", frame.to_csv(index=False).encode("utf-8"),
                       file_name="earth_pulse_measured_series.csv", mime="text/csv")
    sig = classify_signature(ser["mean_db"], ser["common_valid_pct"])
    report = {"app": "EARTH PULSE", "data_type": "real NASA-ISRO NISAR acquisitions, not simulations",
              "selected_stack": name, "discovery": state(), "stack_metadata": m.get("stack"),
              "common_valid_pixel_pct": ser["common_valid_pct"],
              "signature": {"label": sig["label"], "rule": sig["rule"], "tags": sig["tags"]},
              "evidence": evidence_panel(ok, m["stack"], ok[0]["layer_path"], ok[0].get("units"), ok[0]["stride"]),
              "observations": rows, "interpretation_warning": "Backscatter change does not establish cause."}
    st.download_button("Download evidence report (JSON)", json.dumps(report, indent=2, default=str).encode("utf-8"),
                       file_name="earth_pulse_evidence.json", mime="application/json")
    st.info("Exports contain the observations already retrieved during this session. They do not create new NASA measurements.")
    source_bar(m)

elif page == "PIXEL EXPLORER":
    st.title("PIXEL EXPLORER — inspect a real radar pixel through time")
    st.caption("Select a pixel on the sampled radar grid. Each point is derived from a downloaded NASA acquisition; no missing values are filled.")
    name, m, ok, arrays = pick_stack("px")
    dbs = [to_db(a) for a in arrays]
    if len({d.shape for d in dbs}) != 1:
        st.error("Incompatible raster grids: pixel comparisons are blocked."); st.stop()
    height, width = dbs[0].shape
    c1, c2 = st.columns(2)
    x = c1.number_input("Pixel column (sampled grid)", 0, width-1, min(width//2, width-1), 1)
    y = c2.number_input("Pixel row (sampled grid)", 0, height-1, min(height//2, height-1), 1)
    fig = go.Figure(go.Heatmap(z=dbs[0], colorscale="Viridis", colorbar=dict(title="dB")))
    fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(color="red", size=12, symbol="x"), name="Selected pixel")
    fig.update_yaxes(autorange="reversed"); fig.update_layout(height=380, title="First acquired raster — selected sampled pixel")
    mission_chart(fig, use_container_width=True)
    vals = [float(d[y, x]) if np.isfinite(d[y, x]) else None for d in dbs]
    dates = [e["start_time"] for e in ok]
    fig2 = go.Figure(go.Scatter(x=dates, y=vals, mode="lines+markers", connectgaps=False,
                                line=dict(color=GREEN), name="Measured pixel backscatter"))
    fig2.update_layout(height=350, yaxis_title="Pixel backscatter (dB)", xaxis_title="Actual acquisition UTC")
    mission_chart(fig2, use_container_width=True)
    rows = [{"acquired_utc": date, "product_id": e["product_id"], "pixel_db": v, "valid": v is not None}
            for date, e, v in zip(dates, ok, vals)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.download_button("Export this pixel's measured history", pd.DataFrame(rows).to_csv(index=False),
                       "earth_pulse_pixel_history.csv", "text/csv")
    st.warning("Pixel coordinates are indices in the downloaded, possibly decimated window—not georeferenced coordinates. Radar changes alone do not establish an event or cause.")
    source_bar(m)

elif page == "CHANGE HOTSPOTS":
    st.title("CHANGE HOTSPOTS — rank measured radar differences")
    st.caption("Exploratory ranking of sampled grid cells. A hotspot is not a verified disaster or environmental event.")
    name, m, ok, arrays = pick_stack("hotspots")
    if len(ok) < 2:
        st.info("At least two real acquisitions are required."); st.stop()
    a, b = st.columns(2)
    first = a.selectbox("Reference acquisition", list(range(len(ok)-1)), format_func=lambda i: ok[i]["start_time"][:19])
    last = b.selectbox("Later acquisition", list(range(first+1, len(ok))), format_func=lambda i: ok[i]["start_time"][:19])
    d0, d1 = to_db(arrays[first]), to_db(arrays[last])
    if d0.shape != d1.shape:
        st.error("Raster shapes differ. Cannot compare without co-registration."); st.stop()
    delta = d1-d0
    threshold = st.slider("Minimum absolute difference (dB)", 0.1, 8.0, 1.0, 0.1)
    valid = np.isfinite(delta)
    flagged = valid & (np.abs(delta) >= threshold)
    c = st.columns(3)
    c[0].metric("Valid sampled pixels", int(valid.sum()))
    c[1].metric("Above analyst threshold", int(flagged.sum()))
    c[2].metric("Flagged share of valid pixels", f"{100*flagged.sum()/valid.sum():.1f}%" if valid.any() else "N/A")
    fig = go.Figure(go.Heatmap(z=delta, colorscale="RdBu", zmid=0, colorbar=dict(title="Δ dB")))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(height=460, title="Measured later-minus-earlier radar difference")
    mission_chart(fig, use_container_width=True)
    ys, xs = np.where(flagged)
    order = np.argsort(-np.abs(delta[ys, xs]))[:100]
    rows = [{"sampled_row": int(ys[k]), "sampled_column": int(xs[k]),
             "difference_db": round(float(delta[ys[k], xs[k]]), 4),
             "reference_product": ok[first]["product_id"], "later_product": ok[last]["product_id"]} for k in order]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("Export top 100 sampled change cells", pd.DataFrame(rows).to_csv(index=False),
                           "earth_pulse_change_hotspots.csv", "text/csv")
    else: st.info("No sampled cells exceed the selected threshold.")
    st.warning("The threshold is user-defined, not a validated detection probability. Different acquisition geometry, noise, and surface conditions may affect differences.")
    source_bar(m)

elif page == "MISSION CONTROL":
    st.title("MISSION CONTROL — the real observation heartbeat")
    st.caption("Follow the actual archive timeline and identify gaps without inserting artificial passes.")
    name, m, ok, arrays = pick_stack("mission")
    from datetime import datetime
    dates = [datetime.fromisoformat(str(e["start_time"]).replace("Z", "+00:00")) for e in ok]
    gaps = [(dates[i]-dates[i-1]).total_seconds()/86400 for i in range(1, len(dates))]
    st.metric("Verified passes in this stack", len(ok))
    if gaps:
        c = st.columns(3)
        c[0].metric("Median observed interval", f"{np.median(gaps):.1f} days")
        c[1].metric("Largest gap", f"{max(gaps):.1f} days")
        c[2].metric("Observation span", f"{(dates[-1]-dates[0]).days} days")
        fig = go.Figure(go.Bar(x=[d.isoformat() for d in dates[1:]], y=gaps, marker_color=BLUE))
        fig.update_layout(height=330, xaxis_title="Actual next acquisition", yaxis_title="Days since prior pass")
        mission_chart(fig, use_container_width=True)
    i = st.select_slider("Replay the acquisition sequence", options=list(range(len(ok))),
                         format_func=lambda k: f"Pass {k+1}: {ok[k]['start_time'][:19]}")
    fig = go.Figure(go.Heatmap(z=to_db(arrays[i]), colorscale="Viridis", colorbar=dict(title="dB")))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(height=430, title=f"Verified acquisition {i+1}/{len(ok)}")
    mission_chart(fig, use_container_width=True)
    st.write(f"**NASA product:** `{ok[i]['product_id']}`")
    st.write(f"**Actual UTC acquisition:** {ok[i]['start_time']}")
    st.info("Use the slider to step through actual NASA passes. The nominal 12-day mission cycle is not a promise that this particular stack has a usable image every 12 days.")
    source_bar(m)

elif page == "AI EARTH SCIENTIST":
    from modules.standout_features import observation_report
    st.title("AI EARTH SCIENTIST — evidence-grounded explanations")
    st.caption("Transparent, rules-based scientific assistant. No external language model is used; it cannot invent satellite measurements or assert causes.")
    name, m, ok, arrays = pick_stack("scientist")
    threshold = st.slider("Analyst change threshold (dB)", 0.1, 3.0, 1.0, 0.1, key="scientist_threshold")
    try:
        report = observation_report(ok, arrays, threshold)
    except (ValueError, KeyError) as ex:
        st.error(f"Unable to interpret this observation stack: {ex}"); st.stop()
    st.subheader("What the NASA observations show")
    st.write(report["story"])
    if beginner:
        st.success("**In simple words:** These readings show how the radar signal from this place changed over the available dates. A change does not tell us the exact cause. See the dates and chart below for the evidence.")
    st.write(f"**Observed pattern:** {report['signature']}")
    st.caption("Rule: " + report["signature_rule"])
    if beginner:
        st.caption("Each dot on the graph comes from an actual NASA observation. Higher or lower does not always mean better or worse.")
    st.subheader("Evidence behind the explanation")
    df = pd.DataFrame(report["observations"])
    st.line_chart(df.set_index("acquired_utc")["mean_db"])
    st.dataframe(df, use_container_width=True)
    with st.expander("What this does NOT tell us"):
        for item in report["limitations"]: st.write("- " + item)
    st.download_button("Download reproducible evidence report (JSON)", json.dumps(report, indent=2).encode(),
                       file_name="earth_pulse_scientist_evidence.json", mime="application/json")
    source_bar(m)

elif page == "3D EARTH EXPLORER":
    from modules.standout_features import center_from_bbox
    st.title("3D EARTH EXPLORER — actual observed regions")
    st.caption("Interactive globe-style geographic map. Pins are drawn only for loaded stacks with valid geographic coordinates; no example locations.")
    points = []
    for name in list_stacks():
        m, ok, arrays, skipped = load(name)
        if not ok: continue
        pos = center_from_bbox(m.get("bbox"))
        if pos is None: continue
        points.append({"lat": pos[0], "lon": pos[1], "stack": name,
                       "observations": len(ok), "first": ok[0]["start_time"], "last": ok[-1]["start_time"]})
    if not points:
        st.info("No georeferenced loaded NISAR stacks yet. Stream observations in LIVE DATA first. Stacks without valid bounding boxes are excluded.")
    else:
        import pydeck as pdk
        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=points[0]["lat"], longitude=points[0]["lon"], zoom=2, pitch=35),
            layers=[pdk.Layer("ScatterplotLayer", data=points, get_position="[lon, lat]",
                              get_radius=25000, radius_min_pixels=5, radius_max_pixels=20,
                              get_fill_color="[47, 191, 143, 190]", pickable=True)],
            tooltip={"html": "<b>{stack}</b><br/>Verified passes: {observations}<br/>{first} — {last}"}))
        st.dataframe(pd.DataFrame(points), use_container_width=True)
    st.info("The basemap is contextual cartography, not NASA radar imagery. Only observation pins and their metadata come from loaded NASA products. This is a tilted 3D map, not a full spherical globe.")
    source_bar()

elif page == "ENVIRONMENTAL WATCHLIST":
    from modules.standout_features import new_products
    st.title("ENVIRONMENTAL WATCHLIST — track real acquisitions")
    st.caption("Manual, session-only monitoring of NASA product IDs. No automated background alerts, and no claim of real-time disaster detection.")
    if "watchlist" not in st.session_state: st.session_state["watchlist"] = {}
    watch = st.session_state["watchlist"]
    names = list_stacks()
    if not names:
        st.info("First stream a verified stack using LIVE DATA, then add it to your watchlist.")
    else:
        chosen = st.selectbox("Choose a retrieved NISAR stack", names, key="watch_pick")
        if st.button("Add / refresh watched stack"):
            m, ok, _, _ = load(chosen)
            ids = [e["product_id"] for e in ok]
            prior = watch.get(chosen, {}).get("product_ids", [])
            additions = new_products(prior, ids)
            watch[chosen] = {"product_ids": ids, "latest_acquisition": max(e["start_time"] for e in ok),
                             "checked_utc": pd.Timestamp.now(tz="UTC").isoformat(), "new_ids": additions,
                             "baseline_initialized": chosen not in watch}
            if prior: st.success(f"{len(additions)} new product(s) in the currently loaded stack since the last check.")
            else: st.success("Watchlist baseline saved for this session. Refresh LIVE DATA before checking for newer acquisitions.")
    for name, item in list(watch.items()):
        with st.expander(name, expanded=True):
            st.write(f"**Last observed acquisition:** {item['latest_acquisition']}")
            st.write(f"**Last checked (UTC):** {item['checked_utc']}")
            st.write(f"**Products in snapshot:** {len(item['product_ids'])}")
            if not item['baseline_initialized']:
                st.write(f"**New IDs since previous check:** {len(item['new_ids'])}")
                if item['new_ids']: st.code("\n".join(item['new_ids']))
            if st.button("Remove from watchlist", key="remove_"+name):
                del watch[name]; st.rerun()
    if watch:
        st.download_button("Export watchlist (JSON)", json.dumps(watch, indent=2).encode(),
                           file_name="earth_pulse_watchlist.json", mime="application/json")
    st.warning("Watchlist changes reflect only newly streamed verified products. Search and stream again in LIVE DATA to check NASA for updates. Session state is lost when the app session ends.")
    source_bar()

elif page == "CAUSE EXPLORER":
    from datetime import date
    from modules.cause_explorer import investigate
    st.title("Cause Explorer")
    st.write("What might explain a change? Compare it with real environmental records. This is an investigation, not a confirmed diagnosis.")
    context = st.session_state.get("active_location")
    if not context:
        st.warning("Start by choosing a location and dates in Get NASA images. Cause Explorer will automatically use that same place and period.")
        st.caption("You never need to type the location again.")
        if st.button("Go to Get NASA images", type="primary"):
            st.session_state["navigation_page"] = "Get NASA images"
            st.rerun()
        st.stop()

    st.subheader("Your selected investigation")
    st.write(f"**Place:** {context['label']}")
    st.write(f"**Period:** {context['start']} to {context['end']}")
    st.caption("These are the exact location and dates from your initial NASA archive search. Cause Explorer uses them automatically.")
    if not st.session_state.get("discovery"):
        st.info("The NASA archive has not yet returned results for this search. Cause Explorer can still check independent environmental records, but cannot link them to a confirmed radar change.")
    elif not list_stacks():
        st.warning("No NISAR images have been loaded yet. You may investigate environmental context, but this is not a confirmed explanation of a satellite change.")
    else:
        st.success("NASA observations are loaded. Compare their change dates with the environmental records below.")
    with st.expander("Advanced: search distance"):
        radius = st.slider("Search distance around selected place (km)", 25, 250, 100, 25)
    if st.button("Investigate possible causes", type="primary"):
        try:
            from datetime import date as _date
            start = _date.fromisoformat(context["start"])
            end = _date.fromisoformat(context["end"])
            with st.spinner("Checking real NASA and USGS records across your full selected period..."):
                st.session_state['cause_result'] = investigate(context['lat'], context['lon'], start, end, radius)
            st.session_state['cause_place'] = context['label']
            st.session_state['cause_context_key'] = (context['lon'], context['lat'], context['start'], context['end'])
        except Exception as ex:
            st.error(f"Investigation could not finish: {ex}")
    if st.session_state.get('cause_context_key') != (context['lon'], context['lat'], context['start'], context['end']):
        st.session_state.pop('cause_result', None)
    result = st.session_state.get('cause_result')
    if result:
        st.subheader("Evidence near " + st.session_state.get('cause_place', 'selected place'))
        st.caption("These are independent supporting datasets. A match in time and location does not establish cause and effect.")
        rain = result.get('rainfall')
        if rain:
            a,b,c = st.columns(3)
            a.metric("Rainfall over selected period", f"{rain['total_mm']:,.1f} mm")
            b.metric("Highest estimated daily rainfall", f"{rain['maximum_daily_mm']:,.1f} mm")
            c.metric("Days with rainfall estimates", rain['days_available'])
            st.caption("NASA POWER precipitation is a gridded estimate, not a NISAR measurement or proof of flooding.")
            if rain['daily']:
                df = pd.DataFrame(rain['daily'])
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                st.line_chart(df.set_index('date')['mm'], y_label="Estimated rainfall (mm/day)")
        else:
            st.info("Rainfall data was not available. No rainfall values have been invented.")
        st.subheader("Earthquake records")
        if result['earthquakes']:
            st.dataframe(pd.DataFrame(result['earthquakes'])[['magnitude','place','url']], hide_index=True, use_container_width=True)
        else:
            st.write("No matching USGS earthquake records were returned for this search. This does not rule out ground movement.")
        st.subheader("NASA recorded natural events")
        if result['nasa_events']:
            for event in result['nasa_events'][:30]:
                st.write(f"**{event['title']}** — {', '.join(str(x) for x in event['categories'])}")
                if event.get('url'): st.link_button("View original NASA event", event['url'])
        else:
            st.write("No nearby point-based NASA EONET events were returned. The catalog is not a complete record of all disasters.")
        st.subheader("What can we conclude?")
        st.warning("Possible context only. Rainfall, earthquake records, and reported natural events can help form hypotheses, but none independently proves what caused a NISAR radar change. Check timing, location, measurement quality, and independent imagery before making a causal claim.")
        if result['errors']:
            for error in result['errors']: st.warning(error)
        st.markdown("**Original evidence sources**")
        for source in result['sources']:
            st.markdown(f"- [{source['name']}]({source['url']})")
        st.download_button("Download evidence report (JSON)", data=json.dumps(result, indent=2), file_name="earth_pulse_cause_evidence.json", mime="application/json")

else:
    st.title("ABOUT / DATA SOURCES")
    st.markdown("""
**Data:** NASA-ISRO NISAR L-band SAR, GCOV products, discovered with the ASF Search API (`asf_search`) and
downloaded with NASA Earthdata authentication. Sources: NASA Earthdata Search, ASF Data Search, NISAR Data User Guide
(nisar-docs.asf.alaska.edu).

**Method:** mean backscatter in dB over pixels valid in every date of a compatible stack (same track, frame,
direction, polarization, maturity and release). Labels use transparent threshold rules (below).

**Limitations:** backscatter alone cannot establish cause; sampling is decimated; thresholds are analyst
parameters, not confidence scores; unverified items are labelled in the evidence panel.
""")
    st.code(RULES_TEXT)
    s = state()
    if s: st.json(s)
    source_bar()
