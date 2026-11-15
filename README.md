Here’s a much shorter README that gets straight to the point while keeping the essentials.

Writing

# EARTH PULSE

### A satellite image shows a moment. Earth Pulse helps us understand the process.

NASA Space Apps Challenge 2026 — Dancing with the SARs

Earth Pulse uses real NASA-ISRO NISAR satellite observations to explore, investigate, compare, and forecast changes on Earth's surface.

All observations come directly from the official ASF Search API. No synthetic or preloaded satellite data is used.

## Features

- LIVE DATA: Search and stream real NISAR observations.
- WATCH: Explore surface changes over time.
- INVESTIGATE: Identify and explain change patterns.
- COMPARE: Compare compatible satellite observations.
- FORECAST: Estimate future backscatter measurements using machine learning.
- AI EARTH SCIENTIST: Generate evidence-based reports.
- 3D EARTH EXPLORER: Explore observation locations.
- ENVIRONMENTAL WATCHLIST: Track satellite product updates.

## Installation

bash

```bash
git clone <repository-url>
cd earth-pulse
pip install -r requirements.txt
```

Set your NASA Earthdata token:

bash

```bash
# Windows
set EARTHDATA_TOKEN=your-token

# macOS / Linux
export EARTHDATA_TOKEN="your-token"
```

Get a token at [https://urs.earthdata.nasa.gov/](<https://urs.earthdata.nasa.gov/>)

Launch the application:

bash

```bash
streamlit run app.py
```

## How It Works

1. Select an area of interest.
2. Search the NASA archive through ASF.
3. Stream real satellite observations.
4. Explore changes, compare data, and generate forecasts.

The live application processes observations in memory without saving satellite data to your computer.

## Development Status

- Phase 1: Data discovery — complete.
- Phase 2: Observation intelligence — complete.
- Phase 3: Forecasting — complete.

Offline tests have passed, but live NASA data retrieval, real-product analysis, and full UI functionality still require verification.

## Our Principle

Real satellite data. Transparent analysis. Evidence-based insights.

Earth Pulse does not invent observations, claim unverified results, or determine environmental causes from radar data alone.
