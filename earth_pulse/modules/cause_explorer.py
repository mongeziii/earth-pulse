"""Evidence-only environmental context. Never treats correlation as proof of causation."""
from datetime import date, timedelta
import requests

POWER = 'https://power.larc.nasa.gov/api/temporal/daily/point'
EONET = 'https://eonet.gsfc.nasa.gov/api/v3/events'
USGS = 'https://earthquake.usgs.gov/fdsnws/event/1/query'

def _get(url, params):
    response = requests.get(url, params=params, timeout=22, headers={'User-Agent': 'EarthPulse/1.0 (educational research)'})
    response.raise_for_status()
    return response.json(), response.url

def investigate(lat, lon, start, end, radius_km=100):
    """Return independently sourced observations, plus explicit failures and caveats."""
    start, end = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    if end < start: raise ValueError('End date must be after start date.')
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): raise ValueError('Invalid location.')
    # Long investigations are divided into provider-sized requests; no app-wide 365-day limit.
    result = {'location': {'latitude': lat, 'longitude': lon}, 'period': [start.isoformat(), end.isoformat()],
              'sources': [], 'errors': [], 'earthquakes': [], 'nasa_events': [], 'rainfall': None}
    def periods(first, last, days):
        current = first
        while current <= last:
            finish = min(last, current + timedelta(days=days - 1))
            yield current, finish
            current = finish + timedelta(days=1)
    # NASA POWER precipitation is a gridded analysis product, NOT a direct NISAR observation.
    rain_by_day = {}
    for a, b in periods(start, end, 300):
        try:
            data, url = _get(POWER, {'parameters':'PRECTOTCORR','community':'RE','longitude':lon,
                                    'latitude':lat,'start':a.strftime('%Y%m%d'),'end':b.strftime('%Y%m%d'),
                                    'format':'JSON'})
            vals = data['properties']['parameter']['PRECTOTCORR']
            for k, v in vals.items():
                if v is not None and float(v) >= 0:
                    rain_by_day[k] = float(v)
            result['sources'].append({'name':f'NASA POWER precipitation ({a} to {b})','url':url})
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            result['errors'].append(f'NASA POWER rainfall unavailable for {a} to {b}: {str(exc)[:140]}')
    if rain_by_day:
        daily = [{'date':k,'mm':v} for k,v in sorted(rain_by_day.items())]
        result['rainfall'] = {'daily':daily,'total_mm':round(sum(rain_by_day.values()),1),
                              'maximum_daily_mm':round(max(rain_by_day.values()),1),
                              'days_available':len(daily),'product':'NASA POWER PRECTOTCORR (gridded precipitation estimate)'}
    # Use bounded USGS requests and deduplicate event identifiers across date windows.
    quake_by_id = {}
    for a, b in periods(start, end, 300):
        try:
            data, url = _get(USGS, {'format':'geojson','starttime':a.isoformat(),
                                    'endtime':(b+timedelta(days=1)).isoformat(),
                                    'latitude':lat,'longitude':lon,'maxradiuskm':radius_km,
                                    'minmagnitude':2.5,'orderby':'magnitude','limit':200})
            for x in data.get('features', []):
                props = x.get('properties') or {}
                key = x.get('id') or props.get('url')
                if key:
                    quake_by_id[key] = {'time_utc_ms':props.get('time'), 'magnitude':props.get('mag'),
                                       'place':props.get('place'), 'url':props.get('url')}
            result['sources'].append({'name':f'USGS Earthquake Catalog ({a} to {b})','url':url})
            if len(data.get('features', [])) >= 200:
                result['errors'].append(f'USGS returned the 200-record window limit for {a} to {b}; results may be incomplete.')
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            result['errors'].append(f'USGS earthquake records unavailable for {a} to {b}: {str(exc)[:140]}')
    result['earthquakes'] = sorted(quake_by_id.values(), key=lambda x: x.get('time_utc_ms') or 0)
    from math import radians, sin, cos, asin, sqrt
    def distance(a,b,c,d):
        q = sin(radians(c-a)/2)**2 + cos(radians(a))*cos(radians(c))*sin(radians(d-b)/2)**2
        return 12742*asin(min(1,sqrt(q)))
    event_by_id = {}
    for a, b in periods(start, end, 90):
        try:
            data, url = _get(EONET, {'status':'all','start':a.isoformat(),'end':b.isoformat(),'limit':500})
            for event in data.get('events', []):
                near = False
                for geo in event.get('geometry', []):
                    if geo.get('type') == 'Point' and isinstance(geo.get('coordinates'),list) and len(geo['coordinates']) >= 2:
                        x,y = geo['coordinates'][:2]
                        if distance(lat,lon,y,x) <= radius_km:
                            near = True
                if near:
                    key = event.get('id') or event.get('link')
                    if key:
                        event_by_id[key] = {'title':event.get('title'),
                            'categories':[x.get('title') for x in event.get('categories',[])],
                            'url':event.get('link'), 'id':event.get('id')}
            result['sources'].append({'name':f'NASA EONET events ({a} to {b})','url':url})
            if len(data.get('events', [])) >= 500:
                result['errors'].append(f'NASA EONET reached its 500-event window limit for {a} to {b}; results may be incomplete.')
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            result['errors'].append(f'NASA EONET records unavailable for {a} to {b}: {str(exc)[:140]}')
    result['nasa_events'] = list(event_by_id.values())
    return result
