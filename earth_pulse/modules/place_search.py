"""Resolve user-entered place names to real OpenStreetMap Nominatim locations.
This service only resolves geography; all satellite observations still come from NASA ASF.
"""
import requests

URL = 'https://nominatim.openstreetmap.org/search'


def find_places(query, limit=5):
    query = query.strip()
    if len(query) < 3:
        raise ValueError('Enter a town, city, region, or landmark (at least 3 characters).')
    response = requests.get(URL, params={'q': query, 'format': 'jsonv2', 'limit': limit,
                                          'addressdetails': 1},
                            headers={'User-Agent': 'EarthPulseEducationalApp/1.0 (NASA satellite research)'}, timeout=18)
    response.raise_for_status()
    places = []
    for item in response.json():
        lon, lat = float(item['lon']), float(item['lat'])
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            places.append({'name': item['display_name'], 'lon': lon, 'lat': lat,
                           'type': item.get('type', 'place'), 'osm_id': item.get('osm_id')})
    return places
