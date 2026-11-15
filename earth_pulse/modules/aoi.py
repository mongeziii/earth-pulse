"""Areas of interest: WKT builders and a coarse global sampling grid."""


def point_wkt(lon, lat):
    return f"POINT({lon} {lat})"


def bbox_wkt(min_lon, min_lat, max_lon, max_lat):
    return (f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
            f"{min_lon} {max_lat},{min_lon} {min_lat}))")


def parse_aoi(text):
    """'lon,lat' -> point; 'minlon,minlat,maxlon,maxlat' -> box. Returns (label, wkt)."""
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        raise ValueError(f"Cannot parse '{text}': use lon,lat or minlon,minlat,maxlon,maxlat")
    if len(v) == 2:
        lon, lat = v
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("lon must be -180..180 and lat -90..90 (order is lon,lat)")
        return f"point({lon},{lat})", point_wkt(lon, lat)
    if len(v) == 4:
        a, b, c, d = v
        if not (-180 <= a < c <= 180 and -90 <= b < d <= 90):
            raise ValueError("bbox must be minlon,minlat,maxlon,maxlat with min < max")
        return f"bbox({a},{b},{c},{d})", bbox_wkt(a, b, c, d)
    raise ValueError("expected 2 numbers (lon,lat) or 4 numbers (bbox)")


def grid_points(step_deg):
    """Cell-centre points of a global grid. NOTE: a SAMPLE of the globe, not an exhaustive search."""
    if not 5 <= step_deg <= 90:
        raise ValueError("grid step must be between 5 and 90 degrees")
    n_lat, n_lon = int(180 // step_deg), int(360 // step_deg)
    return [(-90 + step_deg * (i + 0.5), -180 + step_deg * (j + 0.5))
            for i in range(n_lat) for j in range(n_lon)]   # (lat, lon)


def center_lonlat(text):
    """Point to read around: the point itself, or the centre of a box. Validates like parse_aoi."""
    parse_aoi(text)
    v = [float(x) for x in text.split(",")]
    return (v[0], v[1]) if len(v) == 2 else ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)
