from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

geolocator = Nominatim(user_agent="object-detection-api")

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'address' in location.raw:
            address = location.raw['address']
            return {
                "country": address.get("country", "undefined"),
                "state": address.get("state", "undefined"),
                "city": address.get("city", address.get("town", address.get("village", "undefined"))),
                "road": address.get("road", "undefined")
            }
        return None
    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        print(f"Geocoding error: {e}")
        return None
