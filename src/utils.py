import numpy as np
import pandas as pd

# Helps transform lon and lat points into distances
def distance(lat1, lon1, lat2, lon2):

    # NOTE: this function is using code from the stackoverflow answer in the link below:
    # "http://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206"

    from numpy import cos, asin, sqrt, pi

    r = 6371  # Earth radius in kilometers 
    p = np.pi / 180

    # Convert latitude/longitude differences and values to radians
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p

    a = (
        0.5
        - np.cos(dlat) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos(dlon)) / 2
    )

    dist = 2 * r * np.arcsin(np.sqrt(a))

    # fill in any NaN values with 0 (this is for the first game of the season)

    dist_final = np.nan_to_num(dist, nan=0.0)

    return dist_final