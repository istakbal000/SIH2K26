import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import shapely.wkb
import os

def calculate_osm_features(latitudes, longitudes, osm_pbf_path=None):
    """
    Efficient spatial processing of OSM data using Osmium and GeoPandas.
    Fallback from pyrosm since osmium installs on Windows without C++ build tools.
    """
    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        print("WARNING: No valid OSM PBF file provided. Creating empty OSM features.")
        return pd.DataFrame(index=range(len(latitudes)))
        
    print(f"Loading OSM data from {osm_pbf_path} (This might take a while)...")
    
    features = pd.DataFrame(index=range(len(latitudes)))
    
    try:
        # pyrefly: ignore [missing-import]
        import osmium
        
        # WKB Factory for creating Shapely geometries
        wkbfab = osmium.geom.WKBFactory()
        
        class ForestHandler(osmium.SimpleHandler):
            def __init__(self):
                super(ForestHandler, self).__init__()
                self.forest_polygons = []
                
            def area(self, a):
                if ('natural' in a.tags and a.tags['natural'] == 'wood') or \
                   ('landuse' in a.tags and a.tags['landuse'] == 'forest'):
                    try:
                        wkb = wkbfab.create_multipolygon(a)
                        poly = shapely.wkb.loads(wkb, hex=True)
                        self.forest_polygons.append(poly)
                    except Exception:
                        pass
                        
        print("Extracting natural=wood and landuse=forest...")
        handler = ForestHandler()
        handler.apply_file(osm_pbf_path)
        
        # Create GeoDataFrame for forests
        if handler.forest_polygons:
            forests_gdf = gpd.GeoDataFrame(geometry=handler.forest_polygons, crs="EPSG:4326")
            
            # Create GeoDataFrame for our points
            geometry = [Point(xy) for xy in zip(longitudes, latitudes)]
            points_gdf = gpd.GeoDataFrame(features, geometry=geometry, crs="EPSG:4326")
            
            # Project to a metric CRS for accurate distance calculation (e.g. Web Mercator)
            points_metric = points_gdf.to_crs(epsg=3857)
            forests_metric = forests_gdf.to_crs(epsg=3857)
            
            print("Calculating distances to nearest forest...")
            # Using sjoin_nearest to find distance
            nearest = gpd.sjoin_nearest(points_metric, forests_metric, how='left', distance_col='distance_to_forest')
            
            # Since a point might match multiple equidistant features, we drop duplicates
            nearest = nearest[~nearest.index.duplicated(keep='first')]
            
            features['distance_to_forest'] = nearest['distance_to_forest'].values
        else:
            print("No forest polygons found in the dataset.")
            features['distance_to_forest'] = float('nan')
            
    except ImportError:
        print("ERROR: osmium is not installed. Run `pip install osmium`.")
        features['distance_to_forest'] = float('nan')
    except Exception as e:
        print(f"Error processing OSM data: {e}")
        features['distance_to_forest'] = float('nan')
        
    features['distance_to_industrial_area'] = float('nan')
    features['forest_area_ratio_1km'] = float('nan')
    features['industrial_area_ratio_1km'] = float('nan')
    
    return features
