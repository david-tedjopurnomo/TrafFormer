from argparse import ArgumentParser
from pathlib import Path
from shapely.geometry import Polygon
import configparser
import os
import osmnx as ox
import time

# Read the ini file path argument 
parser = ArgumentParser(description='inputs')
parser.add_argument('--config', dest = 'config',
                    help='The path to the .ini config file. FORMAT: ' + 
                         'a string.')
ini_path = parser.parse_args().config

config = configparser.ConfigParser()
config.read(ini_path)
        
# Read the arguments
out_path = config['OUTPUT']['RoadNetworkOutputPath']


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = gdf_edges.index
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

print("osmnx version",ox.__version__)
print("Please use osmnx version 0.15.0 for compatibility with the FMM used in the mapmatching script")

# Download by a place
#place = "Victoria, Australia"
#print("DOWNLOADING FROM PLACE: %s" % place)
#G = ox.graph_from_place(place, network_type='drive_service')

# Download by a bounding box
#bounds = (144.55, 145.65,-38.3,-37.4)
#bounds = (144.55, 144.7,-38.16,-38.1)
#bounds = (145.65,144.55,-37.4,-38.3)
bounds = (-121.84,-122.08,37.43,37.26)
print("DOWNLOADING FROM BOUNDING BOX: %s" % (str(bounds)))
x1,x2,y1,y2 = bounds
boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
start_time = time.time()
G = ox.graph_from_polygon(boundary_polygon, network_type='drive_service')
save_graph_shapefile_directional(G, filepath=out_path)
print("--- %s seconds ---" % (time.time() - start_time))
