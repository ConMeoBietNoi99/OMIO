"""Download the Sioux Falls data (Transportation Networks for Research) into ./data."""
import os, urllib.request
BASE = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/"
os.makedirs("data", exist_ok=True)
for f in ("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp", "SiouxFalls_node.tntp"):
    urllib.request.urlretrieve(BASE + f, os.path.join("data", f)); print("downloaded", f)
