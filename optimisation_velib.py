import json
import numpy as np
from scipy.spatial import Delaunay, Voronoi
import networkx as nx
import statistics
import folium
import webbrowser

# Charger le fichier JSON
with open('station_information.json', 'r') as file:
    json_data = json.load(file)

# Créer une carte Folium centrée sur Paris
m = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

# Dictionnaire des stations
stations_dict = {}
for station in json_data["data"]["stations"]:
    station_id = station["station_id"]
    station_name = station["name"]
    coordonnees = (station["lat"], station["lon"])
    stations_dict[station_id] = (station_name, coordonnees)

# Tableaux des noms et des coordonnées des stations
place_names = np.array([name for name, _ in stations_dict.values()])
points = np.array([coords for _, coords in stations_dict.values()])

# Triangulation de Delaunay
tri = Delaunay(points)
G = nx.Graph()
for simplex in tri.simplices:
    for i in range(3):
        p1, p2 = tuple(points[simplex[i]]), tuple(points[simplex[(i+1) % 3]])
        weight = np.linalg.norm(np.array(p1) - np.array(p2))
        G.add_edge(p1, p2, weight=weight)

# Couche pour la triangulation de Delaunay
delaunay_layer = folium.FeatureGroup(name="Triangulation de Delaunay")
m.add_child(delaunay_layer)
for u, v in G.edges():
    line = folium.PolyLine(locations=[(u[0], u[1]), (v[0], v[1])], color='black', weight=2, opacity=0.8)
    line.add_to(delaunay_layer)

# Calcul de l'arbre couvrant minimal avec Kruskal
mst = nx.minimum_spanning_tree(G, algorithm="kruskal")

# Couche pour l'arbre couvrant minimal
kruskal_layer = folium.FeatureGroup(name="Arbre couvrant minimal (Kruskal)")
m.add_child(kruskal_layer)
for u, v in mst.edges():
    line = folium.PolyLine(locations=[(u[0], u[1]), (v[0], v[1])], color='purple', weight=3, opacity=0.8)
    line.add_to(kruskal_layer)

# Calcul des indices de qualité de la répartition
donnee = json_data['data']['stations']
final = {}
cap = []
adj = {}

for i in donnee:
    id = i['station_id']
    final[id] = [i['lat'], i['lon']]

a = final.values()
a = list(a)
points = np.array(a)
tri = Delaunay(points)

for j in range(len(final)):
    adj[j] = []
    for i in tri.simplices:
        if j in i:
            for k in i:
                if k not in adj[j] and k != j:
                    adj[j].append(k)

for i in donnee:
    cap.append(i['capacity'])

alpha = 0.5
indice = []
max_cap = max(cap)

for i in range(len(cap)):
    if len(adj[i]) == 6:
        indice.append(0)
    else:
        I = (alpha * (len(adj[i]) - 6)) / 6 + ((1 + alpha) * (max_cap - cap[i])) / max_cap
        indice.append(I)

quartiles = statistics.quantiles(indice, n=4)

# Couche pour les stations avec des couleurs basées sur l'indice
stations_layer = folium.FeatureGroup(name="Stations (Indice de qualité)")
m.add_child(stations_layer)
for i in range(len(points)):
    if indice[i] > 0.8333333333333334:
        color = "darkgreen"
    elif indice[i] > 0:
        color = "lightgreen"
    elif indice[i] < -0.021929824561403508:
        color = "#FF6666"
    elif indice[i] == 0:
        color = "yellow"
    else:
        color = "#8B0000"

    folium.CircleMarker(
        location=points[i],
        radius=2,
        color=color,
        fill=True,
        fill_color="cyan",
        fill_opacity=0.7,
        tooltip=place_names[i]
    ).add_to(stations_layer)

# Fonction pour fermer les zones Voronoï
def voronoi_finite_polygons_2d(vor, radius=0.05):
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            far_point = vor.vertices[v2] + n * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

def sort_polygon(polygon):
    center = np.mean(polygon, axis=0)
    return sorted(polygon, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

# Diagramme de Voronoï
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# Couche pour le diagramme de Voronoï
voronoi_layer = folium.FeatureGroup(name="Diagramme de Voronoï")
m.add_child(voronoi_layer)
for i, region in enumerate(regions):
    if len(region) > 0:
        polygon = [vertices[j] for j in region]
        polygon = sort_polygon(polygon)
        if i < len(indice):
            if indice[i] > 0.2:
                color = "blue"
            elif indice[i] < -0.02:
                color = "red"
            else:
                color = "#66ff66"
        else:
            color = "gray"
        folium.Polygon(
            locations=[(p[0], p[1]) for p in polygon],
            color="black",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(voronoi_layer)

# Ajouter un contrôle des couches
folium.LayerControl().add_to(m)

# Sauvegarder et afficher la carte
m.save("combined_map_with_layers.html")
webbrowser.open("combined_map_with_layers.html")
