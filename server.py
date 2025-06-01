import os
import flask
import numpy as np
import time
from PIL import Image

from tqdm import tqdm

import torch

from sklearn.cluster import KMeans

from flask import Flask
from flask_cors import CORS
import math

from matplotlib import pyplot as plt

import umap

# create Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# TODO load all of the data generated from preprocessing
STATIC_DIR    = 'static'
IOU_DIR       = os.path.join(STATIC_DIR, 'ious')
UMAP_FILE     = os.path.join(STATIC_DIR, 'umap.npy')
X23_FILE      = os.path.join(IOU_DIR, 'X23.npy')
X34_FILE      = os.path.join(IOU_DIR, 'X34.npy')
# number of clusters - feel free to adjust
n_clusters = 9

# these variables will contain the clustering of channels for the different layers
a2_clustering,a3_clustering,a4_clustering = None,None,None
X23 = None                   
X34 = None                   
projections = None           

clusterCorr23 = None         
clusterCorr34 = None         
counts2 = None                
counts3 = None                
counts4 = None               
'''
Do not cache images on browser, see: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
'''
@app.after_request
def add_header(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
#

def spectral_clustering(affinity_mat, n_clusters):
    model = SpectralBiclustering(n_clusters=n_clusters, method='bistochastic', random_state=42)
    model.fit(affinity_mat.astype(np.float64))
    return model.row_labels_.astype(int)

def multiway_spectral_clustering(sim23, sim34, n_clusters):
    S2 = sim23.dot(sim23.T)
    a2 = spectral_cluster_affinity(S2, n_clusters)

    # 2) Para layer4: S4 = sim34.T @ sim34   
    S4 = sim34.T.dot(sim34)
    a4 = spectral_cluster_affinity(S4, n_clusters)

    # 3) Para layer3: combinamos sim23^T @ sim23 + sim34 @ sim34^T
    S3a = sim23.T.dot(sim23)    
    S3b = sim34.dot(sim34.T)     
    S3  = S3a + S3b
    a3  = spectral_cluster_affinity(S3, n_clusters)

    return a2, a3, a4
    


def aggregate_cluster_corr(sim_tensor, clustering_src, clustering_tgt, n_clusters):

    n, C_src, C_tgt = sim_tensor.shape
    cc = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    counts_src = np.bincount(clustering_src, minlength=n_clusters)
    counts_tgt = np.bincount(clustering_tgt, minlength=n_clusters)

    for i in range(n_clusters):
        idx_src = np.where(clustering_src == i)[0]
        for j in range(n_clusters):
            idx_tgt = np.where(clustering_tgt == j)[0]
            if len(idx_src) == 0 or len(idx_tgt) == 0:
                cc[i, j] = 0.0
                continue
            # extraemos sub-bloque de sim_tensor[:, idx_src, idx_tgt]
            block = sim_tensor[:, :, :][:, idx_src[:, None], idx_tgt[None, :]]  # shape: (n, |idx_src|, |idx_tgt|)
            s = block.sum()  # suma total
            norm = float(n * len(idx_src) * len(idx_tgt))
            cc[i, j] = s / norm if norm > 0 else 0.0

    return cc, counts_src, counts_tgt


@app.route('/activation_correlation_clustering', methods=['GET'])
def activation_correlation_clustering():

    global n_clusters, counts2, counts3, counts4, clusterCorr23, clusterCorr34

    resp = {
        "K": int(n_clusters),
        "counts": {
            "2": counts2.tolist(),
            "3": counts3.tolist(),
            "4": counts4.tolist()
        },
        "cluster_corr23": clusterCorr23.tolist(),
        "cluster_corr34": clusterCorr34.tolist()
    }
    return jsonify(resp)


@app.route('/link_score', methods=['GET', 'POST'])
def link_score():

    layer_pair = request.args.get('layer_pair', None)
    c1 = request.args.get('c1', None)
    c2 = request.args.get('c2', None)

    if layer_pair not in ('23', '34'):
        return "Error: layer_pair debe ser '23' o '34'", 400
    try:
        c1 = int(c1)
        c2 = int(c2)
    except:
        return "Error: c1 y c2 deben ser enteros válidos", 400

    global X23, X34
    if layer_pair == '23':
        if c1 < 0 or c1 >= X23.shape[1] or c2 < 0 or c2 >= X23.shape[2]:
            return "Error: índices c1 o c2 fuera de rango para X23", 400
        arr = X23[:, c1, c2]  
    else:
        if c1 < 0 or c1 >= X34.shape[1] or c2 < 0 or c2 >= X34.shape[2]:
            return "Error: índices c1 o c2 fuera de rango para X34", 400
        arr = X34[:, c1, c2]

    return jsonify(arr.tolist())


@app.route('/channel_dr', methods=['GET', 'POST'])
def channel_dr():
    """
    Retorna las proyecciones en 2D para cada muestra (UMAP). JSON:
      { "projections": [[x0,y0], [x1,y1], …] }
    """
    global projections
    if projections is None:
        return "Error: Proyecciones UMAP no disponibles", 500
    return jsonify({"projections": projections.tolist()})


@app.route('/selected_correlation', methods=['GET', 'POST'])
def selected_correlation():
    """
    Dado un arreglo de índices de instancias seleccionadas (brushed) en el request args:
      ids = [i1, i2, …]
    Retorna dos matrices JSON de tamaño K×K:
    {
      "cluster_corr23_brushed": [[…] K×K],
      "cluster_corr34_brushed": [[…] K×K]
    }
    Si ids está vacío o no es válido, retorna ceros.
    """
    ids_str = request.args.get('ids', '[]')
    try:
        ids = flask.json.loads(ids_str)
    except:
        return "Error: ids debe ser un array JSON de enteros", 400

    if not isinstance(ids, list):
        return "Error: ids debe ser un array JSON de enteros", 400

    ids = [int(i) for i in ids if isinstance(i, (int, float))]
    if len(ids) == 0:
        zero23 = np.zeros((n_clusters, n_clusters), dtype=float).tolist()
        zero34 = np.zeros((n_clusters, n_clusters), dtype=float).tolist()
        return jsonify({
            "cluster_corr23_brushed": zero23,
            "cluster_corr34_brushed": zero34
        })

    global X23, X34, a2_clustering, a3_clustering, a4_clustering

 
    subX23 = X23[ids, :, :]  
    subX34 = X34[ids, :, :]  

    C23b, _, _ = aggregate_cluster_corr(subX23, a2_clustering, a3_clustering, n_clusters)
    C34b, _, _ = aggregate_cluster_corr(subX34, a3_clustering, a4_clustering, n_clusters)

    return jsonify({
        "cluster_corr23_brushed": C23b.tolist(),
        "cluster_corr34_brushed": C34b.tolist()
    })


if __name__ == '__main__':
    print("=== Iniciando servidor Flask y cargando datos preprocesados… ===")

    print("  • Cargando X23 desde", X23_FILE)
    X23 = np.load(X23_FILE)  
    print("  • Cargando X34 desde", X34_FILE)
    X34 = np.load(X34_FILE)

    print("  • Cargando UMAP desde", UMAP_FILE)
    projections = np.load(UMAP_FILE) 

    print("  • Agregando similitudes globales…")
    S23 = X23.sum(axis=0)   
    S34 = X34.sum(axis=0)   

    print(f"  • Ejecutando multiway spectral clustering con K={n_clusters}…")
    a2, a3, a4 = multiway_spectral_clustering(S23, S34, n_clusters)

    print("  • Calculando correlaciones a nivel de cluster (global)…")
    C23, c2, c3 = aggregate_cluster_corr(S23[np.newaxis, :, :], a2, a3, n_clusters)
    C34, c3b, c4 = aggregate_cluster_corr(S34[np.newaxis, :, :], a3, a4, n_clusters)

    a2_clustering = a2
    a3_clustering = a3
    a4_clustering = a4

    counts2 = c2
    counts3 = c3
    counts4 = c4

    clusterCorr23 = C23
    clusterCorr34 = C34

    print("  • Clustering finalizado. Servidor listo en puerto 5000.\n")

    app.run(host='0.0.0.0', port=5000, debug=True)