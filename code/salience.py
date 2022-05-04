import math
import json
import skimage
import warnings
import cv2 as cv
import skimage.io
import numpy as np
import scipy.signal
import networkx as nx
import scipy.spatial.distance
from skimage.segmentation import slic
from skimage.util import img_as_float

# Sources
# scale_image - https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
# get_saliency_rbd - https://github.com/yhenon/pyimgsaliency
# find_aalir - based on https://github.com/lukasalexanderweber/lir


# region Ignore sklearn warnings
def warn(*args, **kwargs):
    # Ignore sklearn warnings
    pass


warnings.warn = warn
# endregion


def scale_image(img, scale):
    """Scale an iamge by a scale factor."""
    """Source: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/"""
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def get_saliency_rbd(img):
    """Saliency map calculation based on: Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. https://ieeexplore.ieee.org/document/6909756"""
    """Source: https://github.com/yhenon/pyimgsaliency"""
    def S(x1, x2, geodesic, sigma_clr=10):
        return math.exp(-pow(geodesic[x1, x2], 2)/(2*sigma_clr*sigma_clr))

    def compute_saliency_cost(smoothness, w_bg, wCtr):
        n = len(w_bg)
        A = np.zeros((n, n))
        b = np.zeros((n))

        for x in range(0, n):
            A[x, x] = 2 * w_bg[x] + 2 * (wCtr[x])
            b[x] = 2 * wCtr[x]
            for y in range(0, n):
                A[x, x] += 2 * smoothness[x, y]
                A[x, y] -= 2 * smoothness[x, y]

        x = np.linalg.solve(A, b)

        return x

    def path_length(path, G):
        dist = 0.0
        for i in range(1, len(path)):
            dist += G[path[i - 1]][path[i]]['weight']
        return dist

    def make_graph(grid):
        # get unique labels
        vertices = np.unique(grid)

        # map unique labels to [1,...,num_labels]
        reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
        grid = np.array([reverse_dict[x]
                        for x in grid.flat]).reshape(grid.shape)

        # create edges
        down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
        right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
        all_edges = np.vstack([right, down])
        all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
        all_edges = np.sort(all_edges, axis=1)
        num_vertices = len(vertices)
        edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
        # find unique connections
        edges = np.unique(edge_hash)
        # undo hashing
        edges = [[vertices[x % num_vertices],
                  vertices[int(x/num_vertices)]] for x in edges]

        return vertices, edges

    if len(img.shape) != 3:  # got a grayscale image
        img = skimage.color.gray2rgb(img)

    img_lab = img_as_float(skimage.color.rgb2lab(img))

    img_rgb = img_as_float(img)

    img_gray = img_as_float(skimage.color.rgb2gray(img))

    segments_slic = slic(img_rgb, n_segments=250,
                         compactness=10, sigma=1, enforce_connectivity=False)

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows*nrows + ncols*ncols)

    grid = segments_slic

    (vertices, edges) = make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

    centers = dict()
    colors = dict()
    distances = dict()
    boundary = dict()

    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
        colors[v] = np.mean(img_lab[grid == v], axis=0)

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

    G = nx.Graph()

    # buid the graph
    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        color_distance = scipy.spatial.distance.euclidean(
            colors[pt1], colors[pt2])
        G.add_edge(pt1, pt2, weight=color_distance)

    # add a new edge in graph if edges are both on boundary
    for v1 in vertices:
        if boundary[v1] == 1:
            for v2 in vertices:
                if boundary[v2] == 1:
                    color_distance = scipy.spatial.distance.euclidean(
                        colors[v1], colors[v2])
                    G.add_edge(v1, v2, weight=color_distance)

    geodesic = np.zeros((len(vertices), len(vertices)), dtype=float)
    spatial = np.zeros((len(vertices), len(vertices)), dtype=float)
    smoothness = np.zeros((len(vertices), len(vertices)), dtype=float)
    adjacency = np.zeros((len(vertices), len(vertices)), dtype=float)

    sigma_clr = 10.0
    sigma_bndcon = 1.0
    sigma_spa = 0.25
    mu = 0.1

    all_shortest_paths_color = nx.shortest_path(
        G, source=None, target=None, weight='weight')

    for v1 in vertices:
        for v2 in vertices:
            if v1 == v2:
                geodesic[v1, v2] = 0
                spatial[v1, v2] = 0
                smoothness[v1, v2] = 0
            else:
                geodesic[v1, v2] = path_length(
                    all_shortest_paths_color[v1][v2], G)
                spatial[v1, v2] = scipy.spatial.distance.euclidean(
                    centers[v1], centers[v2]) / max_dist
                smoothness[v1, v2] = math.exp(- (geodesic[v1, v2]
                                              * geodesic[v1, v2])/(2.0*sigma_clr*sigma_clr)) + mu

    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        adjacency[pt1, pt2] = 1
        adjacency[pt2, pt1] = 1

    for v1 in vertices:
        for v2 in vertices:
            smoothness[v1, v2] = adjacency[v1, v2] * smoothness[v1, v2]

    area = dict()
    len_bnd = dict()
    bnd_con = dict()
    w_bg = dict()
    ctr = dict()
    wCtr = dict()

    for v1 in vertices:
        area[v1] = 0
        len_bnd[v1] = 0
        ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- ((d_spa)*(d_spa))/(2.0*sigma_spa*sigma_spa))
            area_i = S(v1, v2, geodesic)
            area[v1] += area_i
            len_bnd[v1] += area_i * boundary[v2]
            ctr[v1] += d_app * w_spa
        bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
        w_bg[v1] = 1.0 - \
            math.exp(- (bnd_con[v1]*bnd_con[v1])/(2*sigma_bndcon*sigma_bndcon))

    for v1 in vertices:
        wCtr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- (d_spa*d_spa)/(2.0*sigma_spa*sigma_spa))
            wCtr[v1] += d_app * w_spa * w_bg[v2]

    # normalise value for wCtr

    min_value = min(wCtr.values())
    max_value = max(wCtr.values())

    minVal = [key for key, value in wCtr.items() if value == min_value]
    maxVal = [key for key, value in wCtr.items() if value == max_value]

    for v in vertices:
        wCtr[v] = (wCtr[v] - min_value)/(max_value - min_value)

    img_disp1 = img_gray.copy()
    img_disp2 = img_gray.copy()

    x = compute_saliency_cost(smoothness, w_bg, wCtr)

    for v in vertices:
        img_disp1[grid == v] = x[v]

    img_disp2 = img_disp1.copy()
    sal = np.zeros((img_disp1.shape[0], img_disp1.shape[1], 3))

    sal = img_disp2
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))

    return sal


def get_bbox(img):
    # Scale source image for efficient rbd calculation
    scale = 0.05
    img = scale_image(img, scale)

    # Estimate saliency
    img = get_saliency_rbd(img).astype('uint8')

    # Threshold saliency
    ret, img = cv.threshold(img, 200, 255, 0)

    # Find contours
    contours, _ = cv.findContours(
        img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Get biggest contour
    c = max(contours, key=cv.contourArea)

    # Get bounding rect of contour
    x, y, w, h = cv.boundingRect(c)

    # Returned scaled bbox
    return x/scale, y/scale, w/scale, h/scale


def find_aalir(mask):
    """Find axis-aligned largest interior rectangle."""
    """Based on: https://github.com/lukasalexanderweber/lir"""

    # If there are overlapping areas
    if 1 in mask:
        mask_t = mask.transpose()

        # Initialisations
        h_adjacency = np.zeros(mask.shape)
        v_adjacency = np.zeros(mask_t.shape)
        widths = np.zeros(mask.shape)
        heights = np.zeros(mask.shape)
        areas = np.zeros(mask.shape)

        # region Calculate adjacencies
        # Horizontal
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if(mask[y][x] == 0):
                    r = np.where(mask[y][x:] == 1)[0]
                    if(len(r) == 0):
                        r = mask.shape[1] - x
                    else:
                        r = r[0]

                    h_adjacency[y][x] = r

        # Vertical
        for x in range(mask_t.shape[0]):
            for y in range(mask_t.shape[1]):
                if(mask_t[x][y] == 0):
                    r = np.where(mask_t[x][y:] == 1)[0]
                    if(len(r) == 0):
                        r = mask_t.shape[1] - y
                    else:
                        r = r[0]
                    v_adjacency[x][y] = r
        # endregion

        # Calculate vectors
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                cell = [y, x]
                # Horizontal
                h_vec = [h_adjacency[cell[0]][cell[1]]]
                for i in range(cell[0]+1, h_adjacency.shape[0]):
                    h_vec.append(min(h_vec[-1], h_adjacency[i][cell[1]]))
                h_vec = list(set(h_vec))
                h_vec.sort(reverse=True)
                h_vec = [i for i in h_vec if i != 0]

                # Vertical
                v_vec = [v_adjacency[cell[1]][cell[0]]]
                for i in range(cell[1]+1, v_adjacency.shape[0]):
                    v_vec.append(min(v_vec[-1], v_adjacency[i][cell[0]]))
                v_vec = list(set(v_vec))
                v_vec.sort()
                v_vec = [i for i in v_vec if i != 0]

                # Calculate max area
                max_area = 0
                for i in range(len(h_vec)):
                    area = h_vec[i]*v_vec[i]

                    if(area > max_area):
                        max_area = area

                        widths[cell[0]][cell[1]] = h_vec[i]
                        heights[cell[0]][cell[1]] = v_vec[i]
                        areas[cell[0]][cell[1]] = area

        # Extract bbox
        y, x = np.unravel_index(areas.argmax(), areas.shape)
        w = int(widths[y][x])
        h = int(heights[y][x])

    else:
        # If there are no overlapping areas
        x = 0
        y = 0
        w = mask.shape[1]
        h = mask.shape[0]

    return x, y, w, h


def crop_img_to_mask(img_name, mask, scalar, drawFocal=False, recompute=False, drawBbox=False):
    """Crop an image to fit a binary mask. This is our image cropping algorithm."""

    # Read image
    img = cv.imread(img_name)

    # region Determine focal point
    focal_point = None
    # Attempt to get precomputed/cached value
    if not recompute:
        try:
            with open('./images/focal_points.json') as infile:
                focal_points = json.load(infile)
                focal_point = focal_points[img_name]
        except:
            pass

    # If we wish to recompute, or we tried to get a precomputed value but failed
    if recompute or focal_point == None:
        # Find the focal point of the image
        x, y, w, h = get_bbox(img)
        focal_point = [int(y+h/2), int(x+w/2)]

        # Append this focal point to focal_points.json file
        with open('./images/focal_points.json') as infile:
            focal_points = json.load(infile)
            focal_points[img_name] = focal_point
        with open('./images/focal_points.json', "w") as outfile:
            json.dump(focal_points, outfile)
    # endregion

    # region Helper drawing utils
    # Draw focal point to image if requested
    if drawFocal:
        cv.circle(img, (focal_point[1], focal_point[0]), 40, (255, 0, 255), -1)

    # Draw bbox to image if requested
    if drawBbox:
        with open('./images/gt_bboxes.json') as infile:
            bboxs = json.load(infile)
            bbox = bboxs[img_name]
        img[:, :] = (0, 0, 0)
        cv.rectangle(img, (bbox[0], bbox[1]), (bbox[0] +
                     bbox[2], bbox[1]+bbox[3]), (255, 255, 255), -1)
    # endregion

    # Find axis-aligned largest interal rectangle of mask - i.e. biggest part of the slot which isn't blocked by any overlapping stuff
    x, y, w, h = find_aalir(mask)
    x *= scalar
    y *= scalar
    w *= scalar
    h *= scalar

    # Calculate center of aalir
    aalir_center = (int(y+h/2), int(x+w/2))

    # Get scaled mask dimensions
    dim = (mask.shape[0]*scalar, mask.shape[1]*scalar)

    # Calculate the distances from the aalir center to the edges of the slot
    aalir_dists = {
        "l": aalir_center[1],
        "t": aalir_center[0],
        "r": dim[1]-aalir_center[1],
        "b": dim[0]-aalir_center[0]
    }

    # Get image dim
    img_dim = (img.shape[0], img.shape[1])

    # Calculate the distances from the focal point to the edges of the image
    img_dists = {
        "l": focal_point[1],
        "t": focal_point[0],
        "r": img_dim[1]-focal_point[1],
        "b": img_dim[0]-focal_point[0]
    }

    # Calculate the ratios between the distances in order to calculate the scale factor for the image
    ratios = {
        "l": aalir_dists["l"]/img_dists["l"],
        "t": aalir_dists["t"]/img_dists["t"],
        "r": aalir_dists["r"]/img_dists["r"],
        "b": aalir_dists["b"]/img_dists["b"],
    }

    # Calculate the scale factor
    scale_factor = ratios[max(ratios, key=ratios.get)]

    # Scale the image to appropriate size
    img = scale_image(img, scale_factor)

    # Crop image to correct size
    v_start = int(focal_point[0] * scale_factor - aalir_dists["t"])
    v_end = int(v_start + dim[0])

    h_start = int(focal_point[1] * scale_factor - aalir_dists["l"])
    h_end = int(h_start + dim[1])

    crop_img = img[v_start:v_end+1, h_start:h_end+1]

    return crop_img
