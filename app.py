import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import pandas as pd

# Read the CSV file
file_path = 'isolated.csv'
np_path_XYs = np.genfromtxt(file_path, delimiter=',')
path_XYs = []
unique_paths = np.unique(np_path_XYs[:, 0])
for i in unique_paths:
    npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
    XYs = []
    unique_shapes = np.unique(npXYs[:, 0])
    for j in unique_shapes:
        XY = npXYs[npXYs[:, 0] == j][:, 1:]
        XYs.append(XY)
    path_XYs.append(XYs)

shapes_data = path_XYs

def fit_circle(x, y):
    def calc_radius(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def cost(params):
        xc, yc, r = params
        return np.sum((calc_radius(xc, yc) - r) ** 2)

    x_m, y_m = np.mean(x), np.mean(y)
    r_guess = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))
    result = minimize(cost, (x_m, y_m, r_guess), method='L-BFGS-B')
    xc, yc, r = result.x
    return xc, yc, r

def generate_circle_points(xc, yc, r, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = xc + r * np.cos(angles)
    y = yc + r * np.sin(angles)
    return np.column_stack((x, y)), (xc, yc)

def fit_star(points, num_peaks=5):
    kmeans = KMeans(n_clusters=num_peaks * 2, random_state=42).fit(points)
    vertices = kmeans.cluster_centers_

    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    ordered_vertices = vertices[sorted_indices]

    return ordered_vertices

def generate_star_points(vertices):
    points = np.vstack((vertices, vertices[0]))
    return points

def adjust_to_quadrilateral(points):
    kmeans = KMeans(n_clusters=4, random_state=42).fit(points)
    vertices = kmeans.cluster_centers_

    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    quadrilateral = vertices[sorted_indices]

    adjusted_points = np.vstack((quadrilateral, quadrilateral[0]))

    return adjusted_points

def detect_symmetry(points):
    def reflection_cost(params):
        angle, x0, y0 = params
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        normal = np.array([cos_a, sin_a])
        midpoint = np.array([x0, y0])
        
        reflected_points = points - 2 * ((points - midpoint).dot(normal).reshape(-1, 1)) * normal
        cost = np.mean(np.linalg.norm(points - reflected_points, axis=1))
        return cost

    x_m, y_m = np.mean(points, axis=0)
    initial_params = [0, x_m, y_m]
    result = minimize(reflection_cost, initial_params, method='L-BFGS-B', bounds=[(0, np.pi), (None, None), (None, None)])
    return result.x

def plot_shapes_with_symmetry(title, shapes, symmetry_lines):
    plt.figure(figsize=(8, 8))
    for path_index, shape in enumerate(shapes):
        if path_index == 0:
            label = "Circle"
        elif path_index == 1:
            label = "Star"
        else:
            label = "Quadrilateral"
        plt.plot(shape[:, 0], shape[:, 1], '-', label=f"{label} ({title})")

        (x1, y1), (x2, y2) = symmetry_lines[path_index]
        plt.plot([x1, x2], [y1, y2], '--', color='grey')

    plt.title(f'Shapes with Symmetry Lines - {title}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

original_symmetry_lines = []
for path_index, path_shapes in enumerate(shapes_data):
    for shape_index, shape in enumerate(path_shapes):
        angle, x0, y0 = detect_symmetry(shape)
        length = np.max(np.linalg.norm(shape - np.array([x0, y0]), axis=1))
        original_symmetry_lines.append(((x0 - length * np.cos(angle), y0 - length * np.sin(angle)),
                                        (x0 + length * np.cos(angle), y0 + length * np.sin(angle))))

plot_shapes_with_symmetry("Original", [shape[0] for shape in shapes_data], original_symmetry_lines)

adjusted_shapes = []
adjusted_symmetry_lines = []
for path_index, path_shapes in enumerate(shapes_data):
    for shape_index, shape in enumerate(path_shapes):
        if path_index == 0 and shape_index == 0:
            xc, yc, r = fit_circle(shape[:, 0], shape[:, 1])
            circle_points, center = generate_circle_points(xc, yc, r)
            adjusted_shapes.append(circle_points)
            adjusted_symmetry_lines.append(((center[0] - r, center[1]), (center[0] + r, center[1])))
        elif path_index == 1 and shape_index == 0:
            smooth_star = fit_star(shape)
            star_points = generate_star_points(smooth_star)
            adjusted_shapes.append(star_points)
            angle, x0, y0 = detect_symmetry(star_points)
            length = np.max(np.linalg.norm(star_points - np.array([x0, y0]), axis=1))
            adjusted_symmetry_lines.append(((x0 - length * np.cos(angle), y0 - length * np.sin(angle)),
                                            (x0 + length * np.cos(angle), y0 + length * np.sin(angle))))
        elif path_index == 2 and shape_index == 0:
            adjusted_quadrilateral = adjust_to_quadrilateral(shape)
            adjusted_shapes.append(adjusted_quadrilateral)
            angle, x0, y0 = detect_symmetry(adjusted_quadrilateral)
            length = np.max(np.linalg.norm(adjusted_quadrilateral - np.array([x0, y0]), axis=1))
            adjusted_symmetry_lines.append(((x0 - length * np.cos(angle), y0 - length * np.sin(angle)),
                                            (x0 + length * np.cos(angle), y0 + length * np.sin(angle))))

plot_shapes_with_symmetry("Smoothed", adjusted_shapes, adjusted_symmetry_lines)
