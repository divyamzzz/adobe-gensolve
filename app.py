import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from scipy.optimize import minimize
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

# Data for each shape
shapes_data = path_XYs
num_paths = len(shapes_data)

# Fit circle function
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
    return np.column_stack((x, y))

# Fit star shape
def fit_star(points, num_peaks=5):
    # Approximate star by finding vertices using KMeans
    kmeans = KMeans(n_clusters=num_peaks * 2, random_state=42).fit(points)
    vertices = kmeans.cluster_centers_

    # Order vertices by angle
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    ordered_vertices = vertices[sorted_indices]

    # Return as star-like shape
    return ordered_vertices

def generate_star_points(vertices):
    points = np.vstack((vertices, vertices[0]))  # Connect last point to first
    return points

# Function to adjust the third shape to form a quadrilateral
def adjust_to_quadrilateral(points):
    # Use KMeans to find four central points as the vertices of a quadrilateral
    kmeans = KMeans(n_clusters=4, random_state=42).fit(points)
    vertices = kmeans.cluster_centers_

    # Order the vertices in a clockwise direction
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    quadrilateral = vertices[sorted_indices]

    # Connect the vertices to form a quadrilateral
    adjusted_points = np.vstack((quadrilateral, quadrilateral[0]))

    return adjusted_points

# Adjust shapes and replace in the dataset
adjusted_shapes = []
for path_index, path_shapes in enumerate(shapes_data):
    for shape_index, shape in enumerate(path_shapes):
        if path_index == 0 and shape_index == 0:
            # Smooth the circle
            xc, yc, r = fit_circle(shape[:, 0], shape[:, 1])
            adjusted_shapes.append(generate_circle_points(xc, yc, r))
        elif path_index == 1 and shape_index == 0:
            # Smooth the star
            smooth_star = fit_star(shape)
            adjusted_shapes.append(generate_star_points(smooth_star))
        elif path_index == 2 and shape_index == 0:
            # Adjust to quadrilateral
            adjusted_quadrilateral = adjust_to_quadrilateral(shape)
            adjusted_shapes.append(adjusted_quadrilateral)

# Prepare data for CSV output
output_data = []
for path_index, shape in enumerate(adjusted_shapes):
    for point_index, (x, y) in enumerate(shape):
        output_data.append([path_index + 1, 1, x, y])  # Path and shape identifiers

# Convert to DataFrame and save as CSV
df = pd.DataFrame(output_data, columns=['Path', 'Shape', 'X', 'Y'])
df.to_csv('adjusted_shapes.csv', index=False)

# Plot the adjusted shapes
plt.figure(figsize=(8, 8))
for path_index, shape in enumerate(adjusted_shapes):
    if path_index == 0:
        label = "Smooth Circle"
    elif path_index == 1:
        label = "Smooth Star"
    else:
        label = "Quadrilateral"
    plt.plot(shape[:, 0], shape[:, 1], '-', label=label)

plt.title('Adjusted Shapes')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()
