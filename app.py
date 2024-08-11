import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
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
        return path_XYs
    except Exception as e:
        print("An error occurred:", e)
        raise

def detect_shapes(path_XYs):
    categorized_shapes = []
    for shapes in path_XYs:
        for XY in shapes:
            if len(XY) < 5:  
                continue
            # Circle Detection
            center = np.mean(XY, axis=0)
            radii = np.linalg.norm(XY - center, axis=1)
            radius_variance = np.std(radii)
            radius_mean = np.mean(radii)

            # Fit a line to the points
            X = XY[:, 0].reshape(-1, 1)
            y = XY[:, 1]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            normalized_mse = mse / (np.ptp(y)**2)  

            # Classification based on variance and MSE
            if radius_variance < 0.1 * radius_mean:
                categorized_shapes.append(('Circle', XY))
            elif normalized_mse < 0.01:
                categorized_shapes.append(('Line', XY))
            elif normalized_mse < 0.1:
                categorized_shapes.append(('Near-Line', XY))
            else:
                categorized_shapes.append(('Doodle', XY))
    return categorized_shapes

def plot_categorized_shapes(categorized_shapes):
    colours = {'Line': 'blue', 'Near-Line': 'cyan', 'Circle': 'red', 'Doodle': 'green'}
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))
    for shape_type, XY in categorized_shapes:
        ax.plot(XY[:, 0], XY[:, 1], c=colours[shape_type], label=shape_type, linewidth=2)
    ax.set_aspect('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

# Example usage
csv_path = 'frag0.csv'  
path_XYs = read_csv(csv_path)
categorized_shapes = detect_shapes(path_XYs)
plot_categorized_shapes(categorized_shapes)
