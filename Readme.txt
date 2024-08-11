# Project Overview: Doodle Shape Regularization and Symmetry Detection

This project was developed during Round 2 of Adobe GenSolve. The primary objective of this project is to detect shapes present in a file, identify their symmetry, and smoothen them. It focuses on converting doodles into aesthetically pleasing images by regularizing and enhancing their geometric properties. This process involves utilizing several computational techniques and technologies to achieve shape detection, symmetry analysis, and shape smoothing.

## Technologies Used

1. **Python**: The programming language used to implement the entire project, offering powerful libraries for numerical computation and data manipulation.
2. **NumPy**: Used for numerical operations and handling of multidimensional arrays, which are crucial for processing shape data.
3. **Matplotlib**: Utilized for visualizing the shapes and symmetry lines before and after smoothing, providing clear graphical output.
4. **SciPy**: Specifically, the `optimize` module is used for fitting geometric shapes (such as circles) and minimizing functions during symmetry detection.
5. **scikit-learn (KMeans)**: Employed for clustering points to approximate vertices of shapes (e.g., stars and quadrilaterals), aiding in shape detection and adjustment.
6. **Pandas**: Used to handle data frames for organizing the output data before saving it to CSV, facilitating structured data manipulation and export.

## Detailed Explanation

### 1. Data Reading and Preprocessing
- The project begins by reading shape data from a CSV file using NumPy's `genfromtxt` function.
- The data is organized into paths and shapes, allowing the identification of unique shapes that need to be processed.

### 2. Shape Detection
- **Circle Fitting**: The project uses optimization techniques to fit circles to the detected shapes. The `minimize` function from SciPy is used to determine the center and radius of circles by minimizing the difference between actual and calculated radii.
- **Star and Quadrilateral Detection**: KMeans clustering is applied to approximate the vertices of stars and quadrilaterals. The vertices are ordered by angles to identify the shape's structure accurately.

### 3. Symmetry Detection
- The project implements a symmetry detection algorithm that minimizes the distance between original points and their reflections across a potential symmetry line.
- A cost function is defined to evaluate the quality of symmetry, with the `minimize` function used to find the optimal line of symmetry for each shape.

### 4. Shape Regularization and Smoothing
- Once the symmetry is detected, shapes are adjusted to regularize their structure, such as smoothing a star into evenly spaced points or adjusting irregular polygons into quadrilaterals.
- The circle is inherently symmetric and regular, but adjustments ensure precision by recalculating based on the fitted circle parameters.

### 5. Visualization
- The project visualizes both original and smoothed shapes, along with their symmetry lines, using Matplotlib. 
- Two sets of plots are generated to allow a side-by-side comparison of the shapes before and after regularization.

### 6. Output and Export
- The regularized shape data is converted into a structured format using Pandas and exported as a CSV file.
- This allows for further analysis or integration into other systems that may utilize the processed shape data for applications like graphic design or digital art enhancement.

### Conclusion
The project effectively transforms doodles into regular, symmetric, and visually appealing shapes. By leveraging advanced computational techniques and data processing libraries, the project enhances the aesthetics of hand-drawn shapes, making them suitable for professional or creative applications. The use of symmetry detection and regularization algorithms ensures that the output shapes maintain their artistic integrity while achieving geometric perfection.
