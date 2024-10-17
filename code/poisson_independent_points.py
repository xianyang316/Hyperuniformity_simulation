import numpy as np
import matplotlib.pyplot as plt

class PoissonPointProcess2D:
    """
    Class to generate Poisson-distributed random points in 2D space.
    """
    def __init__(self, intensity, area_size):
        """
        Initialize the Poisson point process.

        Parameters:
        - intensity (float): Average number of points per unit area (λ).
        - area_size (tuple): Size of the area as (width, height).
        """
        self.intensity = intensity
        self.width = area_size[0]
        self.height = area_size[1]
        self.num_points = None
        self.points = None

    def generate_points(self):
        """
        Generate Poisson-distributed random points within the specified area.
        """
        # Total area
        area = self.width * self.height

        # Expected number of points (Poisson-distributed)
        expected_num_points = self.intensity * area

        # Actual number of points (sampled from a Poisson distribution)
        self.num_points = np.random.poisson(expected_num_points)

        # Generate uniform random points within the area
        x_coords = np.random.uniform(0, self.width, self.num_points)
        y_coords = np.random.uniform(0, self.height, self.num_points)

        # Store points as a NumPy array of shape (num_points, 2)
        self.points = np.column_stack((x_coords, y_coords))

    def get_points(self):
        """
        Return the generated points.

        Returns:
        - points (ndarray): Array of shape (num_points, 2) containing point coordinates.
        """
        if self.points is None:
            raise ValueError("Points have not been generated yet. Call generate_points() first.")
        return self.points

    def plot_points(self):
        """
        Plot the generated points using Matplotlib.
        """
        if self.points is None:
            raise ValueError("Points have not been generated yet. Call generate_points() first.")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.points[:, 0], self.points[:, 1], color='blue', s=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Poisson Point Process in 2D (λ={self.intensity}, N={self.num_points})')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()