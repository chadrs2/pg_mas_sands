import numpy as np
import matplotlib.pyplot as plt

def generate_curved_line_image(N, M, curvature):
    # Create a grid of indices
    x = np.linspace(-1, 1, M)
    y = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, y)

    # Generate curved line
    curve = np.exp(-curvature * (- xx**2 + yy**2))

    return curve

# Parameters
N = 100  # Number of rows
M = 200  # Number of columns
curvature = 1 # Adjust the curvature as needed

# Generate the curved line image
image_array = generate_curved_line_image(N, M, curvature)

# Plot the image
plt.imshow(image_array, cmap='gray', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Curved Line Image')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
