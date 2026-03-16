import os
import cv2
import numpy as np


# ------------------ Task C1 Parameter Constants ------------------
# Gaussian Parameters
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 2
# Canny edge detection params
SOBEL_KERNEL_SIZE = 3
LOW_THRESHOLD_RATIO = 0.7
HIGH_THRESHOLD_RATIO = 0.5
# Hough Line Fitting params
NHOOD_SIZE = 21
NUM_THETAS = 360
NUM_PEAKS = 6

# ------------------ Canny Edge Detection Helpers ------------------
# Steps for canny edge detection
# 1. Compute gradients
# 2. Apply Non-maximum Suppression
# 3. Apply Hysteresis Thresholding


def non_max_suppression(g_magnitude, g_orientation):
    # g_magnitude and g_orientation will have the same shape so we could've picked either
    h, w = g_magnitude.shape
    # Matrix that will store the results after nms
    nms = np.zeros((h, w), dtype=np.float32)
    # Adjust gradient orientation angles to be between 0-180 degrees
    angles = g_orientation % 180

    # We essentially want to map all angles to one of four directions: 0, 45, 90, 135
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # These represent the two neighbour pixels that we want to compare against
            # They are arbitrarily initialized to infinity
            r, q = float("inf"), float("inf")

            # direction 0: compare left and right neighbours
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                r = g_magnitude[i, j + 1]
                q = g_magnitude[i, j - 1]
            # direction 45: compare top-right and bottom-left neighbours
            elif 22.5 <= angles[i, j] < 67.5:
                r = g_magnitude[i + 1, j - 1]
                q = g_magnitude[i - 1, j + 1]
            # direction 90: compare top and bottom neighbours
            elif 67.5 <= angles[i, j] < 112.5:
                r = g_magnitude[i + 1, j]
                q = g_magnitude[i - 1, j]
            # direction 135: compare top-left and bottom-right neighbours
            elif 112.5 <= angles[i, j] < 157.5:
                r = g_magnitude[i - 1, j - 1]
                q = g_magnitude[i + 1, j + 1]

            # Compare to neighbours and suppress non-maximums
            if (g_magnitude[i, j] >= r) and (g_magnitude[i, j] >= q):
                nms[i, j] = g_magnitude[i, j]
            else:
                nms[i, j] = 0
    return nms


def hysteresis_thresholding(nms, low_thresh, high_thresh):
    h, w = nms.shape
    # Matrix to store result of edges detected after hysteresis
    edges = np.zeros((h, w), dtype=np.uint8)
    stack = []
    # Intermediate pixel values for tracking edge strength
    WEAK = 1
    STRONG = 2

    for r in range(h):
        for c in range(w):
            if nms[r, c] >= high_thresh:
                edges[r, c] = STRONG
                # Add strong edges to the stack
                # We want to mark any of their weak neighbours as strong later in DFS
                stack.append((r, c))
            elif nms[r, c] >= low_thresh:
                # Mark as weak edge point - could be promoted to strong later in DFS
                edges[r, c] = WEAK
            # Any pixels below low_thresh remain 0 - non edges

    # DFS on strong edge nodes(pixels) to find any weak connecting edges to promote to strong
    while stack:
        r, c = stack.pop()
        # Check all 8 neighbours
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                # Skip the centre of the 3x3 grid
                if dr == 0 and dc == 0:
                    continue
                # Coordinates of neighbours in the 3x3 grid
                nr, nc = r + dr, c + dc
                # Prevent out of bounds access in the matrix
                if 0 <= nr < h and 0 <= nc < w:
                    # If neighbour is weak, promote to strong and add to stack
                    if edges[nr, nc] == WEAK:
                        edges[nr, nc] = STRONG
                        stack.append((nr, nc))
    # Keep only pixels marked as strong edges, suppressing the rest
    edges[edges != STRONG] = 0
    edges[edges == STRONG] = 1
    return edges


def canny_edge_detection(processed_img):
    # Compute gradients with Sobel
    g_x = cv2.Sobel(processed_img, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    g_y = cv2.Sobel(processed_img, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

    # Compute gradient magnitude and orientation in degrees
    g_magnitude = cv2.magnitude(g_x, g_y)
    g_orientation = np.arctan2(g_y, g_x) * (180.0 / np.pi)

    # Apply Non-Maximum Suppression
    nms = non_max_suppression(g_magnitude, g_orientation)

    # Calculate thresholds for hysteresis
    high_thresh = nms.max() * HIGH_THRESHOLD_RATIO
    low_thresh = high_thresh * LOW_THRESHOLD_RATIO

    # Apply hysteresis thresholding
    edges = hysteresis_thresholding(nms, low_thresh, high_thresh)
    return edges


# ------------------ Image Processing Helpers ------------------
def get_gaussian_kernel(kernel_size, sigma):
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def process_img(img):
    # Convert to grayscale and normalise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    # Denoise the grayscaled image via Gaussian blur
    kernel = get_gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    denoised_img = cv2.filter2D(gray, -1, kernel)
    return denoised_img


# ------------------ Hough Line Fitting Helpers ------------------
def hough_transform(edge_points_matrix):
    # Note: Hough transform fits lines to equation rho = x*cos(theta) + y*sin(theta)

    # Dimensions of the edge points matrix (same as original image)
    h, w = edge_points_matrix.shape
    # Get diagonal length of the image - this is the largest possible rho value
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    # Our rho values can be anywhere between +/- this diagonal length (a line going across the image from corner to corner)
    num_rhos = (2 * diag_len) + 1
    rho_vals = np.linspace(-diag_len, diag_len, num_rhos)
    # Each edge pixel at (x,y) can form infinite lines, one for each theta between 0 and 180 degrees (radians)
    # We can choose the resolution of theta in this range
    # With num_theta = 360, we have a resolution of 0.5 degrees
    theta_vals = np.linspace(0, np.pi, NUM_THETAS, endpoint=False)
    # We need sin and cos values for each theta as shown in the line equation
    cos_thetas = np.cos(theta_vals)
    sin_thetas = np.sin(theta_vals)
    # Matrix to hold accumulator values
    accumulator = np.zeros((num_rhos, NUM_THETAS), dtype=int)
    # Get coordinates of the strong edge points that we want to fit lines to
    y_coords, x_coords = np.nonzero(edge_points_matrix)

    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]

        # For each theta, calculate the corresponding rho for this edge point
        for theta_idx in range(NUM_THETAS):
            # Calculate rho value
            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            # Map the calculated value to the closest index
            rho_idx = int(round(rho + diag_len))
            # Increment the vote for that rho theta pair in the accumulator
            accumulator[rho_idx, theta_idx] += 1
    return accumulator, rho_vals, theta_vals


def find_hough_peaks(accumulator, thetas, rhos, num_peaks, nhood_size):
    peaks = []
    accumulator_suppressed = accumulator.copy()
    # Radius of the suppression neighbourhood (half window size)
    nhood_radius = nhood_size // 2
    h, w = accumulator_suppressed.shape

    # Loop over until we've found num_peaks number of peaks
    for _ in range(num_peaks):
        # Index of the peak with the most votes in the accumulator
        largest_vote_count_idx = np.argmax(accumulator_suppressed)
        # Indices of peak with most votes
        rho_idx, theta_idx = np.unravel_index(largest_vote_count_idx, accumulator_suppressed.shape)
        # Rho, Theta pair with the most votes - these are the parameters we need for our line
        rho_val = rhos[rho_idx]
        theta_val = thetas[theta_idx]
        # Add this line to our peaks list for consideration later
        peaks.append((rho_val, theta_val))

        # Zero out the votes around this peak to avoid picking nearby duplicates in the next iteration
        rho_min_nhood = max(0, rho_idx - nhood_radius)
        rho_max_nhood = min(h, rho_idx + nhood_radius)
        theta_min_nhood = max(0, theta_idx - nhood_radius)
        theta_max_nhood = min(w, theta_idx + nhood_radius)

        accumulator_suppressed[rho_min_nhood:rho_max_nhood, theta_min_nhood:theta_max_nhood] = 0

    return peaks


## ----------------- Angle Detection Helpers -----------------
def direction_strength(u, P, edges, line_length=35):
    # Get the edge support along a line starting at point P in direction u
    h, w = edges.shape
    strength = 0
    for k in range(1, line_length + 1):
        q = P + k * u
        y, x = int(round(q[1])), int(round(q[0]))
        if 0 <= y < h and 0 <= x < w:
            if edges[y, x] > 0:
                strength += 1
        else:
            break
    return strength


def interior_angle_from_edges(peaks, edges, min_degree_difference):
    h, w = edges.shape
    if len(peaks) < 2:
        return 0

    # Choose lines with two distinct directions
    min_difference = np.deg2rad(min_degree_difference)
    rho1, theta1 = peaks[0]
    rho2, theta2 = None, None
    for r, t in peaks[1:]:
        degree_difference = abs(theta1 - t) % np.pi
        degree_difference = degree_difference if degree_difference <= (np.pi / 2) else (np.pi - degree_difference)

        if degree_difference >= min_difference:
            rho2, theta2 = r, t
            break
    if theta2 is None:
        rho2, theta2 = peaks[1]

    # Find the intersection of the two lines, solving system of equations
    cos1, sin1 = np.cos(theta1), np.sin(theta1)
    cos2, sin2 = np.cos(theta2), np.sin(theta2)
    A = np.array([[cos1, sin1], [cos2, sin2]], dtype=np.float64)
    b = np.array([rho1, rho2], dtype=np.float64)

    try:
        # P is our x and y coordinates of intersection between the lines
        P = np.linalg.solve(A, b)
    except:
        # Lines are parallel
        return 0

    ### Get the direction of the lines
    unit_vec1 = np.array([-sin1, cos1], dtype=np.float64)
    unit_vec2 = np.array([-sin2, cos2], dtype=np.float64)
    # Normalise and use small val to prevent division by 0
    unit_vec1 = unit_vec1 / (np.linalg.norm(unit_vec1) + 1e-10)
    unit_vec2 = unit_vec2 / (np.linalg.norm(unit_vec2) + 1e-10)

    if direction_strength(unit_vec1, P, edges) < direction_strength(-unit_vec1, P, edges):
        unit_vec1 = -unit_vec1
    if direction_strength(unit_vec2, P, edges) < direction_strength(-unit_vec2, P, edges):
        unit_vec2 = -unit_vec2

    # Dot product to get interior angle between the lines - clip between -1 and 1 since this is the valid range for arccos
    dot = np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    return angle
