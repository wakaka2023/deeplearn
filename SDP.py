import numpy as np
import matplotlib.pyplot as plt

def generate_signal():
    # Time values
    t = np.linspace(0, 2 * np.pi, 1000)

    # Create two sine waves with different frequencies and amplitudes
    sine_wave_1 = np.sin(2 * np.pi * 1 * t)  # Frequency = 1 Hz
    sine_wave_2 = 0.5 * np.sin(2 * np.pi * 3 * t)  # Frequency = 3 Hz, with reduced amplitude

    x = sine_wave_1 + sine_wave_2

    noise = np.random.normal(0, 0.1, len(t))

    x_noisy = x + noise

    return x_noisy


def sdp_transformation(x, l, g, h=60):
    # Normalize the signal
    x_min = np.min(x)
    x_max = np.max(x)
    r = (x - x_min) / (x_max - x_min)

    # Define the angles
    n = len(x)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(n):
        # Calculate theta and phi using the formulas
        theta[i] = h + (x[(i + l) % n] - x_min) / (x_max - x_min) * g
        phi[i] = h - (x[(i + l) % n] - x_min) / (x_max - x_min) * g

    return r, theta, phi


def plot_sdp(r, theta, phi):
    # Plot the SDP pattern in polar coordinates
    plt.figure(figsize=(6, 6))

    ax = plt.subplot(111, projection='polar')
    ax.scatter(np.radians(theta), r, marker='.')
    ax.scatter(np.radians(phi), r,   marker='.')
    ax.set_axis_off()
    plt.show()


def rotate_pattern(r, theta, phi, n_rotations, angle_step):
    all_r = np.copy(r)
    all_theta = np.copy(theta)
    all_phi = np.copy(phi)

    for i in range(1, n_rotations):
        rotated_theta = theta + i * angle_step
        rotated_phi = phi + i * angle_step

        rotated_theta = np.mod(rotated_theta, 360)
        rotated_phi = np.mod(rotated_phi, 360)

        all_r = np.concatenate((all_r, r))
        all_theta = np.concatenate((all_theta, rotated_theta))
        all_phi = np.concatenate((all_phi, rotated_phi))

    return all_r, all_theta, all_phi

# Generate signal with two sine waves and added noise
x = generate_signal()

# Parameters for SDP
l = 7  # Time interval parameter
g = 40  # Angle magnification factor

# Generate SDP
r, theta, phi = sdp_transformation(x, l, g)
r_full, theta_full, phi_full = rotate_pattern(r, theta, phi, 6, 60)

# Plot the SDP pattern
plot_sdp(r_full, theta_full, phi_full)
