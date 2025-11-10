# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Simulation parameters
# T = 10  # total time (seconds)
# dt = 0.05  # time step
# times = np.arange(0, T, dt)
#
# # Adjustable parameters (try changing these)
# omega0 = 2.0  # initial speed (radians/s)
# accel = 5.0  # acceleration (radians/s^2)
# phi0 = 0.0  # initial phase (radians)
#
# # Compute true and wrapped phase
# true_phase = phi0 + omega0 * times + 0.5 * accel * times ** 2
# wrapped_phase = np.mod(true_phase, 2 * np.pi)
#
# # Setup figure
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax_circle, ax_phase = ax
#
# # Draw circle (representing the ring)
# theta = np.linspace(0, 2 * np.pi, 200)
# ax_circle.plot(np.cos(theta), np.sin(theta), 'gray', lw=1)
# ax_circle.set_aspect('equal', 'box')
# ax_circle.set_title('Car on Circular Path')
# car_point, = ax_circle.plot([], [], 'ro', ms=10)
#
# # Plot phase vs time
# ax_phase.set_xlim(0, T)
# ax_phase.set_ylim(0, 2 * np.pi)
# ax_phase.set_title('Wrapped Phase (radians)')
# ax_phase.set_xlabel('Time (s)')
# ax_phase.set_ylabel('Phase mod 2π')
# phase_line, = ax_phase.plot([], [], 'b-')
# true_line, = ax_phase.plot([], [], 'r--', alpha=0.4, label='True (unwrapped % 2π)')
# ax_phase.legend()
#
#
# # Update function for animation
# def update(frame):
#     t = times[:frame]
#     phase_line.set_data(t, wrapped_phase[:frame])
#     true_line.set_data(t, np.mod(true_phase[:frame], 2 * np.pi))
#
#     # car position on circle
#     x = np.cos(wrapped_phase[frame])
#     y = np.sin(wrapped_phase[frame])
#     car_point.set_data(x, y)
#
#     return car_point, phase_line, true_line
#
#
# ani = animation.FuncAnimation(
#     fig, update, frames=len(times), interval=50, blit=True, repeat=True
# )
#
# plt.tight_layout()
# plt.show()
#
#
# ###
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Simulation parameters
# T = 20          # total time (s)
# dt = 0.05
# times = np.arange(0, T, dt)
#
# # Motion parameters
# omega0 = 0.2    # initial angular speed (rad/s)
# accel = 0.1     # angular acceleration (rad/s^2)
# phi0 = 0.0
#
# # Define reset times (seconds)
# reset_times = [5, 10.5, 15.2]  # try changing these (or make random)
# reset_indices = [np.argmin(np.abs(times - rt)) for rt in reset_times]
#
# # Compute true phase with resets
# true_phase = np.zeros_like(times)
# phi = phi0
# for i in range(1, len(times)):
#     if i in reset_indices:
#         phi = 0.0  # reset car position
#     else:
#         phi += (omega0 + accel * times[i]) * dt
#     true_phase[i] = phi
#
# wrapped_phase = np.mod(true_phase, 2*np.pi)
#
# # --- Plot setup ---
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax_circle, ax_phase = ax
#
# # Circle plot
# theta = np.linspace(0, 2*np.pi, 200)
# ax_circle.plot(np.cos(theta), np.sin(theta), 'gray', lw=1)
# ax_circle.set_aspect('equal', 'box')
# ax_circle.set_title('Car on Circular Path')
# car_point, = ax_circle.plot([], [], 'ro', ms=10)
#
# # Phase vs time plot
# ax_phase.set_xlim(0, T)
# ax_phase.set_ylim(0, 2*np.pi)
# ax_phase.set_title('Wrapped Phase (with resets)')
# ax_phase.set_xlabel('Time (s)')
# ax_phase.set_ylabel('Phase mod 2π')
# phase_line, = ax_phase.plot([], [], 'b-', lw=2, label='Wrapped phase')
# ax_phase.legend()
# # mark reset times
# for rt in reset_times:
#     ax_phase.axvline(rt, color='r', linestyle='--', alpha=0.4)
#     ax_phase.text(rt, np.pi, 'RESET', rotation=90, color='r', ha='right', va='center')
#
# # Animation update
# def update(frame):
#     t = times[:frame]
#     phase_line.set_data(t, wrapped_phase[:frame])
#     # car position on circle
#     x = np.cos(wrapped_phase[frame])
#     y = np.sin(wrapped_phase[frame])
#     car_point.set_data(x, y)
#     return car_point, phase_line
#
# ani = animation.FuncAnimation(
#     fig, update, frames=len(times), interval=50, blit=True, repeat=True
# )
#
# plt.tight_layout()
# plt.show()
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # --- Simulation parameters ---
# T = 20.0          # total time (s)
# dt = 0.05
# times = np.arange(0.0, T + dt, dt)
#
# # Motion parameters
# omega0 = 3.0    # initial angular speed (rad/s)
# accel = 1.5     # angular acceleration (rad/s^2)
# phi0 = 0.0
#
# # Define reset times (seconds) -- change these or make random
# reset_times = [5.0, 10.5, 15.2]
#
# # Convert reset times to the nearest index (ints)
# reset_indices = [int(np.argmin(np.abs(times - rt))) for rt in reset_times]
# reset_indices = sorted(set(reset_indices))  # unique, sorted
#
# # --- Compute true phase with resets and record distances ---
# true_phase = np.zeros_like(times)
# phi = phi0
# distances = []  # phase before each reset
# last_reset_idx = 0
#
# for i in range(1, len(times)):
#     # simple kinematic increment (using current time for accel term)
#     phi += (omega0 + accel * times[i]) * dt
#
#     if i in reset_indices:
#         # record how far (phase) we got since last reset (phi currently before reset)
#         distances.append(phi)
#         phi = 0.0  # reset
#         last_reset_idx = i
#
#     true_phase[i] = phi
#
# # wrapped phase for visualization
# wrapped_phase = np.mod(true_phase, 2 * np.pi)
#
# # --- Plot setup ---
# fig = plt.figure(figsize=(12, 6))
# gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
#
# # Circle subplot
# ax_circle = fig.add_subplot(gs[0, 0])
# theta = np.linspace(0, 2 * np.pi, 200)
# ax_circle.plot(np.cos(theta), np.sin(theta), "gray", lw=1)
# ax_circle.set_aspect("equal", "box")
# ax_circle.set_title("Car on Circular Path")
# car_point, = ax_circle.plot([], [], "ro", ms=10)
#
# # Phase vs time subplot
# ax_phase = fig.add_subplot(gs[0, 1])
# ax_phase.set_xlim(0, T)
# ax_phase.set_ylim(0, 2 * np.pi)
# ax_phase.set_title("Wrapped Phase (with resets)")
# ax_phase.set_xlabel("Time (s)")
# ax_phase.set_ylabel("Phase mod 2π")
# phase_line, = ax_phase.plot([], [], "b-", lw=2, label="Wrapped phase")
# ax_phase.legend()
#
# # mark reset times on phase plot
# for rt in reset_times:
#     ax_phase.axvline(rt, color="r", linestyle="--", alpha=0.4)
#     ax_phase.text(rt, np.pi, "RESET", rotation=90, color="r", ha="right", va="center")
#
# # Distance bars subplot (bottom)
# ax_diff = fig.add_subplot(gs[1, :])
# n_resets = len(distances)
# if n_resets > 0:
#     bar_positions = np.arange(n_resets)
#     bars = ax_diff.bar(bar_positions, [0.0] * n_resets, align="center")
#     ax_diff.set_ylim(0.0, max(distances) * 1.2)
# else:
#     bars = []
#     ax_diff.set_ylim(0.0, 1.0)
# ax_diff.set_title("Phase Difference at Reset (Distance Travelled)")
# ax_diff.set_xlabel("Reset Index")
# ax_diff.set_ylabel("ΔPhase (radians)")
#
# # --- Animation update function ---
# def update(frame):
#     # update phase line
#     t = times[: frame + 1]
#     phase_line.set_data(t, wrapped_phase[: frame + 1])
#
#     # update car point on circle (use wrapped phase at current frame)
#     x = np.cos(wrapped_phase[frame])
#     y = np.sin(wrapped_phase[frame])
#     car_point.set_data(x, y)
#
#     # update bars for resets that have happened up to current frame
#     artists = [car_point, phase_line]
#     for j, idx in enumerate(reset_indices):
#         if frame >= idx and n_resets > 0:
#             # set the bar height to the recorded distance
#             rect = bars[j]
#             rect.set_height(distances[j])
#         if n_resets > 0:
#             artists.append(bars[j])
#
#     # return a list or tuple of artists
#     return artists
#
# # Create animation
# ani = animation.FuncAnimation(
#     fig, update, frames=len(times), interval=50, blit=True, repeat=True
# )
#
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation parameters ---
T_segment = 5.0     # time per run (s)
dt = 0.05
num_segments = 12    # number of car runs (resets)
times = np.arange(0, T_segment, dt)
T_total = T_segment * num_segments

# --- Motion parameters ---
omega0 = 1.0      # starting angular speed (rad/s)
omega_step = 1.2  # speed increment each new start
accel0 = 4.8      # starting acceleration (rad/s²)
accel_step = 0.05 # acceleration increment per start

# --- Arrays to hold full timeline ---
all_time = []
all_phase = []
all_dphase = []
all_accel = []
segment_labels = []

phase = 0.0

for i in range(num_segments):
    omega = omega0 + i * omega_step
    accel = accel0 + i * accel_step

    # compute motion for this segment
    phi = omega * times + 0.5 * accel * times**2
    dphi = omega + accel * times
    aphi = np.full_like(times, accel)

    # store with time offset
    time_offset = i * T_segment
    all_time.append(times + time_offset)
    all_phase.append(phi)
    all_dphase.append(dphi)
    all_accel.append(aphi)
    segment_labels.append(np.ones_like(times) * i)

# concatenate all runs
all_time = np.concatenate(all_time)
all_phase = np.concatenate(all_phase)
all_dphase = np.concatenate(all_dphase)
all_accel = np.concatenate(all_accel)
segment_labels = np.concatenate(segment_labels)

# wrap phase to circle (for visualization)
wrapped_phase = np.mod(all_phase, 2 * np.pi)

# --- Setup figure ---
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

# 1️⃣ Car on circle
ax_circle = fig.add_subplot(gs[0, 0])
theta = np.linspace(0, 2*np.pi, 200)
ax_circle.plot(np.cos(theta), np.sin(theta), 'gray', lw=1)
ax_circle.set_aspect('equal', 'box')
ax_circle.set_title("Car on Circular Path")
car_point, = ax_circle.plot([], [], 'ro', ms=10)

# 2️⃣ Phase plot
ax_phase = fig.add_subplot(gs[0, 1])
ax_phase.set_xlim(0, T_total)
ax_phase.set_ylim(0, 2*np.pi)
ax_phase.set_title("Wrapped Phase vs Time")
ax_phase.set_xlabel("Time (s)")
ax_phase.set_ylabel("Phase mod 2π")
phase_line, = ax_phase.plot([], [], 'b-', lw=2)
for i in range(1, num_segments):
    ax_phase.axvline(i * T_segment, color='r', linestyle='--', alpha=0.4)

# 3️⃣ Delta phase (speed)
ax_dphi = fig.add_subplot(gs[1, :])
ax_dphi.set_xlim(0, T_total)
ax_dphi.set_title("ΔPhase (angular speed)")
ax_dphi.set_xlabel("Time (s)")
ax_dphi.set_ylabel("ΔPhase (rad/s)")
line_dphi, = ax_dphi.plot([], [], 'g-', lw=2)

# 4️⃣ Phase acceleration
ax_acc = fig.add_subplot(gs[2, :])
ax_acc.set_xlim(0, T_total)
ax_acc.set_title("Phase Acceleration")
ax_acc.set_xlabel("Time (s)")
ax_acc.set_ylabel("Acceleration (rad/s²)")
line_acc, = ax_acc.plot([], [], 'm-', lw=2)

# Set y-limits dynamically later
ax_dphi.set_ylim(0, np.max(all_dphase) * 1.2)
ax_acc.set_ylim(0, np.max(all_accel) * 1.5)

# --- Animation update ---
def update(frame):
    # data until current frame
    t = all_time[:frame]
    wrapped = wrapped_phase[:frame]
    dphi = all_dphase[:frame]
    acc = all_accel[:frame]

    # update circle
    x = np.cos(wrapped_phase[frame])
    y = np.sin(wrapped_phase[frame])
    car_point.set_data(x, y)

    # update lines
    phase_line.set_data(t, wrapped)
    line_dphi.set_data(t, dphi)
    line_acc.set_data(t, acc)

    return [car_point, phase_line, line_dphi, line_acc]

ani = animation.FuncAnimation(
    fig, update, frames=len(all_time), interval=40, blit=True, repeat=True
)

plt.tight_layout()
plt.show()
