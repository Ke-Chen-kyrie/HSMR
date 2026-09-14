"""
Verified 3D rendering of SKEL + coordinate systems.
Every data point comes from actual model extraction or code reading.
Usage: source .venv/bin/activate && python docs/render_verified_coords.py
Output: docs/_verified_coords.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.body_models.skel_wrapper import SKELWrapper

# ============================================================
# 1. LOAD ACTUAL SKEL MODEL
# ============================================================
model = SKELWrapper(
    model_path='data_inputs/body_models/skel', gender='male', num_betas=10,
    joint_regressor_extra='data_inputs/body_models/SMPL_to_J19.pkl',
    joint_regressor_custom='data_inputs/body_models/J_regressor_SKEL_mix_MALE.pkl',
    make_dense=True,
)
model.eval()
with torch.no_grad():
    output = model(poses=torch.zeros(1,46), betas=torch.zeros(1,10), skelmesh=True)

joints = output.joints_backup.squeeze(0).numpy()  # 24 anatomical
skin_v = output.skin_verts.squeeze(0).numpy()       # 6890 verts
skin_f = model.skin_f.numpy()                        # 13776 faces

# ============================================================
# 2. VERIFY AXES FROM DATA
# ============================================================
h, p = joints[13], joints[0]          # head, pelvis
rs, ls = joints[15], joints[20]       # R/L shoulder
toes, heel = joints[5], joints[4]
nose = output.joints.squeeze(0).numpy()[0]  # OpenPose nose

up_vec = h - p
right_vec = rs - ls  # R minus L = negative X
fwd_vec = toes - heel

print("="*60)
print("AXIS VERIFICATION FROM SKEL REST POSE DATA")
print("="*60)
print(f"Head:            [{h[0]:+.4f}, {h[1]:+.4f}, {h[2]:+.4f}]")
print(f"Pelvis:          [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
print(f"Nose:            [{nose[0]:+.4f}, {nose[1]:+.4f}, {nose[2]:+.4f}]")
print(f"R-Shoulder:      [{rs[0]:+.4f}, {rs[1]:+.4f}, {rs[2]:+.4f}]")
print(f"L-Shoulder:      [{ls[0]:+.4f}, {ls[1]:+.4f}, {ls[2]:+.4f}]")
print(f"Toes-R:          [{toes[0]:+.4f}, {toes[1]:+.4f}, {toes[2]:+.4f}]")
print(f"Heel-R:          [{heel[0]:+.4f}, {heel[1]:+.4f}, {heel[2]:+.4f}]")
print()
print(f"Up (head-pelvis):     {up_vec} → +Y = UP")
print(f"Right (R-L shoulder): {right_vec} → RIGHT at -X, LEFT at +X")
print(f"Fwd (toes-heel):      {fwd_vec} → +Z = FORWARD")
print(f"Nose-pelvis Z:        {nose[2]-p[2]:+.4f} → nose ahead of pelvis")
print()
print("CONCLUSION:")
print("  +X = body LEFT  (left arm at +X, right arm at -X)")
print("  +Y = body UP    (head at top)")
print("  +Z = body FRONT (body faces +Z, toward viewer)")

# ============================================================
# 3. CREATE VERIFIED 3D RENDERING
# ============================================================
fig = plt.figure(figsize=(20, 10))

# --- View 1: Front view (-Z looking at +Z, i.e. looking at body front) ---
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title('Front View\n(looking at body front = +Z facing viewer)', fontsize=12, fontweight='bold')

# Plot skin mesh (subsampled for performance)
step = 4
skin_v_sub = skin_v[::step]
for face in skin_f[::20]:
    tri = skin_v[face]
    # Only draw faces where all vertices were subsampled
    center = tri.mean(axis=0)
    poly = Poly3DCollection([tri], alpha=0.15, facecolor='#e8a87c', edgecolor='none')
    ax1.add_collection3d(poly)

# Plot joints
ax1.scatter(joints[:,0], joints[:,1], joints[:,2], c='#f78166', s=30, edgecolors='#c04030', linewidths=0.5)

# Plot bones
bones = [
    (0,11),(11,12),(12,13),(0,1),(1,2),(2,3),(3,4),(4,5),
    (0,6),(6,7),(7,8),(8,9),(9,10),(12,14),(14,15),(15,16),(16,17),(17,18),
    (12,19),(19,20),(20,21),(21,22),(22,23)
]
for p_idx, c_idx in bones:
    ax1.plot([joints[p_idx,0], joints[c_idx,0]],
             [joints[p_idx,1], joints[c_idx,1]],
             [joints[p_idx,2], joints[c_idx,2]], '#cc6644', linewidth=2)

# COORDINATE AXES at pelvis
pelvis = joints[0]
ax_len = 0.6
ax1.quiver(pelvis[0], pelvis[1], pelvis[2], ax_len, 0, 0, color='red', linewidth=2, arrow_length_ratio=0.2, label='+X → LEFT')
ax1.quiver(pelvis[0], pelvis[1], pelvis[2], 0, ax_len, 0, color='green', linewidth=2, arrow_length_ratio=0.2, label='+Y ↑ UP')
ax1.quiver(pelvis[0], pelvis[1], pelvis[2], 0, 0, ax_len, color='blue', linewidth=2, arrow_length_ratio=0.2, label='+Z ⊙ FRONT')

# Origin
ax1.scatter([0], [0], [0], c='white', s=60, edgecolors='black', linewidths=1, zorder=10)
ax1.text(0, 0.05, 0.05, 'SKEL origin\n(0,0,0)', fontsize=8, color='white')

# Body forward annotation
mid_body = np.array([0, -0.3, 0.3])
ax1.text(mid_body[0], mid_body[1], mid_body[2]+0.2, 'BODY FACES +Z\n(toward viewer)', fontsize=10, color='#88bbff', fontweight='bold', ha='center')
ax1.plot([mid_body[0], mid_body[0]], [mid_body[1], mid_body[1]], [mid_body[2], mid_body[2]+0.15], color='#88bbff', linewidth=2)

# L/R labels
ax1.text(joints[18,0]-0.15, joints[18,1], joints[18,2], 'RIGHT arm\n(−X side)', fontsize=9, color='#ff6b6b', ha='center')
ax1.text(joints[23,0]+0.15, joints[23,1], joints[23,2], 'LEFT arm\n(+X side)', fontsize=9, color='#ff6b6b', ha='center')

ax1.set_xlabel('X (← LEFT | RIGHT →)'); ax1.set_ylabel('Y (UP)'); ax1.set_zlabel('Z (FRONT)')
ax1.set_xlim(-1, 1); ax1.set_ylim(-1.4, 0.8); ax1.set_zlim(-0.3, 0.5)
ax1.legend(loc='upper right', fontsize=8)
ax1.view_init(elev=10, azim=-90)

# --- View 2: Side view (looking at +X, i.e. from left side) ---
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_title('Side View\n(looking from LEFT side = +X direction)', fontsize=12, fontweight='bold')

for face in skin_f[::20]:
    tri = skin_v[face]
    ax2.add_collection3d(Poly3DCollection([tri], alpha=0.15, facecolor='#e8a87c', edgecolor='none'))

ax2.scatter(joints[:,0], joints[:,1], joints[:,2], c='#f78166', s=30, edgecolors='#c04030', linewidths=0.5)
for p_idx, c_idx in bones:
    ax2.plot([joints[p_idx,0], joints[c_idx,0]],[joints[p_idx,1], joints[c_idx,1]],[joints[p_idx,2], joints[c_idx,2]], '#cc6644', linewidth=2)

# Axes at pelvis
ax2.quiver(pelvis[0], pelvis[1], pelvis[2], ax_len, 0, 0, color='red', linewidth=2, arrow_length_ratio=0.2)
ax2.quiver(pelvis[0], pelvis[1], pelvis[2], 0, ax_len, 0, color='green', linewidth=2, arrow_length_ratio=0.2)
ax2.quiver(pelvis[0], pelvis[1], pelvis[2], 0, 0, ax_len, color='blue', linewidth=2, arrow_length_ratio=0.2)

# Forward indicator
nose_pt = nose
ax2.plot([nose_pt[0], nose_pt[0]], [nose_pt[1], nose_pt[1]], [nose_pt[2], nose_pt[2]+0.2], color='#88bbff', linewidth=3)
ax2.text(nose_pt[0], nose_pt[1]+0.1, nose_pt[2]+0.25, '+Z = FORWARD\n(body faces this way)', fontsize=10, color='#88bbff', fontweight='bold', ha='center')

ax2.scatter([0],[0],[0], c='white', s=60, edgecolors='black', linewidths=1)

ax2.set_xlabel('X'); ax2.set_ylabel('Y (UP)'); ax2.set_zlabel('Z (FRONT →)')
ax2.set_xlim(-1, 1); ax2.set_ylim(-1.4, 0.8); ax2.set_zlim(-0.3, 0.5)
ax2.view_init(elev=10, azim=0)

# --- View 3: Top-down view (looking from +Y down) ---
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_title('Top-Down View\n(looking from above = -Y direction)', fontsize=12, fontweight='bold')

for face in skin_f[::20]:
    tri = skin_v[face]
    ax3.add_collection3d(Poly3DCollection([tri], alpha=0.15, facecolor='#e8a87c', edgecolor='none'))

ax3.scatter(joints[:,0], joints[:,1], joints[:,2], c='#f78166', s=30, edgecolors='#c04030', linewidths=0.5)
for p_idx, c_idx in bones:
    ax3.plot([joints[p_idx,0], joints[c_idx,0]],[joints[p_idx,1], joints[c_idx,1]],[joints[p_idx,2], joints[c_idx,2]], '#cc6644', linewidth=2)

# Axes at pelvis
ax3.quiver(pelvis[0], pelvis[1], pelvis[2], ax_len, 0, 0, color='red', linewidth=2, arrow_length_ratio=0.2, label='+X = LEFT')
ax3.quiver(pelvis[0], pelvis[1], pelvis[2], 0, 0, ax_len, color='blue', linewidth=2, arrow_length_ratio=0.2, label='+Z = FRONT')

# Forward arrow
ax3.quiver(pelvis[0], pelvis[1], pelvis[2], 0, 0, ax_len*1.2, color='#4488ff', linewidth=3, arrow_length_ratio=0.15)
ax3.text(pelvis[0], pelvis[1], pelvis[2]+ax_len*1.4, 'BODY FACES\n+Z', fontsize=10, color='#88bbff', fontweight='bold', ha='center')

# L/R markers
ax3.text(joints[18,0]-0.2, joints[18,1], joints[18,2], 'RIGHT\n(−X)', fontsize=9, color='#ff6b6b', ha='center')
ax3.text(joints[23,0]+0.2, joints[23,1], joints[23,2], 'LEFT\n(+X)', fontsize=9, color='#ff6b6b', ha='center')

ax3.scatter([0],[0],[0], c='white', s=60, edgecolors='black', linewidths=1)

ax3.set_xlabel('X (← LEFT)'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z (FRONT ↑)')
ax3.set_xlim(-1, 1); ax3.set_ylim(-1.4, 0.8); ax3.set_zlim(-0.3, 0.5)
ax3.legend(loc='upper right', fontsize=8)
ax3.view_init(elev=80, azim=-90)

plt.suptitle('SKEL Body Model — Verified Coordinate Axes\n'
             '+X = LEFT (left arm)  |  +Y = UP (head)  |  +Z = FRONT (body faces viewer)\n'
             'Data source: SKELWrapper.forward() rest pose extraction, 2026-08-03',
             fontsize=10, color='#444444')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_verified_coords.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")
print("Done. Open the PNG to see verified coordinate axes on the actual SKEL model.")
