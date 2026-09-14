"""
虚拟相机可视化 — 侧视图 + 正视图
展示 HSMR 虚拟相机如何把 3D 人体投影到 256×256 图像平面。
Usage: source .venv/bin/activate && python docs/render_virtual_camera_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
import matplotlib.patches as mpatches

# ====================================================================
# PARAMETERS (from HSMR code)
# ====================================================================
FOCAL = 5000.0
PATCH = 256
CX = CY = PATCH / 2  # 128

# Typical camera_translation for a person in the patch
# scale ≈ 0.02 → tz ≈ 10000/(256*0.02) ≈ 1953
# But for visualization we use SCALED values
SCALE = 0.02
tx = 0.03
ty = -0.05
tz = 2 * FOCAL / (PATCH * SCALE)  # ≈ 1953

# Body is in camera coordinates: root at [tx, ty, tz]
# Body Y range: roughly [-1.2, +0.6], body height ≈ 1.8
# In camera space, head at Y ≈ ty + 0.6, feet at Y ≈ ty - 1.2

print(f"scale={SCALE}, tx={tx:.3f}, ty={ty:.3f}, tz={tz:.1f}")
print(f"Body root in camera coords: [{tx:.3f}, {ty:.3f}, {tz:.1f}]")

# ====================================================================
# FIGURE: Side view + Front view
# ====================================================================
fig, (ax_side, ax_front) = plt.subplots(1, 2, figsize=(16, 7))

# ---- SIDE VIEW (X-Z plane) ----
ax_side.set_title('虚拟相机 — 侧视图 (X-Z 平面)', fontsize=14, fontweight='bold')
ax_side.set_xlabel('Z (相机前方 →)', fontsize=11)
ax_side.set_ylabel('X (右 →)', fontsize=11)
ax_side.set_xlim(-200, 2200)
ax_side.set_ylim(-400, 400)
ax_side.grid(True, alpha=0.3)
ax_side.axhline(y=0, color='gray', linewidth=0.5)
ax_side.axvline(x=0, color='gray', linewidth=0.5)

# Camera at origin
ax_side.plot(0, 0, 's', color='#a371f7', markersize=14, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
ax_side.annotate('Camera\n原点 (0,0,0)', xy=(0, 0), xytext=(-180, 60),
                fontsize=10, color='#a371f7', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#a371f7', lw=1.5))

# Z axis
ax_side.arrow(0, -60, 2100, 0, head_width=20, head_length=40, fc='#4dabf7', ec='#4dabf7', linewidth=2)
ax_side.text(2150, -80, '+Z (相机前方)', fontsize=11, color='#4dabf7', fontweight='bold')

# Image plane (at Z = focal? No — the image plane in the conceptual diagram
# is at the patch distance. Actually the projection happens in normalized coords.
# For visualization, show the conceptual image plane)
# The "image plane" for the 256×256 patch is conceptually at Z = tz where body is.
# But in pinhole model, image plane is behind the lens.
# For simplicity, show the image plane near the body.

# Body stick figure (side view)
body_x = tx  # body root X in camera coords
body_z = tz  # body root Z in camera coords
body_height = 1.8  # approximate

# Head, thorax, pelvis, knees, feet (in camera Y, mapped to X axis in side view)
# Body: head at +0.6, feet at -1.2 relative to root
head_y = ty + 0.6
pelvis_y = ty
feet_y = ty - 1.2

# In side view, we see X (horizontal) vs Z (depth)
# Body segments
segments_y = [feet_y, ty - 0.7, ty, ty + 0.25, head_y]
segments_labels = ['脚', '膝', '骨盆', '胸', '头']

for i, sy in enumerate(segments_y):
    ax_side.plot(sy, body_z, 'o', color='#f78166', markersize=8, zorder=5)
    if i > 0:
        ax_side.plot([segments_y[i-1], sy], [body_z, body_z], '-', color='#f78166', linewidth=3)

# Body annotation
ax_side.annotate('人体 (侧视)\njoints_44_body\n+ camera_translation',
                xy=(ty, body_z), xytext=(-300, body_z - 300),
                fontsize=10, color='#f78166', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#f78166', lw=2))

# Projection rays from camera to image plane
image_plane_z = body_z + 100  # image plane right behind body for clarity
ax_side.axvline(x=image_plane_z, color='#b392f0', linewidth=3, alpha=0.5, linestyle='--')
ax_side.text(image_plane_z + 20, 350, '图像平面\n256×256', fontsize=9, color='#b392f0')

# Draw projection rays from camera to body extremities, then to image plane
for sy, label in [(head_y, '头顶'), (ty, '骨盆'), (feet_y, '脚底')]:
    # Ray from camera to body point
    ax_side.plot([0, sy], [0, body_z], ':', color='#d2991d', alpha=0.5, linewidth=1.5)
    # Projection to image plane: u = focal * X / Z
    u_proj = FOCAL * sy / body_z
    # Map to image plane: at image_plane_z, X coordinate = (sy/body_z) * (image_plane_z - 0) + 0
    # Actually: point on image plane at distance image_plane_z:
    # The ray from origin through (sy, body_z) hits Z=image_plane_z at X = sy * image_plane_z / body_z
    img_x = sy * image_plane_z / body_z
    ax_side.plot(img_x, image_plane_z, 'o', color='#d2991d', markersize=6)
    # Pixel coordinate on image
    u_pixel = FOCAL * sy / body_z + CX
    ax_side.annotate(f'{label}\nu={u_pixel:.0f}px',
                    xy=(img_x, image_plane_z), xytext=(img_x + 100, image_plane_z - 30),
                    fontsize=8, color='#d2991d')

# Camera_translation distance mark
ax_side.annotate('', xy=(0, body_z/2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='#ffd166', lw=2))
ax_side.text(-180, body_z/2, f'tz = {tz:.0f}\n(从 scale 反算)\n= 2×5000/(256×{SCALE})',
            fontsize=9, color='#ffd166', fontweight='bold', va='center')

# Camera_translation vector
ax_side.annotate('camera_translation\n= [tx, ty, tz]',
                xy=(tx/2, body_z/2), xytext=(-250, -150),
                fontsize=9, color='#ffd166', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffd166', lw=1.5))

# ---- FRONT VIEW (X-Y plane at body location) ----
ax_front.set_title('虚拟相机 — 正视图 (图像平面)', fontsize=14, fontweight='bold')
ax_front.set_xlabel('X → u (图像列)', fontsize=11)
ax_front.set_ylabel('Y → v (图像行, ↓)', fontsize=11)
ax_front.set_xlim(-200, 200)
ax_front.set_ylim(400, -100)  # inverted Y for image convention
ax_front.grid(True, alpha=0.3)

# Image plane border (256×256 scaled to fit)
# The image plane in "camera-normalized" coordinates
# At Z = tz, the visible X range is roughly [-tx_range, +tx_range]
# Actually, the 256×256 patch maps to camera X range of [-128*tz/5000, +128*tz/5000]
half_range = CX * tz / FOCAL
print(f"Camera X range at Z={tz:.0f}: [-{half_range:.1f}, +{half_range:.1f}]")

# Draw image plane as rectangle
img_rect = Rectangle((-half_range, -half_range), 2*half_range, 2*half_range,
                      fill=True, facecolor='#b392f0', alpha=0.1, edgecolor='#b392f0', linewidth=2)
ax_front.add_patch(img_rect)
ax_front.text(half_range*0.6, -half_range*0.9, '256×256 px\n图像平面',
             fontsize=10, color='#b392f0', fontweight='bold')

# Body joints projected onto image plane
# u = FOCAL * X_cam / Z_cam + CX
# X_cam = body_X + tx, where body_X is root-relative
# At body root: u_root = FOCAL * tx / tz + CX
u_root = FOCAL * tx / tz + CX
v_root = FOCAL * ty / tz + CX  # note: in image convention, larger v = lower

# Plot stick figure projected points
body_pts = [
    ('头', 0.0, 0.6, 0.0),
    ('胸', 0.0, 0.25, 0.0),
    ('骨盆', 0.0, 0.0, 0.0),
    ('L膝', 0.1, -0.5, 0.0),
    ('R膝', -0.1, -0.5, 0.0),
    ('L脚', 0.1, -1.2, 0.0),
    ('R脚', -0.1, -1.2, 0.0),
    ('L肘', 0.45, 0.2, 0.0),
    ('R肘', -0.45, 0.2, 0.0),
]

for label, bx, by, bz in body_pts:
    X_cam = bx + tx
    Y_cam = by + ty
    Z_cam = bz + tz
    u = FOCAL * X_cam / Z_cam + CX
    v = FOCAL * Y_cam / Z_cam + CX
    ax_front.plot(u - CX, v - CX, 'o', color='#f78166', markersize=8)
    ax_front.annotate(label, (u - CX + 5, v - CX + 5), fontsize=8, color='#f78166')

# Draw lines connecting body parts
limb_pairs = [
    (0, 1), (1, 2),  # head → chest → pelvis
    (2, 3), (3, 5),   # pelvis → L knee → L foot
    (2, 4), (4, 6),   # pelvis → R knee → R foot
    (1, 7), (1, 8),   # chest → L/R elbow
]
for i, j in limb_pairs:
    u1 = FOCAL * (body_pts[i][1] + tx) / tz
    v1 = FOCAL * (body_pts[i][2] + ty) / tz
    u2 = FOCAL * (body_pts[j][1] + tx) / tz
    v2 = FOCAL * (body_pts[j][2] + ty) / tz
    ax_front.plot([u1, u2], [v1, v2], '-', color='#f78166', linewidth=2)

# Image center crosshair
ax_front.axhline(y=0, color='#b392f0', linewidth=0.5, linestyle='--')
ax_front.axvline(x=0, color='#b392f0', linewidth=0.5, linestyle='--')
ax_front.plot(0, 0, '+', color='#b392f0', markersize=14, markeredgewidth=2)

# ====================================================================
# INSET: The actual projection math
# ====================================================================
formula_text = (
    f"投影公式（HSMR 虚拟相机）:\n"
    f"─────────────────────────\n"
    f"网络输出: camera = [scale={SCALE}, tx={tx:.2f}, ty={ty:.2f}]\n"
    f"tz = 2×{FOCAL:.0f} / ({PATCH}×{SCALE}+1e-9) = {tz:.0f}\n"
    f"camera_translation = [{tx:.2f}, {ty:.2f}, {tz:.0f}]\n"
    f"joints_44_camera = joints_44_body + camera_translation\n"
    f"\n"
    f"投影到 256×256 patch:\n"
    f"u = {FOCAL:.0f} × X_cam / Z_cam + {CX:.0f}\n"
    f"v = {FOCAL:.0f} × Y_cam / Z_cam + {CX:.0f}\n"
    f"\n"
    f"⚠️ 这不是物理相机！f={FOCAL:.0f} 是经验的，tz 不是米制深度。\n"
    f"虚拟相机的唯一作用：让 3D 人体投影与 2D 图像对齐。"
)

fig.text(0.5, 0.02, formula_text, ha='center', fontfamily='monospace', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', alpha=0.9),
         color='#c9d1d9', linespacing=1.4)

plt.tight_layout(rect=[0, 0.28, 1, 0.98])
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_virtual_camera_demo.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")
print("Open to see side view + front view of virtual camera projection.")
