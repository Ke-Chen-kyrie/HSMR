"""
HSMR + SKEL Blender Scene — matches pyrender output (real pose, real mesh, same colors)
"""
import bpy, json, math, os, struct, numpy as np
from mathutils import Vector, Matrix, Euler

# ── Clear scene ──
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for d in list(bpy.data.meshes): bpy.data.meshes.remove(d)
for d in list(bpy.data.materials): bpy.data.materials.remove(d)
for d in list(bpy.data.curves): bpy.data.curves.remove(d)

# ── Load data ──
with open('docs/_posed_meshes.bin', 'rb') as f:
    n_sv, n_sf, n_kv, n_kf, n_j = struct.unpack('IIIII', f.read(20))
    skin_v = np.frombuffer(f.read(n_sv*12), dtype=np.float32).reshape(n_sv, 3)
    skin_f = np.frombuffer(f.read(n_sf*12), dtype=np.uint32).reshape(n_sf, 3)
    skel_v = np.frombuffer(f.read(n_kv*12), dtype=np.float32).reshape(n_kv, 3)
    skel_f = np.frombuffer(f.read(n_kf*12), dtype=np.uint32).reshape(n_kf, 3)
    joints = np.frombuffer(f.read(n_j*12), dtype=np.float32).reshape(n_j, 3)

with open('docs/_posed_meta.json') as f:
    meta = json.load(f)
BONES = meta['bones']; NAMES = meta['joint_names']

print(f"Skin: {n_sv}v {n_sf}f | Skel: {n_kv}v {n_kf}f | Joints: {n_j}")

# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════
def mkcol(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c

def link(obj, col): col.objects.link(obj)

def get_principled(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED': return n
    return None

def material(name, rgba, emission=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = get_principled(m)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba[:3]+(1,)
        if rgba[3] < 1:
            bsdf.inputs['Alpha'].default_value = rgba[3]; m.blend_method = 'BLEND'
        if emission > 0:
            bsdf.inputs['Emission Color'].default_value = rgba[:3]+(1,)
            bsdf.inputs['Emission Strength'].default_value = emission
    return m

def mesh_obj(name, verts, faces, mat, coll):
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces]); m.update()
    obj = bpy.data.objects.new(name, m); obj.data.materials.append(mat); link(obj, coll)
    return obj

def arrow(name, origin, direction, length, mat, coll, radius=0.015):
    d = direction.normalized()
    # Cylinder
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=length*0.85, location=(0,0,0))
    c = bpy.context.active_object; c.name = f'{name}_cyl'
    c.rotation_mode = 'QUATERNION'; c.rotation_quaternion = Vector((0,0,1)).rotation_difference(d)
    c.location = Vector(origin) + d*(length*0.425); c.data.materials.append(mat); link(c, coll)
    # Cone
    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=radius*3, radius2=0, depth=length*0.15, location=(0,0,0))
    cn = bpy.context.active_object; cn.name = f'{name}_tip'
    cn.rotation_mode = 'QUATERNION'; cn.rotation_quaternion = Vector((0,0,1)).rotation_difference(d)
    cn.location = Vector(origin) + d*length; cn.data.materials.append(mat); link(cn, coll)

def text_label(name, text, loc, color, size=0.06, coll=None):
    td = bpy.data.curves.new(name, 'FONT'); to = bpy.data.objects.new(name, td)
    td.body = text; td.size = size; td.align_x = 'CENTER'; td.align_y = 'CENTER'
    to.location = Vector(loc); to.rotation_euler = Euler((math.radians(90),0,0))
    tm = material(f'{name}_m', color, emission=0.4)
    to.data.materials.append(tm); link(to, coll); return to

# ═══════════════════════════════════════
# COLLECTIONS
# ═══════════════════════════════════════
C_SKIN   = mkcol('01_Skin_Blue')
C_SKEL   = mkcol('02_Skeleton_Yellow')
C_JOINTS = mkcol('03_Joints')
C_AXES   = mkcol('04_Axes')
C_VCAM   = mkcol('05_Virtual_Camera')
C_VIEWS  = mkcol('06_Preset_Views')
C_HELP   = mkcol('07_Instructions')

# ═══════════════════════════════════════
# MATERIALS — match pyrender colors
# ═══════════════════════════════════════
# pyrender: skin='blue', skeleton='human_yellow'
# Blue in pyrender color palette: ~(0.1, 0.4, 0.9)
# Human yellow: ~(0.95, 0.85, 0.4)
M_SKIN   = material('Skin_Blue',    (0.15, 0.40, 0.90, 0.55))
M_SKEL   = material('Skel_Yellow',  (0.95, 0.82, 0.30, 1.0))
M_JOINT  = material('Joint',        (0.90, 0.35, 0.20, 1.0))
M_AX_R   = material('Axis_X',       (1.0, 0.15, 0.15, 1.0))
M_AX_G   = material('Axis_Y',       (0.15, 1.0, 0.25, 1.0))
M_AX_B   = material('Axis_Z',       (0.20, 0.40, 1.0, 1.0))
M_FWD    = material('Forward',      (1.0, 0.82, 0.0, 1.0))
M_ORIGIN = material('Origin',       (1.0, 1.0, 1.0, 1.0))
M_CAM    = material('VCam',         (0.64, 0.44, 0.97, 1.0))
M_PLANE  = material('ImgPlane',     (0.70, 0.55, 0.94, 0.30))
M_FRUSTUM= material('Frustum',      (0.70, 0.50, 1.0, 1.0))
M_LABEL  = material('Label',        (1.0, 1.0, 1.0, 1.0))

# ═══════════════════════════════════════
# 1. SKIN MESH (blue, semi-transparent — matching pyrender)
# ═══════════════════════════════════════
mesh_obj('Skin', skin_v, skin_f, M_SKIN, C_SKIN)

# ═══════════════════════════════════════
# 2. SKELETON MESH (yellow — actual bone geometry from SKEL!)
# ═══════════════════════════════════════
mesh_obj('Skeleton', skel_v, skel_f, M_SKEL, C_SKEL)

# ═══════════════════════════════════════
# 3. JOINTS
# ═══════════════════════════════════════
for i, pos in enumerate(joints):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=tuple(pos))
    s = bpy.context.active_object; s.name = f'J_{NAMES[i]}'; s.data.materials.append(M_JOINT); link(s, C_JOINTS)

# Joint labels
LABEL_J = {0:'pelvis', 13:'head', 5:'toes-R', 10:'toes-L', 18:'hand-R', 23:'hand-L', 15:'shoulder-R', 20:'shoulder-L'}
for jid, name in LABEL_J.items():
    text_label(f'JLB_{name}', name, joints[jid]+[0,0.05,0.07], (1,0.7,0.5,1), 0.05, C_JOINTS)

# ═══════════════════════════════════════
# 4. COORDINATE AXES (at pelvis)
# ═══════════════════════════════════════
P = Vector(joints[0])
arrow('AX_X', P, Vector((1,0,0)), 0.8, M_AX_R, C_AXES)
arrow('AX_Y', P, Vector((0,1,0)), 0.8, M_AX_G, C_AXES)
arrow('AX_Z', P, Vector((0,0,1)), 0.8, M_AX_B, C_AXES)
arrow('FWD',  P, Vector((0,0,1)), 1.3, M_FWD,  C_AXES, radius=0.022)

text_label('LB_X', '+X = LEFT',  P+Vector((1.0,0,0)),    (1,0.2,0.2,1), 0.07, C_AXES)
text_label('LB_Y', '+Y = UP',    P+Vector((0,1.05,0)),   (0.2,1,0.2,1), 0.07, C_AXES)
text_label('LB_Z', '+Z = FORWARD',P+Vector((0,0,1.05)),  (0.2,0.4,1,1), 0.07, C_AXES)
text_label('LB_FW','BODY FACES +Z',P+Vector((0,-0.15,1.4)),(1,0.82,0,1),0.08, C_AXES)

# Origin (mesh centroid = (0,0,0))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(0,0,0))
os_ = bpy.context.active_object; os_.name = 'Origin'; os_.data.materials.append(M_ORIGIN); link(os_, C_AXES)
# Thin origin axes
for d, m in [(Vector((1,0,0)), M_AX_R),(Vector((0,1,0)), M_AX_G),(Vector((0,0,1)), M_AX_B)]:
    arrow('O'+str(d[0]), Vector((0,0,0)), d, 0.3, m, C_AXES, radius=0.007)

# ═══════════════════════════════════════
# 5. HSMR VIRTUAL CAMERA
# ═══════════════════════════════════════
cam_loc = Vector((0.05, -0.1, -3.0))

# Camera body
bpy.ops.mesh.primitive_cube_add(size=1, location=cam_loc)
cb = bpy.context.active_object; cb.name = 'VCam'; cb.scale = (0.12,0.08,0.15); cb.data.materials.append(M_CAM); link(cb, C_VCAM)
# Lens
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.08, location=cam_loc+Vector((0,0,0.12)))
ln = bpy.context.active_object; ln.name = 'VCam_Lens'; ln.rotation_euler = Euler((math.pi/2,0,0)); ln.data.materials.append(M_CAM); link(ln, C_VCAM)

# Image plane
ip_d = 1.0; ip_h = 0.5
bpy.ops.mesh.primitive_plane_add(size=ip_h*2, location=cam_loc+Vector((0,0,ip_d)))
ip = bpy.context.active_object; ip.name = 'ImagePlane_256x256'; ip.data.materials.append(M_PLANE); link(ip, C_VCAM)

# Frustum edges
for cx,cy in [(-ip_h,-ip_h),(ip_h,-ip_h),(ip_h,ip_h),(-ip_h,ip_h)]:
    tgt = cam_loc + Vector((cx,cy,ip_d))
    m = bpy.data.meshes.new('Fe'); m.from_pydata([cam_loc, tgt], [(0,1)], []); m.update()
    fe = bpy.data.objects.new('Fe', m); fe.data.materials.append(M_FRUSTUM); link(fe, C_VCAM)

text_label('L_CAM', 'HSMR Virtual Camera', cam_loc+Vector((0,0.2,0)), (0.7,0.5,1,1), 0.06, C_VCAM)
text_label('L_IP', 'Image Plane 256x256', cam_loc+Vector((0,0.4,ip_d)), (0.7,0.5,1,1), 0.05, C_VCAM)

# ═══════════════════════════════════════
# 6. PRESET VIEW CAMERAS (no numpad needed)
# ═══════════════════════════════════════
def view_cam(name, loc, rot):
    cd = bpy.data.cameras.new(name); co = bpy.data.objects.new(name, cd)
    co.location = loc; co.rotation_euler = Euler(rot, 'XYZ')
    bpy.context.collection.objects.link(co); link(co, C_VIEWS)
    co.show_name = True; co.data.display_size = 0.3
    return co

vf = view_cam('VIEW_Front_面对你',  (0.0,-0.2,5.0),  (math.radians(90),0,0))
vs = view_cam('VIEW_Right_侧面',   (5.0,-0.2,0.1),  (math.radians(90),0,math.radians(90)))
vt = view_cam('VIEW_Top_俯视',     (0.0,5.0,0.1),   (0,0,0))
va = view_cam('VIEW_3Quarter',     (2.8,1.4,4.5),   (math.radians(72),0,math.radians(58)))
bpy.context.scene.camera = vf  # default to front view

# ═══════════════════════════════════════
# 7. INSTRUCTIONS (on screen)
# ═══════════════════════════════════════
help_lines = [
    "=== BLENDER 操作 (无需数字键盘) ===",
    "切换视角: 右侧Outliner → 06_Preset_Views → 双击名字",
    "  VIEW_Front_面对你 → 正面(人体面对你,+Z方向)",
    "  VIEW_Right_侧面  → 侧面(从左侧看)",
    "  VIEW_Top_俯视    → 俯视",
    "旋转: 鼠标中键拖拽  缩放: 滚轮  平移: Shift+中键",
    "显示/隐藏: Outliner中点击 Collection 的眼睛图标",
    "",
    "颜色: 蓝色=皮肤(半透明)  黄色=骨骼网格(SKEL真实几何)",
    "  红色箭头=+X(左) 绿色=+Y(上) 蓝色=+Z(前)",
    "  金色粗箭头=人体前方向量  白色球=SKEL原点",
    "  紫色方块=HSMR虚拟相机  紫色线框=视锥体",
]
for i, line in enumerate(help_lines):
    text_label(f'HLP{i}', line, Vector((-1.5, 2.5-i*0.09, 3.0)), (1,1,1,1), 0.045, C_HELP)

# ═══════════════════════════════════════
# 8. LIGHTING & BACKGROUND
# ═══════════════════════════════════════
bpy.ops.object.light_add(type='SUN', location=(8,12,6)); sun=bpy.context.active_object; sun.name='Sun'; sun.data.energy=4
bpy.ops.object.light_add(type='AREA', location=(-4,2,2)); fill=bpy.context.active_object; fill.name='Fill'; fill.data.energy=80; fill.data.size=6
bpy.ops.object.light_add(type='AREA', location=(0,-2,3)); rim=bpy.context.active_object; rim.name='Rim'; rim.data.energy=40; rim.data.size=4

world = bpy.data.worlds['World']; world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.08,0.09,0.14,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.4

# Viewport
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for sp in area.spaces:
            if sp.type == 'VIEW_3D':
                sp.shading.type = 'MATERIAL'; sp.clip_start = 0.01; sp.clip_end = 50

# ═══════════════════════════════════════
# SAVE
# ═══════════════════════════════════════
bpy.ops.wm.save_as_mainfile(filepath='docs/hsmr_scene.blend')
print("Saved: docs/hsmr_scene.blend")
print("Open with: bash docs/open_blender_scene.sh")
