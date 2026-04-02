
import bpy
import numpy as np

# Clear any stale handlers first
bpy.app.handlers.frame_change_pre.clear()

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

frames = np.load("C:/minimal_surface/neck_pinch_frames.npy")
num_frames, rows, cols, _ = frames.shape

verts = frames[0].reshape(-1, 3).tolist()
faces = []
for r in range(rows - 1):
    for c in range(cols - 1):
        i = r * cols + c
        faces.append((i, i + 1, i + cols + 1, i + cols))
for c in range(cols - 1):
    i_last = (rows - 1) * cols + c
    i_first = c
    faces.append((i_last, i_last + 1, i_first + 1, i_first))

mesh = bpy.data.meshes.new("NeckPinch")
mesh.from_pydata(verts, [], faces)
mesh.update()

obj = bpy.data.objects.new("NeckPinch", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

for poly in obj.data.polygons:
    poly.use_smooth = True
obj.data.update()

# ── Material ───────────────────────────────────────────────────────────────
mat = bpy.data.materials.new("CurvatureMat")
mat.use_nodes = True
mat.use_backface_culling = False
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial")
output.location = (800, 0)

bsdf = nodes.new("ShaderNodeBsdfPrincipled")
bsdf.location = (500, 0)
bsdf.inputs["Roughness"].default_value = 0.2
bsdf.inputs["Metallic"].default_value = 0.2
try:
    bsdf.inputs["Specular"].default_value = 0.8
except KeyError:
    bsdf.inputs["IOR"].default_value = 1.5

ramp = nodes.new("ShaderNodeValToRGB")
ramp.location = (100, 0)
ramp.color_ramp.interpolation = 'B_SPLINE'
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0.017, 0.314, 0.780, 1.0)
ramp.color_ramp.elements[1].position = 1.0
ramp.color_ramp.elements[1].color = (0.780, 0.094, 0.094, 1.0)
mid = ramp.color_ramp.elements.new(0.5)
mid.color = (1.0, 1.0, 1.0, 1.0)

tex_coord = nodes.new("ShaderNodeTexCoord")
tex_coord.location = (-600, 0)
separate_xyz = nodes.new("ShaderNodeSeparateXYZ")
separate_xyz.location = (-400, 0)
map_range = nodes.new("ShaderNodeMapRange")
map_range.location = (-100, 0)
map_range.inputs["From Min"].default_value = -1.0
map_range.inputs["From Max"].default_value = 1.0
map_range.inputs["To Min"].default_value = 0.0
map_range.inputs["To Max"].default_value = 1.0
map_range.clamp = True

links.new(tex_coord.outputs["Generated"], separate_xyz.inputs["Vector"])
links.new(separate_xyz.outputs["Z"], map_range.inputs["Value"])
links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
obj.data.materials.append(mat)

# ── Camera and light ───────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(5, -5, 3))
cam = bpy.context.object
cam.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = cam

bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
bpy.context.object.data.energy = 3.0

# ── Render settings ────────────────────────────────────────────────────────
scene = bpy.context.scene
fps = 24
scene.frame_start = 1
scene.frame_end = num_frames * fps
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 50

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for device in prefs.devices:
    device.use = True

scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.filepath = "C:/minimal_surface/neck_pinch_final.mp4"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

# ── Register handler LAST, just before rendering ───────────────────────────
def update_mesh(scene):
    frame = scene.frame_current
    frame_idx = min(int((frame - 1) / fps), num_frames - 1)
    obj = bpy.data.objects.get("NeckPinch")
    if obj is None:
        return
    flat = frames[frame_idx].reshape(-1, 3)
    for i, coord in enumerate(flat):
        obj.data.vertices[i].co = coord
    obj.data.update()

bpy.app.handlers.frame_change_pre.append(update_mesh)
print("Handler registered — starting render")

bpy.ops.render.render(animation=True)
print("Render complete")
