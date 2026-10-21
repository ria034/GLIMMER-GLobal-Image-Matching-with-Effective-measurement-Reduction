import open3d as o3d
import numpy as np
import os
import struct

# Load point cloud (agar .bin se .ply banaya hai toh wo use kar, ya direct open3d support dekho)
ply_path = r"C:\Users\lenovo\Desktop\repos\GLIMMER-GLobal-Image-Matching-with-Effective-measurement-Reduction\outputs\0\gerrad_hall_output"
if not os.path.exists(ply_path):
    print("Pehle points3D.bin ko ply mein convert kar le jo upar diya tha!")
    exit()

print("Loading point cloud...")
pcd = o3d.io.read_point_cloud(ply_path)
import numpy as np
import open3d as o3d
import os

# --- 1. COLMAP Binary Readers ---
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4) # Simple mapping
            params = read_next_bytes(fid, num_params * 8, f"{num_params}d")
            cameras[camera_id] = (model_id, width, height, params)
    return cameras

def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    rot_mat = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return rot_mat

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = read_next_bytes(fid, 32, "dddd")
            tvec = read_next_bytes(fid, 24, "ddd")
            camera_id = read_next_bytes(fid, 4, "i")[0]
            name = ""
            while True:
                char = fid.read(1)
                if char == b"\x00": break
                name += char.decode("utf-8")
            
            # Read points
            num_points = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points * 24) # Skip x, y, point3d_id
            
            images[image_id] = (qvec, tvec, camera_id, name)
    return images

def read_points3d_binary(path):
    xyzs, rgbs = [], []
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_header = read_next_bytes(fid, 43, "QdddB B Bd")
            xyz = np.array(binary_point_line_header[1:4])
            rgb = np.array(binary_point_line_header[4:7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.array(xyzs), np.array(rgbs)

# --- 2. Load Data from outputs/0/ ---
model_dir = r"C:\Users\lenovo\Desktop\repos\GLIMMER-GLobal-Image-Matching-with-Effective-measurement-Reduction\outputs\0\gerrad_hall_output\0"
if not os.path.exists(os.path.join(model_dir, "points3D.bin")):
    print("Error: Make sure 'outputs/0/' folder contains cameras.bin, images.bin, and points3D.bin!")
    exit()

print("Loading COLMAP binary files...")
points, colors = read_points3d_binary(os.path.join(model_dir, "points3D.bin"))
images = read_images_binary(os.path.join(model_dir, "images.bin"))

# --- 3. Setup Open3D Visualization ---
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="GLOMAP 3D Reconstruction & Poses", width=1280, height=720)

# Add Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
vis.add_geometry(pcd)

# Add Camera Frustums / Coordinate Frames
print("Adding camera poses to 3D scene...")
for image_id, (qvec, tvec, camera_id, name) in images.items():
    R = qvec_to_rotmat(qvec)
    t = np.array(tvec)
    # Compute camera position in world coordinates: C = -R^T * t
    c_pos = -R.T @ t
    
    # Create a small coordinate frame or camera pyramid for each pose
    # Scale down the frame size so it doesn't crowd the point cloud
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # Transform to world pose
    cam_frame.rotate(R.T, center=(0, 0, 0))
    cam_frame.translate(c_pos)
    vis.add_geometry(cam_frame)

opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])
opt.point_size = 2.0

print("\n--- Controls ---")
print("Press 'R' in the Open3D window to start/stop automatic rotation recording.")
print("Frames will be saved in 'anim_frames' folder automatically.")

os.makedirs("anim_frames", exist_ok=True)
frame_count = [0]
is_recording = [False]

def rotate_view(vis):
    if is_recording[0]:
        ctr = vis.get_view_control()
        ctr.rotate(5.0, 0.0)
        vis.poll_events()
        vis.update_render()
        
        filename = f"anim_frames/frame_{frame_count[0]:04d}.png"
        vis.capture_screen_image(filename)
        frame_count[0] += 1
        if frame_count[0] > 150:
            is_recording[0] = False
            print(f"Finished capturing {frame_count[0]} frames in 'anim_frames/' folder!")
    return False

def toggle_recording(vis):
    is_recording[0] = not is_recording[0]
    print("Recording started..." if is_recording[0] else "Recording stopped.")
    return False

vis.register_key_callback(ord("R"), toggle_recording)
vis.register_animation_callback(rotate_view)

vis.run()
vis.destroy_window()

# Visualizer with automatic recorder
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="GLOMAP 3D Animation", width=1280, height=720)
vis.add_geometry(pcd)

# Optional: Background color dark ya light karne ke liye
opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05]) # Dark background thumbnail ke liye best lagta hai
opt.point_size = 2.0

print("\n--- Controls ---")
print("Press 'R' in the window to start/stop automatic turntable rotation recording.")
print("Frames will be saved in an 'anim_frames' folder automatically.")

os.makedirs("anim_frames", exist_ok=True)
frame_count = [0]
is_recording = [False]

def rotate_view(vis):
    if is_recording[0]:
        ctr = vis.get_view_control()
        ctr.rotate(5.0, 0.0) # Camera ko thoda rotate karo har frame par
        vis.poll_events()
        vis.update_render()
        
        # Frame capture
        filename = f"anim_frames/frame_{frame_count[0]:04d}.png"
        vis.capture_screen_image(filename)
        frame_count[0] += 1
        if frame_count[0] > 120: # 120 frames (approx 4 seconds at 30fps)
            is_recording[0] = False
            print(f"Finished capturing {frame_count[0]} frames in 'anim_frames/' folder!")
    return False

def toggle_recording(vis):
    is_recording[0] = not is_recording[0]
    print("Recording started..." if is_recording[0] else "Recording stopped.")
    return False

vis.register_key_callback(ord("R"), toggle_recording)
vis.register_animation_callback(rotate_view)

vis.run()
vis.destroy_window()                