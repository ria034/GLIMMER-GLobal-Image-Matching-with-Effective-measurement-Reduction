"""
View and compare COLMAP / GLOMAP reconstructions.

Usage:
  python view_models.py outputs\\south\\100_budget\\0
  python view_models.py outputs\\south\\100_budget\\0 outputs\\south\\50_budget\\0 outputs\\south\\25_budget\\0
  python view_models.py MODEL_A MODEL_B --rgb          # keep original point colors
  python view_models.py MODEL_A --record               # save turntable frames
  python view_models.py A B --dataset South_building --backbones MST MST --budgets 100 30

Labels (dataset, backbone, budget) are read from the folder names, e.g.
  outputs\\South_building\\MST\\30_MST\\0  ->  South_building | MST | 30%
and can be overridden with --dataset / --backbones / --budgets, or per model in
DEFAULT_MODELS. They appear in the window title, the console tables, and on
every saved image (press S for a snapshot, R to record with --record).

With 2+ models, every model is aligned to the FIRST one (similarity transform
on camera centers of shared images) and drawn in its own color, and the pose
differences are printed. Global SfM has an arbitrary scale/rotation/origin,
so models must be aligned before they can be compared.
"""

import argparse
import os
import re
import struct

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# EDIT THIS: models used when the script is run without arguments
# (e.g. with the Run button). The first one is the reference.
# ---------------------------------------------------------------------------
REPO = r"C:\Users\lenovo\Desktop\repos\GLIMMER-GLobal-Image-Matching-with-Effective-measurement-Reduction"
# Each entry is a path, or a dict with "path" and any of "dataset",
# "backbone", "budget" to override what is read from the folder names.
DEFAULT_MODELS = [
    os.path.join(REPO, r"outputs\South_building\MST\100_MST\0"),
    {"path": os.path.join(REPO, r"outputs\South_building\MST\30_MST\0"),
     "backbone": "MST", "budget": 30},
]

MODEL_COLORS = [
    (0.90, 0.90, 0.90),  # first model: light grey
    (0.95, 0.35, 0.25),  # red
    (0.25, 0.65, 0.95),  # blue
    (0.35, 0.85, 0.40),  # green
    (0.95, 0.75, 0.25),  # yellow
    (0.75, 0.40, 0.90),  # purple
]


# ---------------------------------------------------------------------------
# Labels: dataset / backbone / budget
# ---------------------------------------------------------------------------
NON_BACKBONE_TOKENS = {"budget", "b", "bud", "ratio", "r", "model", "output", "outputs"}


def parse_labels(path):
    """Guess dataset, backbone and budget from a model path such as
    outputs\\South_building\\MST\\30_MST\\0."""
    parts = [p for p in re.split(r"[\\/]+", os.path.normpath(path)) if p]
    if parts and parts[-1].isdigit():  # glomap's "0" sub-model folder
        parts = parts[:-1]
    name = parts[-1] if parts else ""

    # Budget: first number in the model folder name ("30_MST", "b30", "0.3").
    budget = None
    m = re.search(r"\d+(?:\.\d+)?", name)
    if m:
        value = float(m.group())
        budget = value * 100 if ("." in m.group() and value <= 1.0) else value

    # Backbone: the words in the model folder name, minus budget-like words;
    # otherwise the parent folder (e.g. ...\MST\30\0).
    words = [w for w in re.split(r"[^A-Za-z]+", name)
             if w and w.lower() not in NON_BACKBONE_TOKENS]
    backbone = "_".join(words) if words else None
    if backbone is None and len(parts) >= 2 and parts[-2].lower() not in NON_BACKBONE_TOKENS:
        backbone = parts[-2]

    # Dataset: the folder right after "outputs", if any.
    dataset = None
    lower = [p.lower() for p in parts]
    if "outputs" in lower:
        i = lower.index("outputs")
        if i + 1 < len(parts) - 1:
            dataset = parts[i + 1]
    return {"dataset": dataset, "backbone": backbone, "budget": budget}


def format_budget(budget):
    if budget is None:
        return "?"
    return f"{budget:g}%"


def model_label(lbl, with_dataset=False):
    text = f"{lbl['backbone'] or '?'} | budget {format_budget(lbl['budget'])}"
    if with_dataset:
        text = f"{lbl['dataset'] or '?'} | " + text
    return text


def save_annotated(path, title, legend):
    """Draw a title and a color legend onto a saved screenshot (needs Pillow)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    pad = 12
    lines = [(title, None, font)] + [(text, color, small) for text, color in legend]
    heights = [draw.textbbox((0, 0), t, font=f)[3] + 8 for t, _, f in lines]
    widths = [draw.textbbox((0, 0), t, font=f)[2] + (26 if c else 0) for t, c, f in lines]
    box = (pad, pad, pad + max(widths) + 2 * pad, pad + sum(heights) + pad)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    y = 2 * pad
    for (text, color, f), h in zip(lines, heights):
        x = 2 * pad
        if color is not None:
            rgb = tuple(int(255 * c) for c in color)
            draw.rectangle((x, y + 3, x + 16, y + 17), fill=rgb)
            x += 26
        draw.text((x, y), text, fill=(255, 255, 255), font=f)
        y += h
    img.save(path)


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------
def read_bytes(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def qvec_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def read_images_binary(path):
    """Returns {name: (R_world_to_cam, camera_center)}."""
    images = {}
    with open(path, "rb") as fid:
        num_images = read_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            read_bytes(fid, 4, "i")  # image_id
            qvec = read_bytes(fid, 32, "dddd")
            tvec = np.array(read_bytes(fid, 24, "ddd"))
            read_bytes(fid, 4, "i")  # camera_id
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2d = read_bytes(fid, 8, "Q")[0]
            fid.read(num_points2d * 24)  # x, y (double), point3D_id (int64)
            R = qvec_to_rotmat(qvec)
            images[name.decode("utf-8")] = (R, -R.T @ tvec)
    return images


def read_points3d_binary(path):
    """Returns (xyz Nx3, rgb Nx3 in [0,1], track_lengths N)."""
    xyzs, rgbs, track_lengths = [], [], []
    with open(path, "rb") as fid:
        num_points = read_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            # point3D_id (Q), xyz (3d), rgb (3B), error (d) = 43 bytes
            data = read_bytes(fid, 43, "QdddBBBd")
            xyzs.append(data[1:4])
            rgbs.append(data[4:7])
            track_len = read_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_len)  # image_id (int32), point2D_idx (int32)
            track_lengths.append(track_len)
    return (np.array(xyzs, dtype=float),
            np.array(rgbs, dtype=float) / 255.0,
            np.array(track_lengths))


def load_model(model_dir):
    for f in ("images.bin", "points3D.bin"):
        if not os.path.exists(os.path.join(model_dir, f)):
            raise FileNotFoundError(f"{f} not found in {model_dir}")
    images = read_images_binary(os.path.join(model_dir, "images.bin"))
    points, colors, tracks = read_points3d_binary(
        os.path.join(model_dir, "points3D.bin"))
    return images, points, colors, tracks


# ---------------------------------------------------------------------------
# Alignment (Umeyama similarity transform: dst ~= s * R @ src + t)
# ---------------------------------------------------------------------------
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def rotation_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def compare_to_reference(ref_images, images):
    common = sorted(set(ref_images) & set(images))
    if len(common) < 3:
        return None
    src = np.array([images[n][1] for n in common])
    dst = np.array([ref_images[n][1] for n in common])
    s, R, t = umeyama(src, dst)
    pos_err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    rot_err = np.array([
        rotation_angle_deg(ref_images[n][0] @ (images[n][0] @ R.T).T)
        for n in common
    ])
    scene_size = np.linalg.norm(dst - dst.mean(0), axis=1).max()
    return {
        "transform": (s, R, t),
        "common": len(common),
        "pos_median": np.median(pos_err),
        "pos_max": pos_err.max(),
        "pos_median_pct": 100 * np.median(pos_err) / max(scene_size, 1e-12),
        "rot_median": np.median(rot_err),
        "rot_max": rot_err.max(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model folders (containing images.bin, points3D.bin); "
                        "defaults to DEFAULT_MODELS at the top of this file")
    parser.add_argument("--rgb", action="store_true", help="use original point colors instead of one color per model")
    parser.add_argument("--no-cameras", action="store_true", help="hide camera frames")
    parser.add_argument("--record", action="store_true", help="enable 'R' key turntable frame recording")
    parser.add_argument("--frames", type=int, default=150, help="frames to record")
    parser.add_argument("--dataset", help="dataset name for all models (overrides folder names)")
    parser.add_argument("--backbones", nargs="+", help="backbone label per model, in order")
    parser.add_argument("--budgets", nargs="+", type=float, help="budget in %% per model, in order")
    parser.add_argument("--out-dir", default="anim_frames", help="folder for snapshots / recorded frames")
    args = parser.parse_args()
    entries = args.models or DEFAULT_MODELS

    models, labels = [], []
    for k, entry in enumerate(entries):
        path = entry["path"] if isinstance(entry, dict) else entry
        lbl = parse_labels(path)
        if isinstance(entry, dict):
            lbl.update({key: entry[key] for key in ("dataset", "backbone", "budget") if key in entry})
        if args.dataset:
            lbl["dataset"] = args.dataset
        if args.backbones and k < len(args.backbones):
            lbl["backbone"] = args.backbones[k]
        if args.budgets and k < len(args.budgets):
            lbl["budget"] = args.budgets[k]
        print(f"Loading {path} ...")
        try:
            models.append((path, *load_model(path)))
            labels.append(lbl)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
        except (struct.error, UnicodeDecodeError, MemoryError) as e:
            print(f"  FAILED to parse ({type(e).__name__}: {e}). "
                  f"The binary format may differ from what this reader expects.")
    if not models:
        print("No model could be loaded. Check the paths in DEFAULT_MODELS or on the command line.")
        return

    # --- Statistics ---
    datasets = sorted({l["dataset"] or "?" for l in labels})
    dataset_title = " / ".join(datasets)

    print("\n" + "=" * 86)
    print(f"Dataset: {dataset_title}")
    print("-" * 86)
    print(f"{'Dataset':18s} {'Backbone':14s} {'Budget':>8s} {'Images':>8s} {'Points':>9s} {'Mean track':>11s}")
    print("-" * 86)
    for (path, images, points, _, tracks), lbl in zip(models, labels):
        print(f"{(lbl['dataset'] or '?')[:18]:18s} {(lbl['backbone'] or '?')[:14]:14s} "
              f"{format_budget(lbl['budget']):>8s} {len(images):8d} {len(points):9d} "
              f"{tracks.mean() if len(tracks) else 0:11.2f}")
    print("=" * 86)

    # --- Alignment and pose differences vs. first model ---
    ref_path, ref_images = models[0][0], models[0][1]
    transforms = [(1.0, np.eye(3), np.zeros(3))]
    if len(models) > 1:
        print(f"\nPose differences vs. reference: {model_label(labels[0], True)}")
        print(f"{'Model':40s} {'Shared':>7s} {'Pos med':>9s} {'(% size)':>9s} "
              f"{'Pos max':>9s} {'Rot med°':>9s} {'Rot max°':>9s}")
        print("-" * 98)
        for (path, images, *_), lbl in zip(models[1:], labels[1:]):
            label = model_label(lbl, len(datasets) > 1)[:40]
            res = compare_to_reference(ref_images, images)
            if res is None:
                print(f"{label:40s}  fewer than 3 shared images, cannot align")
                transforms.append((1.0, np.eye(3), np.zeros(3)))
                continue
            transforms.append(res["transform"])
            print(f"{label:40s} {res['common']:7d} {res['pos_median']:9.4f} "
                  f"{res['pos_median_pct']:8.2f}% {res['pos_max']:9.4f} "
                  f"{res['rot_median']:9.3f} {res['rot_max']:9.3f}")
        print("(positions in the reference model's units; identical models give ~0)")

        # Which images differ between each model and the reference
        print(f"\nImage differences vs. reference ({len(ref_images)} images):")
        for (path, images, *_), lbl in zip(models[1:], labels[1:]):
            label = model_label(lbl, len(datasets) > 1)
            missing = sorted(set(ref_images) - set(images))
            extra = sorted(set(images) - set(ref_images))
            print(f"  {label}")
            print(f"    missing ({len(missing)}): {', '.join(missing) if missing else '-'}")
            if extra:
                print(f"    extra   ({len(extra)}): {', '.join(extra)}")

    # --- Visualization ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    color_names = ["grey", "red", "blue", "green", "yellow", "purple"]
    legend = []
    for i, lbl in enumerate(labels):
        color = MODEL_COLORS[i % len(MODEL_COLORS)] if not args.rgb else (1.0, 1.0, 1.0)
        text = model_label(lbl, len(datasets) > 1)
        if i == 0 and len(models) > 1:
            text += "  (reference)"
        legend.append((text, color))
    window_title = f"{dataset_title} | " + "  vs  ".join(
        (f"{color_names[i % len(color_names)]}: " if len(models) > 1 and not args.rgb else "")
        + model_label(lbl) for i, lbl in enumerate(labels))
    print(f"\nWindow: {window_title}")
    for text, color in legend:
        print(f"  {text}")

    vis.create_window(window_name=window_title, width=1280, height=720)

    ref_centers = np.array([c for _, c in ref_images.values()])
    frame_size = 0.03 * np.linalg.norm(ref_centers - ref_centers.mean(0), axis=1).max()

    for i, ((path, images, points, colors, _), (s, R, t)) in enumerate(zip(models, transforms)):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((s * (R @ points.T)).T + t)
        if args.rgb:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color(color)
        vis.add_geometry(pcd)

        if not args.no_cameras:
            for R_wc, c in images.values():
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
                frame.rotate(R @ R_wc.T, center=(0, 0, 0))
                frame.translate(s * R @ c + t)
                if len(models) > 1:
                    frame.paint_uniform_color(color)
                vis.add_geometry(frame)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 2.0

    os.makedirs(args.out_dir, exist_ok=True)
    title_text = f"{dataset_title}"
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                 f"{dataset_title}_" + "_vs_".join(
                     f"{l['backbone'] or 'x'}{format_budget(l['budget']).rstrip('%')}" for l in labels))
    state = {"frame": 0, "recording": False, "snap": 0}

    def snapshot(v):
        path = os.path.join(args.out_dir, f"{tag}_snapshot_{state['snap']:02d}.png")
        v.capture_screen_image(path, do_render=True)
        save_annotated(path, title_text, legend)
        state["snap"] += 1
        print(f"Saved {path}")
        return False

    vis.register_key_callback(ord("S"), snapshot)
    print("Press 'S' in the window to save a labelled snapshot.")

    if args.record:
        def toggle(v):
            state["recording"] = not state["recording"]
            print("Recording started..." if state["recording"] else "Recording stopped.")
            return False

        def step(v):
            if state["recording"]:
                v.get_view_control().rotate(5.0, 0.0)
                v.poll_events()
                v.update_render()
                path = os.path.join(args.out_dir, f"{tag}_frame_{state['frame']:04d}.png")
                v.capture_screen_image(path)
                save_annotated(path, title_text, legend)
                state["frame"] += 1
                if state["frame"] >= args.frames:
                    state["recording"] = False
                    print(f"Saved {state['frame']} labelled frames to {args.out_dir}/")
            return False

        vis.register_key_callback(ord("R"), toggle)
        vis.register_animation_callback(step)
        print("Press 'R' in the window to start/stop recording.")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()