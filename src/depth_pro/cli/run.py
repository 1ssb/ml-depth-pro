#!/usr/bin/env python3
"""
DepthPro inference script: display point clouds or raw depth maps with optional half-precision.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import argparse
import logging
import numpy as np
import PIL.Image
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Optional

from depth_pro import create_model_and_transforms
from depth_pro.depth2cloud import depth2cloud

# Attempt to import open3d for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

LOGGER = logging.getLogger("depth-pro-run")

# Global caches
MODEL_CACHE = {}
IMAGE_CACHE = {}
DEPTH_CACHE = {}
CLOUD_CACHE = {}

def torch_device() -> torch.device:
    """Pick the best available Torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_logging(verbose: bool) -> None:
    """Configure logging level."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)


def gather_image_paths(base: Path) -> tuple[list[Path], Path]:
    """Gather image paths from a directory or single file."""
    if base.is_dir():
        image_paths = sorted(p for p in base.glob("**/*") if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        rel_root = base
    else:
        image_paths = [base]
        rel_root = base.parent
    return image_paths, rel_root


def get_cached_model(device: torch.device, precision: torch.dtype):
    """Cache and return model and transform for given device/precision."""
    key = (str(device), str(precision))
    if key not in MODEL_CACHE:
        LOGGER.info("Loading model and transform...")
        model, transform = create_model_and_transforms(device=device, precision=precision)
        model.eval()
        MODEL_CACHE[key] = (model, transform)
    return MODEL_CACHE[key]


def get_cached_image(path: Path):
    """Cache and return loaded image."""
    abs_path = str(path.absolute())
    if abs_path not in IMAGE_CACHE:
        try:
            IMAGE_CACHE[abs_path] = PIL.Image.open(path).convert("RGB")
        except Exception as e:
            LOGGER.error(f"Failed to load {path}: {e}")
            return None
    return IMAGE_CACHE[abs_path]


def get_cached_depth(path: Path, model, transform, device: torch.device):
    """Cache and return depth prediction."""
    abs_path = str(path.absolute())
    if abs_path not in DEPTH_CACHE:
        image = get_cached_image(path)
        if image is None:
            return None
        
        # Run inference
        inp = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(inp)
        
        # Extract and store results
        depth = pred["depth"].cpu().numpy().squeeze()
        f_px_tensor = pred.get("focallength_px", None)
        f_px = f_px_tensor.item() if f_px_tensor is not None else None
        
        DEPTH_CACHE[abs_path] = {
            "depth": depth,
            "f_px": f_px
        }
        
    return DEPTH_CACHE[abs_path]


def get_cached_cloud(path: Path, depth_result):
    """Cache and return point cloud."""
    abs_path = str(path.absolute())
    if abs_path not in CLOUD_CACHE and depth_result is not None:
        depth = depth_result["depth"]
        f_px = depth_result["f_px"]
        
        if f_px is not None:
            # Get the original image to extract colors
            image = get_cached_image(path)
            if image is not None:
                # Convert PIL image to numpy array for depth2cloud
                img_array = np.array(image)
                CLOUD_CACHE[abs_path] = depth2cloud(depth, f_px, img_array)
            else:
                CLOUD_CACHE[abs_path] = depth2cloud(depth, f_px)
        else:
            LOGGER.warning(f"Cannot generate point cloud for {path}: focal length (f_px) is missing.")
            return None
            
    return CLOUD_CACHE.get(abs_path)


def create_coordinate_frame(size=1.0):
    """Create a coordinate frame showing axes origin."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return frame


def display_point_cloud(cloud_data: dict, window_title: str = "Point Cloud"):
    """Display a 3D point cloud using Open3D with coordinate axes and pixel colors."""
    if not OPEN3D_AVAILABLE:
        LOGGER.warning("Open3D is not installed. Cannot display point cloud.")
        return
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_data["points"])
        
        # Add colors if available
        if "colors" in cloud_data:
            pcd.colors = o3d.utility.Vector3dVector(cloud_data["colors"])
        
        # Create coordinate frame at origin (camera center)
        frame = create_coordinate_frame(size=0.5)  # 0.5 meter sized axes
        
        # Show both the point cloud and coordinate frame
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title)
        vis.add_geometry(pcd)
        vis.add_geometry(frame)
        
        # Optional: set better viewpoint
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])  # Look from camera position
        view_control.set_up([0, -1, 0])     # Standard OpenCV coordinate system
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        LOGGER.error(f"Failed to display point cloud: {e}")


def process_image(
    path: Path, model, transform, device: torch.device, rel_root: Path, args: argparse.Namespace
) -> None:
    """Process a single image: load, infer, save, and/or display results."""
    # Get cached depth data (loads image and runs inference if needed)
    depth_result = get_cached_depth(path, model, transform, device)
    if depth_result is None:
        return  # Error was already logged
        
    depth = depth_result["depth"]
    f_px = depth_result["f_px"]
    
    # Log depth stats
    valid = np.isfinite(depth)
    if not np.any(valid):
        LOGGER.warning(f"No valid depth values found for {path}")
        return

    dmin, dmax = float(np.min(depth[valid])), float(np.max(depth[valid]))
    LOGGER.info(f"Depth stats: min={dmin:.3f}m, max={dmax:.3f}m")
    if f_px is not None:
        print(f"Predicted focal length (f_px): {f_px:.2f}")
        LOGGER.info(f"Predicted focal length (f_px): {f_px:.2f}")
    else:
        LOGGER.warning("Focal length (f_px) not predicted by the model.")

    # Generate cloud if needed
    cloud = None
    if args.cloud or args.save_cloud:
        cloud = get_cached_cloud(path, depth_result)
        
    # --- Saving ---
    if args.output_path:
        # Always save depth map if output path is given
        save_depth(depth, path, rel_root, args.output_path)
        # Save point cloud if requested and generated
        if args.save_cloud and cloud is not None:
            save_point_cloud(cloud, path, rel_root, args.output_path)

    # --- Displaying ---
    if args.cloud:
        if cloud is not None:
            display_point_cloud(cloud, window_title=f"Cloud: {path.name}")
        else:
            LOGGER.warning(f"Cannot display point cloud for {path} (generation failed or f_px missing).")
    elif args.map:
        # Display original image and depth map side by side
        image = get_cached_image(path)
        if image is not None:
            display_depth(depth, path, args.output_path, dmin, dmax, f_px, np.array(image))
        else:
            display_depth(depth, path, args.output_path, dmin, dmax, f_px)


def save_depth(depth: np.ndarray, path: Path, rel_root: Path, output_path: Path) -> None:
    """Save depth map to a compressed .npz file."""
    rel = path.relative_to(rel_root)
    out_base = output_path / rel.parent / rel.stem
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_base.with_suffix(".npz")
    np.savez_compressed(out_file, depth=depth)
    LOGGER.info(f"Saved depth to {out_file}")


def save_point_cloud(cloud: np.ndarray, path: Path, rel_root: Path, output_path: Path) -> None:
    """Save point cloud to a simple .xyz text file."""
    if output_path is None:
        LOGGER.warning("Cannot save point cloud: --output-path not specified.")
        return
    rel = path.relative_to(rel_root)
    out_base = output_path / rel.parent / rel.stem
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_base.with_suffix(".xyz")
    try:
        np.savetxt(out_file, cloud, fmt="%.6f")  # Save as space-separated X Y Z
        LOGGER.info(f"Saved point cloud to {out_file}")
    except Exception as e:
        LOGGER.error(f"Failed to save point cloud {out_file}: {e}")


def display_depth(depth: np.ndarray, path: Path, output_path: Optional[Path], 
                  dmin: float, dmax: float, f_px=None, rgb_image=None) -> None:
    """Display raw depth map with turbo colormap alongside original RGB image."""
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    f_px_value = f_px
    
    # Left subplot: Original RGB image (if available)
    if rgb_image is not None:
        axes[0].imshow(rgb_image)
        axes[0].set_title(f"RGB: {path.name}")
        axes[0].axis("off")
    else:
        axes[0].text(0.5, 0.5, "RGB image not available", 
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")
    
    # Right subplot: Depth map with turbo colormap
    im = axes[1].imshow(depth, cmap="turbo", vmin=dmin, vmax=dmax)
    title = f"Depth (m): {path.name}"
    if f_px_value is not None:
        title += f"\nPredicted f_px: {f_px_value:.2f}"
    axes[1].set_title(title)
    axes[1].axis("off")
    
    # Add colorbar for depth
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label("Depth (m)")
    
    # Add overall title
    fig.suptitle(f"DepthPro: {path.name}", fontsize=16)
    
    plt.show()
    
    # Save figure if output path is provided
    if output_path:
        out_base = output_path / path.stem
        out_fig_file = out_base.with_suffix(".png")
        try:
            fig.savefig(out_fig_file, bbox_inches='tight', dpi=150)
            LOGGER.info(f"Saved side-by-side visualization to {out_fig_file}")
        except Exception as e:
            LOGGER.error(f"Failed to save figure {out_fig_file}: {e}")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Main inference function."""
    configure_logging(args.verbose)
    device = torch_device()
    precision = torch.half if args.fp16 else torch.float
    model, transform = get_cached_model(device, precision)
    
    # If neither map nor cloud display is specified, default to map
    if not args.map and not args.cloud:
        args.map = True
        LOGGER.info("No display option selected, defaulting to depth map display (-m).")
        
    image_paths, rel_root = gather_image_paths(args.image_path)
    for path in tqdm(image_paths, desc="Processing"):
        process_image(path, model, transform, device, rel_root, args)


def main() -> None:
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="DepthPro inference: display/save depth maps and point clouds."
    )
    parser.add_argument(
        "-i", "--image-path", type=Path, required=True,
        help="Input image file or directory"
    )
    parser.add_argument(
        "-o", "--output-path", type=Path,
        help="Directory to save depth (.npz), point cloud (.xyz), and optionally figure (.png) files"
    )

    # Display options (mutually exclusive)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "-m", "--map", action="store_true",
        help="Display raw depth map (default if no display option is specified)"
    )
    display_group.add_argument(
        "-c", "--cloud", action="store_true",
        help="Display 3D point cloud (requires open3d)"
    )

    # Saving options
    parser.add_argument(
        "--save-cloud", action="store_true",
        help="Save the generated point cloud to a .xyz file (requires --output-path)"
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-f", "--fp16", action="store_true",
        dest="fp16",
        help="Use float16 (half) precision for inference"
    )
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.save_cloud and not args.output_path:
        parser.error("--save-cloud requires --output-path to be set.")

    if args.cloud and not OPEN3D_AVAILABLE:
        LOGGER.warning("Open3D is not installed. -c/--cloud option will be ignored, defaulting to depth map.")
        LOGGER.info("Please install Open3D: pip install open3d")
        args.cloud = False
        args.map = True

    # --- Run ---
    run(args)


if __name__ == "__main__":
    main()
