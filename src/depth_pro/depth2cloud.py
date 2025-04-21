import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Try to import Numba for JIT compilation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Base NumPy implementation
def _depth2cloud_numpy(depth_map, f_px, image=None):
    """NumPy implementation of depth2cloud."""
    H, W = depth_map.shape

    # Assume principal point is at the image center
    cx = W / 2.0
    cy = H / 2.0

    # Assume square pixels
    fx = f_px
    fy = f_px

    # Create coordinate grid
    v, u = np.indices((H, W))

    # Filter out invalid depth values
    valid_depth = np.isfinite(depth_map) & (depth_map > 0)
    Z = depth_map[valid_depth]
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]

    # Unproject
    X = (u_valid - cx) * Z / fx
    Y = (v_valid - cy) * Z / fy

    # Stack coordinates
    cloud_points = np.stack((X, Y, Z), axis=-1)
    
    result = {"points": cloud_points}
    
    if image is not None and image.shape[:2] == (H, W):
        # Extract RGB values for valid points
        if len(image.shape) == 3 and image.shape[2] >= 3:
            colors = image[valid_depth]
            # Normalize colors to [0,1] if they're in [0,255]
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
            result["colors"] = colors[:, :3]  # Ensure only RGB channels
            
    return result

# CuPy implementation for GPU acceleration
def _depth2cloud_cupy(depth_map, f_px, image=None):
    """CuPy implementation of depth2cloud for GPU acceleration."""
    # Transfer data to GPU
    depth_map = cp.asarray(depth_map)
    if image is not None:
        image = cp.asarray(image)
    
    H, W = depth_map.shape

    # Assume principal point is at the image center
    cx = W / 2.0
    cy = H / 2.0

    # Assume square pixels
    fx = f_px
    fy = f_px

    # Create coordinate grid
    v, u = cp.indices((H, W))

    # Filter out invalid depth values
    valid_depth = cp.isfinite(depth_map) & (depth_map > 0)
    Z = depth_map[valid_depth]
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]

    # Unproject
    X = (u_valid - cx) * Z / fx
    Y = (v_valid - cy) * Z / fy

    # Stack coordinates
    cloud_points = cp.stack((X, Y, Z), axis=-1)
    
    # Transfer results back to CPU
    result = {"points": cp.asnumpy(cloud_points)}
    
    if image is not None and image.shape[:2] == (H, W):
        # Extract RGB values for valid points
        if len(image.shape) == 3 and image.shape[2] >= 3:
            colors = image[valid_depth]
            # Normalize colors to [0,1] if they're in [0,255]
            if colors.dtype == cp.uint8:
                colors = colors.astype(cp.float32) / 255.0
            result["colors"] = cp.asnumpy(colors[:, :3])  # Ensure only RGB channels
            
    return result

# Numba implementation for CPU JIT compilation
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, parallel=True)
    def _compute_point_cloud(depth_map, valid_mask, H, W, fx, fy, cx, cy):
        """Compute point cloud coordinates with Numba acceleration."""
        # Pre-allocate output arrays
        num_points = np.sum(valid_mask)
        X = np.empty(num_points, dtype=depth_map.dtype)
        Y = np.empty(num_points, dtype=depth_map.dtype)
        Z = np.empty(num_points, dtype=depth_map.dtype)
        
        # Process each point
        idx = 0
        for v in range(H):
            for u in range(W):
                if valid_mask[v, u]:
                    z = depth_map[v, u]
                    Z[idx] = z
                    X[idx] = (u - cx) * z / fx
                    Y[idx] = (v - cy) * z / fy
                    idx += 1
                    
        return np.column_stack((X, Y, Z))

    def _depth2cloud_numba(depth_map, f_px, image=None):
        """Numba-optimized implementation of depth2cloud."""
        H, W = depth_map.shape

        # Assume principal point is at the image center
        cx = W / 2.0
        cy = H / 2.0

        # Assume square pixels
        fx = f_px
        fy = f_px

        # Filter out invalid depth values
        valid_depth = np.isfinite(depth_map) & (depth_map > 0)
        
        # Compute point cloud using Numba-optimized function
        cloud_points = _compute_point_cloud(depth_map, valid_depth, H, W, fx, fy, cx, cy)
        
        result = {"points": cloud_points}
        
        if image is not None and image.shape[:2] == (H, W):
            # Extract RGB values for valid points
            if len(image.shape) == 3 and image.shape[2] >= 3:
                colors = image[valid_depth]
                # Normalize colors to [0,1] if they're in [0,255]
                if colors.dtype == np.uint8:
                    colors = colors.astype(np.float32) / 255.0
                result["colors"] = colors[:, :3]  # Ensure only RGB channels
                
        return result

def depth2cloud(depth_map: np.ndarray, f_px: float, image: np.ndarray = None):
    """Convert depth map to point cloud using camera intrinsics.
    
    Automatically selects the fastest available implementation:
    1. CuPy (if available) - GPU acceleration
    2. Numba (if available) - CPU JIT compilation
    3. NumPy (fallback) - Vectorized CPU operations

    Args:
        depth_map: A (H, W) numpy array containing depth values (in meters).
        f_px: The estimated horizontal focal length in pixels.
        image: Optional (H, W, 3) RGB image for coloring the point cloud.

    Returns:
        A dictionary containing:
        - "points": (N, 3) numpy array representing the point cloud coordinates
        - "colors": (N, 3) numpy array with RGB colors (if image was provided)
    """
    # Choose the best available implementation
    if CUPY_AVAILABLE:
        return _depth2cloud_cupy(depth_map, f_px, image)
    elif NUMBA_AVAILABLE:
        return _depth2cloud_numba(depth_map, f_px, image)
    else:
        return _depth2cloud_numpy(depth_map, f_px, image)
