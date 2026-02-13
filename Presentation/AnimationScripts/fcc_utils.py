import numpy as np
import matplotlib.pyplot as plt

def get_fcc_nodes(a=1.0):
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ]) * a
    
    faces = np.array([
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5]
    ]) * a
    
    return np.vstack([corners, faces])

def get_fcc_edges(a=1.0):
    # Edges for unit cube
    edges = [
        ([0, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 1, 0]), ([0, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [1, 1, 0]), ([1, 0, 0], [1, 0, 1]),
        ([0, 1, 0], [1, 1, 0]), ([0, 1, 0], [0, 1, 1]),
        ([0, 0, 1], [1, 0, 1]), ([0, 0, 1], [0, 1, 1]),
        ([1, 1, 0], [1, 1, 1]), ([1, 0, 1], [1, 1, 1]), ([0, 1, 1], [1, 1, 1])
    ]
    return np.array(edges) * a

def get_slip_systems():
    # 4 planes x 3 directions = 12 systems
    # Planes (normals)
    n = np.array([
        [1, 1, 1],
        [-1, 1, 1], # Equivalent to (1, -1, -1) but let's stick to standard defs usually
        [1, -1, 1],
        [1, 1, -1]
    ])
    # Normalize
    n = n / np.linalg.norm(n, axis=1)[:, None]
    
    # Directions for each plane. Dot(n, s) = 0
    # Plane 1 (1 1 1): s can be [0 1 -1], [1 0 -1], [1 -1 0]
    # Plane 2 (-1 1 1): s can be [0 1 -1], [1 0 1], [1 1 0]
    # ...
    
    # Standard 12 systems
    # For now, manually defining some for visualization is fine. 
    # Or generating them.
    
    systems = []
    # Plane (111)
    systems.append(({'n': [1,1,1], 's': [0, 1, -1]}))
    systems.append(({'n': [1,1,1], 's': [1, 0, -1]}))
    systems.append(({'n': [1,1,1], 's': [1, -1, 0]}))
    
    # Plane (-1 1 1) -> (1 1 0), (1 0 1), (0 -1 1) ?
    # Let's keep it simple for visuals - we just need representative ones for the "12 systems" GIF.
    # We can just permute signs.
    
    planes = [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ]
    
    full_systems = []
    
    for p in planes:
        # Find orthogonal directions with integer components like <110>
        # s = [u, v, w] where u,v,w in {-1, 0, 1} and u*p0 + v*p1 + w*p2 = 0
        
        candidates = [
            [0, 1, -1], [0, -1, 1],
            [1, 0, -1], [-1, 0, 1],
            [1, -1, 0], [-1, 1, 0],
            [0, 1, 1], [0, -1, -1], # Valid for some
            [1, 1, 0], # etc
            [1, 0, 1]
        ]
        
        valid_s = []
        for s in candidates:
            if np.abs(np.dot(p, s)) < 1e-5:
                # Check uniqueness (ignore negative duplicates for now or keep them)
                # Keep only 3 unique lines
                is_unique = True
                for existing in valid_s:
                    if np.allclose(existing, s) or np.allclose(existing, [-x for x in s]):
                        is_unique = False
                if is_unique:
                    valid_s.append(s)
        
        for s in valid_s[:3]: # Ensure we take 3 per plane
            full_systems.append({'n': np.array(p)/np.linalg.norm(p), 's': np.array(s)/np.linalg.norm(s)})
            
    return full_systems

def setup_plot():
    plt.style.use('default')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    return fig, ax
