import trimesh
import numpy as np
import torch
import plotly.graph_objects as go
from tqdm.auto import tqdm
import time
import networkx as nx
from collections import defaultdict
import community as community_louvain
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
            
    def get_sets(self):
        sets = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(i)
        return list(sets.values())

class MeshBounceAnalyzer:
    def __init__(self, mesh: trimesh.Trimesh, n_bounces: int = 10, n_additional_rays: int = 4, theta_degrees: float = 30.0):
        # Determine the best available device
        self.device = self._get_optimal_device()
        print(f"Using device: {self.device}")
        
        self.mesh = mesh
        self.n_bounces = n_bounces
        self.n_additional_rays = n_additional_rays
        self.theta = np.radians(theta_degrees)
        self.face_count = len(mesh.faces)
        self.hit_counts = torch.zeros(self.face_count, device=self.device)
        
        # Transfer mesh data to the selected device
        self.vertices = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        self.faces = torch.tensor(mesh.faces, device=self.device, dtype=torch.int64)
        self.face_normals = torch.tensor(mesh.face_normals, device=self.device, dtype=torch.float32)
        
        # Precompute triangle data for intersection testing
        self.v0 = self.vertices[self.faces[:, 0]]
        self.edge1 = self.vertices[self.faces[:, 1]] - self.v0
        self.edge2 = self.vertices[self.faces[:, 2]] - self.v0
        
        # Compute face centroids
        self.face_centroids = torch.mean(torch.stack([
            self.vertices[self.faces[:, 0]],
            self.vertices[self.faces[:, 1]],
            self.vertices[self.faces[:, 2]]
        ]), dim=0)
        
        print(f"Initialized with {self.face_count} faces")
        print(f"Number of bounces: {self.n_bounces}")
        print(f"Additional rays per face: {self.n_additional_rays}")
        print(f"Cone angle: {theta_degrees} degrees")

    def _get_optimal_device(self):
        """Determine the best available device for computation"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # def optimize_batch_size(self):
    #     """Determine optimal batch size based on available memory"""
    #     if self.device.type == "cuda":
    #         # For CUDA, use a portion of available memory
    #         free_memory = torch.cuda.get_device_properties(0).total_memory * 0.7  # Use 70% of total memory
    #         # Rough estimation of memory per ray
    #         memory_per_ray = 4 * 3 * self.face_count  # 4 bytes per float32, 3 coordinates, face_count triangles
    #         return int(free_memory / memory_per_ray)
    #     elif self.device.type == "mps":
    #         # For Apple Silicon, use a conservative batch size
    #         return 5000
    #     else:
    #         # For CPU, use a smaller batch size
    #         return 1000
    def optimize_batch_size(self):
        """Determine optimal batch size based on device and constraints"""
        if self.device.type == "mps":
            # More conservative batch size for MPS to avoid 32-bit indexing issues
            return min(2000, self.face_count)
        elif self.device.type == "cuda":
            free_memory = torch.cuda.get_device_properties(0).total_memory * 0.7
            memory_per_ray = 4 * 3 * self.face_count
            return min(int(free_memory / memory_per_ray), 10000)
        else:
            return 1000

    def parallel_ray_triangle_intersect(self, ray_origins, ray_directions, show_progress=False):
        """Vectorized ray-triangle intersection with improved batch handling"""
        n_rays = ray_origins.shape[0]
        batch_size = self.optimize_batch_size()
        n_batches = (n_rays + batch_size - 1) // batch_size
        
        # Initialize results
        distances = torch.full((n_rays,), float('inf'), device=self.device)
        face_indices = torch.full((n_rays,), -1, device=self.device, dtype=torch.long)
        points = torch.zeros((n_rays, 3), device=self.device)
        
        # Create iterator with optional progress bar
        batch_range = range(n_batches)
        if show_progress:
            batch_range = tqdm(batch_range, desc="Processing ray-triangle intersections")
        
        for batch_idx in batch_range:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_rays)
            batch_rays = slice(start_idx, end_idx)
            
            # Process batch
            batch_results = self._process_intersection_batch(
                ray_origins[batch_rays],
                ray_directions[batch_rays]
            )
            
            # Update results
            face_indices[batch_rays] = batch_results[0]
            points[batch_rays] = batch_results[1]
            distances[batch_rays] = batch_results[2]
            
            # Clean up GPU memory if needed
            if self.device.type in ["cuda", "mps"]:
                torch.cuda.empty_cache() if self.device.type == "cuda" else None
        
        return face_indices, points, distances

    def _process_intersection_batch(self, batch_origins, batch_directions):
        """Process a single batch of ray-triangle intersections with improved memory handling"""
        batch_size = batch_origins.shape[0]
        
        # Process faces in sub-batches to avoid 32-bit indexing issues
        max_faces_per_batch = 1000  # Adjust this value based on your GPU memory
        face_batches = (self.face_count + max_faces_per_batch - 1) // max_faces_per_batch
        
        # Initialize results for this batch
        closest_faces = torch.full((batch_size,), -1, device=self.device, dtype=torch.long)
        closest_points = torch.zeros((batch_size, 3), device=self.device)
        closest_distances = torch.full((batch_size,), float('inf'), device=self.device)
        
        for face_batch in range(face_batches):
            start_face = face_batch * max_faces_per_batch
            end_face = min((face_batch + 1) * max_faces_per_batch, self.face_count)
            face_slice = slice(start_face, end_face)
            
            # Expand rays for this face batch
            current_batch_origins = batch_origins.unsqueeze(1).expand(-1, end_face - start_face, -1)
            current_batch_directions = batch_directions.unsqueeze(1).expand(-1, end_face - start_face, -1)
            
            # Get face data for this batch
            current_edge1 = self.edge1[face_slice].unsqueeze(0)
            current_edge2 = self.edge2[face_slice].unsqueeze(0)
            current_v0 = self.v0[face_slice].unsqueeze(0)
            
            # Möller–Trumbore algorithm
            epsilon = 1e-7
            pvec = torch.linalg.cross(current_batch_directions, current_edge2, dim=2)
            det = torch.sum(current_edge1 * pvec, dim=2)
            
            valid_det = torch.abs(det) > epsilon
            if torch.any(valid_det):
                inv_det = torch.where(valid_det, 1.0 / det, torch.zeros_like(det))
                tvec = current_batch_origins - current_v0
                u = torch.sum(tvec * pvec, dim=2) * inv_det
                
                valid_u = (u >= -epsilon) & (u <= 1.0 + epsilon) & valid_det
                if torch.any(valid_u):
                    qvec = torch.linalg.cross(tvec, current_edge1, dim=2)
                    v = torch.sum(current_batch_directions * qvec, dim=2) * inv_det
                    
                    valid_v = (v >= -epsilon) & (u + v <= 1.0 + epsilon) & valid_u
                    if torch.any(valid_v):
                        t = torch.sum(current_edge2 * qvec, dim=2) * inv_det
                        
                        # Check for back face entry
                        current_normals = self.face_normals[face_slice].unsqueeze(0)
                        entering_from_back = torch.sum(current_batch_directions * current_normals, dim=2) > epsilon
                        
                        valid_hits = (t > epsilon) & valid_v & entering_from_back
                        if torch.any(valid_hits):
                            t_masked = torch.where(valid_hits, t, torch.tensor(float('inf'), device=self.device))
                            current_closest_hits, current_closest_indices = torch.min(t_masked, dim=1)
                            
                            # Update global closest hits if better than current
                            update_mask = current_closest_hits < closest_distances
                            if torch.any(update_mask):
                                closest_distances[update_mask] = current_closest_hits[update_mask]
                                closest_faces[update_mask] = current_closest_indices[update_mask] + start_face
                                closest_points[update_mask] = (
                                    batch_origins[update_mask] + 
                                    batch_directions[update_mask] * 
                                    current_closest_hits[update_mask].unsqueeze(1)
                                )
        
        return closest_faces, closest_points, closest_distances

    def _compute_intersections(self, origins, directions, pvec, det, valid_det, epsilon):
        """Compute intersection points with optimizations for different devices"""
        batch_size = origins.shape[0]
        
        # Initialize results
        closest_faces = torch.full((batch_size,), -1, device=self.device, dtype=torch.long)
        closest_points = torch.zeros((batch_size, 3), device=self.device)
        closest_distances = torch.full((batch_size,), float('inf'), device=self.device)
        
        if torch.any(valid_det):
            inv_det = torch.where(valid_det, 1.0 / det, torch.zeros_like(det))
            tvec = origins - self.v0.unsqueeze(0)
            u = torch.sum(tvec * pvec, dim=2) * inv_det
            
            valid_u = (u >= -epsilon) & (u <= 1.0 + epsilon) & valid_det
            
            if torch.any(valid_u):
                qvec = torch.linalg.cross(tvec, self.edge1.unsqueeze(0), dim=2)
                v = torch.sum(directions * qvec, dim=2) * inv_det
                
                valid_v = (v >= -epsilon) & (u + v <= 1.0 + epsilon) & valid_u
                
                if torch.any(valid_v):
                    t = torch.sum(self.edge2.unsqueeze(0) * qvec, dim=2) * inv_det
                    
                    # Check if ray enters from back face (interior)
                    face_normals = self.face_normals.unsqueeze(0).expand(batch_size, -1, -1)
                    entering_from_back = torch.sum(directions * face_normals, dim=2) > epsilon
                    
                    # Final validity check
                    valid_hits = (t > epsilon) & valid_v & entering_from_back
                    
                    if torch.any(valid_hits):
                        t_masked = torch.where(valid_hits, t, torch.tensor(float('inf'), device=self.device))
                        closest_hits, closest_indices = torch.min(t_masked, dim=1)
                        
                        valid_rays = closest_hits < float('inf')
                        closest_faces[valid_rays] = closest_indices[valid_rays]
                        closest_distances[valid_rays] = closest_hits[valid_rays]
                        
                        closest_points[valid_rays] = (
                            origins[valid_rays, 0] + 
                            directions[valid_rays, 0] * closest_hits[valid_rays].unsqueeze(1)
                        )
        
        return closest_faces, closest_points, closest_distances

    def generate_rotated_rays(self, base_directions, n_rays, theta):
        """Generate multiple rays within a cone around base directions"""
        num_faces = base_directions.shape[0]
        
        # Generate random angles for all rays at once
        phi = theta * torch.sqrt(torch.rand(num_faces, n_rays, device=self.device))
        omega = 2 * np.pi * torch.rand(num_faces, n_rays, device=self.device)
        
        # Convert to cartesian coordinates (relative to z-axis)
        sin_phi = torch.sin(phi)
        perturbations = torch.stack([
            sin_phi * torch.cos(omega),
            sin_phi * torch.sin(omega),
            torch.cos(phi)
        ], dim=-1)
        
        # For each base direction, we need a rotation matrix
        z_axis = torch.tensor([0., 0., 1.], device=self.device).expand(num_faces, 3)
        
        # Compute rotation axes and angles
        rotation_axes = torch.linalg.cross(z_axis, base_directions, dim=1)
        rotation_angles = torch.acos(torch.clamp(
            torch.sum(base_directions * z_axis, dim=1),
            -1, 1
        ))
        
        # Normalize rotation axes
        axes_norm = torch.norm(rotation_axes, dim=1, keepdim=True)
        valid_axes = axes_norm > 1e-6
        rotation_axes = torch.where(
            valid_axes, rotation_axes / axes_norm, torch.zeros_like(rotation_axes)
        )
        
        # Compute rotation matrices
        cos_angles = torch.cos(rotation_angles).unsqueeze(-1).unsqueeze(-1)
        sin_angles = torch.sin(rotation_angles).unsqueeze(-1).unsqueeze(-1)
        
        # Create skew-symmetric matrices
        K = torch.zeros(num_faces, 3, 3, device=self.device)
        K[:, 0, 1] = -rotation_axes[:, 2]
        K[:, 0, 2] = rotation_axes[:, 1]
        K[:, 1, 0] = rotation_axes[:, 2]
        K[:, 1, 2] = -rotation_axes[:, 0]
        K[:, 2, 0] = -rotation_axes[:, 1]
        K[:, 2, 1] = rotation_axes[:, 0]
        
        # Compute rotation matrices using Rodrigues' formula
        I = torch.eye(3, device=self.device).expand(num_faces, 3, 3)
        R = I + sin_angles * K + (1 - cos_angles) * torch.bmm(K, K)
        
        # Apply rotations
        perturbations = perturbations.view(-1, 3)
        R = R.repeat_interleave(n_rays, dim=0)
        rotated_directions = torch.bmm(R, perturbations.unsqueeze(-1)).squeeze(-1)
        
        # Normalize the directions
        rotated_directions = rotated_directions / torch.norm(rotated_directions, dim=1, keepdim=True)
        
        return rotated_directions

    def analyze(self):
        """Trace rays from face centroids with multiple initial directions per face"""
        print("\nStarting face-based analysis...")
        
        # Generate base directions for each face
        base_origins = self.face_centroids
        base_directions = -self.face_normals
        
        # Generate all ray directions at once
        print("Generating ray directions...")
        additional_directions = self.generate_rotated_rays(
            base_directions, self.n_additional_rays, self.theta
        )
        
        # Combine base and additional rays
        all_directions = torch.cat([
            base_directions,
            additional_directions.view(self.face_count, self.n_additional_rays, 3).reshape(-1, 3)
        ], dim=0)
        
        # Replicate origins for all rays
        all_origins = base_origins.repeat(self.n_additional_rays + 1, 1)
        
        print(f"Processing {len(all_directions)} rays for initial bounce...")
        
        # Storage for tracking
        all_paths = []
        total_intersections = 0
        hits_per_bounce = []
        
        # Process first bounce with all rays
        hit_faces, hit_points, distances = self.parallel_ray_triangle_intersect(
            all_origins, all_directions, show_progress=True
        )
        
        # Reshape results
        total_rays = self.n_additional_rays + 1
        hit_faces = hit_faces.view(total_rays, self.face_count).t()
        hit_points = hit_points.view(total_rays, self.face_count, 3).transpose(0, 1)
        distances = distances.view(total_rays, self.face_count).t()
        
        # Process hits
        valid_hits = hit_faces >= 0
        any_valid_hits = torch.any(valid_hits, dim=1)
        first_hit_indices = torch.argmax(valid_hits.float(), dim=1)
        
        # Create device-consistent index tensors
        face_range = torch.arange(self.face_count, device=self.device)
        
        # Get hit information
        selected_hit_faces = hit_faces[face_range, first_hit_indices]
        selected_hit_points = hit_points[face_range, first_hit_indices]
        
        # Update for faces with no valid hits
        selected_hit_faces[~any_valid_hits] = -1
        
        # Store first bounce results
        all_paths.append((base_origins, selected_hit_points, selected_hit_faces))
        bounce_hits = torch.sum(any_valid_hits).item()
        hits_per_bounce.append(bounce_hits)
        total_intersections += bounce_hits
        
        # Record ray chains
        face_indices = torch.arange(self.face_count, device=self.device)
        ray_chains = torch.stack([face_indices, selected_hit_faces], dim=1)
        
        # Update hit counts
        if bounce_hits > 0:
            valid_face_indices = selected_hit_faces[any_valid_hits]
            hit_increments = torch.ones(len(valid_face_indices), device=self.device)
            self.hit_counts.index_add_(0, valid_face_indices, hit_increments)
        
        # Continue with subsequent bounces
        current_origins = selected_hit_points
        current_directions = torch.zeros_like(current_origins)
        
        # Compute reflected directions for successful hits
        if torch.any(any_valid_hits):
            valid_indices = torch.nonzero(any_valid_hits).squeeze(1)
            valid_first_hits = first_hit_indices[valid_indices]
            valid_face_range = face_range[valid_indices]
            
            normals = self.face_normals[valid_face_indices]
            
            # Reshape all_directions and index correctly
            reshaped_directions = all_directions.view(total_rays, self.face_count, 3)
            original_dirs = reshaped_directions[valid_first_hits, valid_face_range]
            
            dot_products = torch.sum(original_dirs * normals, dim=1, keepdim=True)
            current_directions[valid_indices] = (
                original_dirs - 2 * dot_products * normals
            )
            current_directions[valid_indices] /= torch.norm(
                current_directions[valid_indices], dim=1, keepdim=True
            )
        
        print(f"\nProcessing subsequent bounces...")
        # Process remaining bounces
        with tqdm(total=self.n_bounces - 1, desc="Processing bounces") as pbar:
            for bounce in range(1, self.n_bounces):
                hit_faces, hit_points, distances = self.parallel_ray_triangle_intersect(
                    current_origins, current_directions, show_progress=False
                )
                
                # Store path segments
                all_paths.append((
                    current_origins.clone(),
                    hit_points.clone(),
                    hit_faces.clone()
                ))
                
                # Update statistics
                valid_hits = hit_faces >= 0
                bounce_hits = torch.sum(valid_hits).item()
                hits_per_bounce.append(bounce_hits)
                
                if bounce_hits > 0:
                    total_intersections += bounce_hits
                    valid_face_indices = hit_faces[valid_hits]
                    hit_increments = torch.ones(len(valid_face_indices), device=self.device)
                    self.hit_counts.index_add_(0, valid_face_indices, hit_increments)
                    
                    # Update ray chains
                    ray_chains = torch.cat((ray_chains, hit_faces.unsqueeze(1)), dim=1)
                    
                    # Prepare for next bounce
                    if bounce < self.n_bounces - 1:
                        # Update origins to hit points
                        new_origins = current_origins.clone()
                        new_directions = current_directions.clone()
                        
                        # Get valid indices
                        valid_indices = torch.nonzero(valid_hits).squeeze(1)
                        
                        # Compute reflections for valid hits
                        normals = self.face_normals[valid_face_indices]
                        valid_directions = new_directions[valid_indices]
                        
                        dot_products = torch.sum(valid_directions * normals, dim=1, keepdim=True)
                        new_directions[valid_indices] = (
                            valid_directions - 2 * dot_products * normals
                        )
                        new_origins[valid_indices] = hit_points[valid_indices]
                        
                        # Normalize reflected directions
                        new_directions[valid_indices] = new_directions[valid_indices] / torch.norm(
                            new_directions[valid_indices], dim=1, keepdim=True
                        )
                        
                        current_origins = new_origins
                        current_directions = new_directions
                    
                pbar.update(1)
                if bounce_hits == 0:
                    break
        
        # Report final statistics
        unsuccessful_faces = ~torch.any(ray_chains[:, 1:] >= 0, dim=1)
        if torch.any(unsuccessful_faces):
            print(f"\nWarning: {torch.sum(unsuccessful_faces).item()} faces never achieved a successful raycast")
            print("Face indices:", torch.nonzero(unsuccessful_faces).cpu().numpy().flatten())
        
        return all_paths, total_intersections, hits_per_bounce, ray_chains

    def build_transition_graph(self, ray_chains):
        """Build a transition graph from ray chains"""
        # Move data to CPU for graph processing
        ray_chains_np = ray_chains.cpu().numpy()
        transition_counts = defaultdict(int)
        
        for chain in ray_chains_np:
            for i in range(len(chain) - 1):
                from_face = chain[i]
                to_face = chain[i+1]
                if from_face >= 0 and to_face >= 0:
                    transition_counts[(from_face, to_face)] += 1
        
        G = nx.DiGraph()
        for (from_face, to_face), count in transition_counts.items():
            G.add_edge(from_face, to_face, weight=count)
        
        return G

    def build_face_adjacency_dict(self):
        """Build a dictionary of face adjacencies"""
        face_adj_dict = defaultdict(list)
        face_adjacency = self.mesh.face_adjacency
        
        for edge in face_adjacency:
            face1, face2 = edge
            face_adj_dict[face1].append(face2)
            face_adj_dict[face2].append(face1)
        
        return face_adj_dict

    def refine_partition_with_connectivity(self, partition):
        """Refine partition based on physical connectivity using Union Find"""
        # Build face adjacency dictionary
        face_adj_dict = self.build_face_adjacency_dict()
        
        # Group faces by their initial partition
        partition_groups = defaultdict(list)
        for face_idx, cluster_id in partition.items():
            partition_groups[cluster_id].append(face_idx)
        
        # New partition starting with max cluster_id + 1
        new_cluster_id = max(partition.values()) + 1
        new_partition = partition.copy()
        
        # Process each partition separately
        for cluster_id, faces in partition_groups.items():
            # Create UnionFind for this partition
            uf = UnionFind(len(faces))
            
            # Create mapping from global face index to local index
            face_to_local = {face: i for i, face in enumerate(faces)}
            
            # Union adjacent faces within this partition
            for i, face in enumerate(faces):
                for adj_face in face_adj_dict[face]:
                    if adj_face in face_to_local:  # if adjacent face is in same partition
                        uf.union(i, face_to_local[adj_face])
            
            # Get connected components
            connected_components = uf.get_sets()
            
            # If more than one connected component, assign new cluster IDs
            if len(connected_components) > 1:
                for component in connected_components[1:]:  # Keep first component with original ID
                    for local_idx in component:
                        global_face_idx = faces[local_idx]
                        new_partition[global_face_idx] = new_cluster_id
                    new_cluster_id += 1
        
        return new_partition
    
    def merge_small_partitions(self, partition, min_partition_size=3, iteration=1):
        """
        Iteratively merge partitions smaller than the specified minimum size until no small partitions remain.
        """
        # Create a copy of the partition to modify
        new_partition = partition.copy()
        
        # Ensure all faces have a partition assigned
        for face_idx in range(self.face_count):
            if face_idx not in new_partition:
                new_partition[face_idx] = max(partition.values()) + 1
        
        # Build face adjacency dictionary
        face_adj_dict = self.build_face_adjacency_dict()
        
        # Group faces by partition ID
        partition_groups = defaultdict(list)
        for face_idx, cluster_id in new_partition.items():
            partition_groups[cluster_id].append(face_idx)
        
        # Sort partition IDs by size (process smallest partitions first)
        partition_sizes = [(cluster_id, len(faces)) for cluster_id, faces in partition_groups.items()]
        partition_sizes.sort(key=lambda x: x[1])  # Sort by size
        
        def get_largest_valid_partition(face_idx, partner_faces, original_cluster, new_partition):
            """Helper function to find the largest valid adjacent partition"""
            adjacent_faces = face_adj_dict[face_idx]
            
            if adjacent_faces:
                # Get partition IDs of adjacent faces, excluding partner faces' partitions
                adjacent_partitions = set()
                
                for adj_face in adjacent_faces:
                    if adj_face not in partner_faces:  # Exclude partner faces
                        if adj_face in new_partition:  # Check if adjacent face has a partition
                            adj_partition = new_partition[adj_face]
                            if adj_partition != original_cluster:  # Exclude original cluster
                                adjacent_partitions.add(adj_partition)
                
                if adjacent_partitions:
                    # Find the largest adjacent partition
                    return max(
                        adjacent_partitions,
                        key=lambda p: len([f for f, c in new_partition.items() if c == p])
                    )
            return None
        
        # Process partitions from smallest to largest
        small_partitions_processed = 0
        for cluster_id, size in partition_sizes:
            if size < min_partition_size:
                small_partitions_processed += 1
                faces = partition_groups[cluster_id]
                original_cluster = cluster_id
                
                # Process each face independently
                for face_idx in faces:
                    # Get partner faces (all other faces in the same partition)
                    partner_faces = [f for f in faces if f != face_idx]
                    largest_partition = get_largest_valid_partition(face_idx, partner_faces, original_cluster, new_partition)
                    
                    if largest_partition is not None:
                        new_partition[face_idx] = largest_partition
                        partition_groups[largest_partition].append(face_idx)
                        partition_groups[cluster_id].remove(face_idx)
        
        # Renumber partitions to be consecutive integers starting from 0
        unique_partitions = sorted(set(new_partition.values()))
        partition_map = {old: new for new, old in enumerate(unique_partitions)}
        final_partition = {face: partition_map[cluster] for face, cluster in new_partition.items()}
        
        # Print merge statistics with detailed breakdown
        print(f"\nMerge Statistics (Iteration {iteration}, minimum size: {min_partition_size}):")
        
        # Count final partition sizes
        final_groups = defaultdict(list)
        for face_idx, cluster_id in final_partition.items():
            final_groups[cluster_id].append(face_idx)
        
        final_partition_sizes = {len(faces): count for count, faces in 
                               enumerate([faces for cluster_id, faces in final_groups.items()])}
        
        print("\nFinal partition size distribution:")
        has_small_partitions = False
        for size, count in sorted(final_partition_sizes.items()):
            suffix = "* (below minimum)" if size < min_partition_size else ""
            print(f"Size {size}: {count} partitions {suffix}")
            if size < min_partition_size:
                has_small_partitions = True
        
        initial_clusters = len(set(partition.values()))
        final_clusters = len(set(final_partition.values()))
        print(f"\nTotal clusters merged in this iteration: {initial_clusters - final_clusters}")
        print(f"Small partitions processed: {small_partitions_processed}")
        
        # If we still have small partitions, recursively call the function
        if has_small_partitions:
            print(f"\nSmall partitions still exist. Starting iteration {iteration + 1}...")
            return self.merge_small_partitions(final_partition, min_partition_size, iteration + 1)
        
        print("\nNo small partitions remain. Merge process complete.")
        return final_partition

    def cluster_graph(self, G, resolution=1.0, min_cluster_size=None, edge_threshold=None):
        """
        Enhanced graph clustering using the Louvain method with additional controls
        """
        # Create a working copy of the graph
        G_working = G.copy()
        
        # Optional edge weight thresholding
        if edge_threshold is not None:
            edges_to_remove = [
                (u, v) for u, v, w in G_working.edges(data='weight')
                if w < edge_threshold
            ]
            G_working.remove_edges_from(edges_to_remove)
            print(f"Removed {len(edges_to_remove)} edges below threshold {edge_threshold}")
        
        # Convert to undirected graph for community detection
        G_undirected = G_working.to_undirected()
        
        # Apply Louvain method with resolution parameter
        initial_partition = community_louvain.best_partition(
            G_undirected, 
            weight='weight',
            resolution=resolution
        )
        
        # If no minimum cluster size specified, return initial partition
        if min_cluster_size is None:
            return initial_partition
        
        # Post-process clusters based on size
        clusters = defaultdict(list)
        for node, cluster in initial_partition.items():
            clusters[cluster].append(node)
        
        # Sort clusters by size for merging
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]))
        
        # Merge small clusters with their most connected neighbors
        final_partition = initial_partition.copy()
        next_cluster_id = max(initial_partition.values()) + 1
        
        for cluster_id, nodes in sorted_clusters:
            if len(nodes) < min_cluster_size:
                # Find the most connected neighboring cluster
                neighbor_connections = defaultdict(float)
                
                for node in nodes:
                    for neighbor in G_undirected.neighbors(node):
                        neighbor_cluster = final_partition[neighbor]
                        if neighbor_cluster != cluster_id:
                            edge_weight = G_undirected[node][neighbor]['weight']
                            neighbor_connections[neighbor_cluster] += edge_weight
                
                if neighbor_connections:
                    # Merge with the most connected cluster
                    best_neighbor = max(neighbor_connections.items(), key=lambda x: x[1])[0]
                    for node in nodes:
                        final_partition[node] = best_neighbor
                else:
                    # If no neighbors, create a new cluster
                    for node in nodes:
                        final_partition[node] = next_cluster_id
                    next_cluster_id += 1
        
        # Print clustering statistics
        initial_clusters = len(set(initial_partition.values()))
        final_clusters = len(set(final_partition.values()))
        print(f"\nClustering Statistics:")
        print(f"Resolution parameter: {resolution}")
        print(f"Initial clusters: {initial_clusters}")
        print(f"Clusters after size threshold: {final_clusters}")
        
        # Analyze cluster sizes
        cluster_sizes = defaultdict(int)
        for cluster in final_partition.values():
            cluster_sizes[cluster] += 1
        
        print("\nCluster size distribution:")
        print(f"Minimum cluster size: {min(cluster_sizes.values())}")
        print(f"Maximum cluster size: {max(cluster_sizes.values())}")
        print(f"Average cluster size: {sum(cluster_sizes.values()) / len(cluster_sizes):.1f}")
        
        return final_partition
    
    def analyze_clustering_resolutions(self, G, resolutions=[0.5, 1.0, 2.0, 4.0]):
        """Analyze the effect of different resolution parameters on clustering"""
        results = {}
        
        for resolution in resolutions:
            partition = self.cluster_graph(G, resolution=resolution)
            n_clusters = len(set(partition.values()))
            
            # Calculate cluster sizes
            cluster_sizes = defaultdict(int)
            for cluster in partition.values():
                cluster_sizes[cluster] += 1
                
            # Calculate modularity
            G_undirected = G.to_undirected()
            modularity = community_louvain.modularity(partition, G_undirected, weight='weight')
            
            results[resolution] = {
                'n_clusters': n_clusters,
                'min_size': min(cluster_sizes.values()),
                'max_size': max(cluster_sizes.values()),
                'avg_size': sum(cluster_sizes.values()) / len(cluster_sizes),
                'modularity': modularity
            }
            
        # Print results
        print("\nClustering Resolution Analysis:")
        print("\nResolution | Clusters | Min Size | Max Size | Avg Size | Modularity")
        print("-" * 65)
        for res, stats in results.items():
            print(f"{res:9.1f} | {stats['n_clusters']:8d} | {stats['min_size']:8d} | "
                  f"{stats['max_size']:8d} | {stats['avg_size']:8.1f} | {stats['modularity']:.3f}")
        
        return results

    def get_optimal_resolution(self, G, target_clusters, tolerance=0.1):
        """Find the resolution parameter that yields a target number of clusters"""
        def cluster_count(resolution):
            partition = self.cluster_graph(G, resolution=resolution)
            return len(set(partition.values()))
        
        # Binary search for optimal resolution
        min_res = 0.1
        max_res = 10.0
        max_iterations = 20
        
        for i in range(max_iterations):
            current_res = (min_res + max_res) / 2
            current_clusters = cluster_count(current_res)
            
            error = abs(current_clusters - target_clusters) / target_clusters
            if error <= tolerance:
                print(f"\nFound resolution {current_res:.3f} yielding {current_clusters} clusters")
                return current_res
            
            if current_clusters < target_clusters:
                min_res = current_res
            else:
                max_res = current_res
        
        print(f"\nWarning: Could not achieve target clusters within tolerance. Best resolution: {current_res}")
        return current_res

    def visualize_segmentation(self, partition):
        """Visualize the mesh segmentation"""
        num_clusters = max(partition.values()) + 1
        face_colors = np.zeros((self.face_count, 3))  # RGB colors
        
        # Generate distinct colors for clusters using updated colormap API
        colors = plt.colormaps['hsv'](np.linspace(0, 1, num_clusters))
        
        for face_idx in range(self.face_count):
            cluster_idx = partition.get(face_idx, -1)
            if cluster_idx >= 0:
                face_colors[face_idx] = colors[cluster_idx][:3]  # Ignore alpha channel
            else:
                # Faces not assigned to any cluster
                face_colors[face_idx] = [0, 0, 0]
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=self.mesh.vertices[:, 0],
                y=self.mesh.vertices[:, 1],
                z=self.mesh.vertices[:, 2],
                i=self.mesh.faces[:, 0],
                j=self.mesh.faces[:, 1],
                k=self.mesh.faces[:, 2],
                facecolor=['rgb({}, {}, {})'.format(*(color * 255).astype(int)) for color in face_colors],
                flatshading=True
            )
        ])
        
        fig.update_layout(
            scene=dict(aspectmode='data'),
            width=800,
            height=1000
        )
        
        return fig
    
    def visualize_cluster_hulls(self, partition):
        """Visualize the convex hulls of mesh clusters"""
        # Group vertices by cluster
        cluster_vertices = {}
        num_clusters = max(partition.values()) + 1
        colors = plt.colormaps['hsv'](np.linspace(0, 1, num_clusters))
        
        # Collect vertices for each cluster
        for face_idx, cluster_id in partition.items():
            if cluster_id not in cluster_vertices:
                cluster_vertices[cluster_id] = set()
            
            # Add all vertices of the face to the cluster's set
            face_verts = self.mesh.faces[face_idx]
            for vert_idx in face_verts:
                cluster_vertices[cluster_id].add(tuple(self.mesh.vertices[vert_idx]))
        
        # Create figure
        fig = go.Figure()
        
        # Add original mesh with transparency
        fig.add_trace(go.Mesh3d(
            x=self.mesh.vertices[:, 0],
            y=self.mesh.vertices[:, 1],
            z=self.mesh.vertices[:, 2],
            i=self.mesh.faces[:, 0],
            j=self.mesh.faces[:, 1],
            k=self.mesh.faces[:, 2],
            opacity=0.1,
            color='lightgray'
        ))
        
        # Create convex hull for each cluster
        for cluster_id, vertices in cluster_vertices.items():
            if len(vertices) < 4:  # Need at least 4 points for 3D convex hull
                continue
            
            # Convert vertices set to numpy array
            points = np.array(list(vertices))
            
            try:
                # Compute convex hull
                hull = ConvexHull(points)
                
                # Get color for this cluster
                color = colors[cluster_id][:3]
                
                # Add convex hull as a mesh
                fig.add_trace(go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                    opacity=0.3,
                    name=f'Cluster {cluster_id}'
                ))
            except Exception as e:
                print(f"Failed to compute convex hull for cluster {cluster_id}: {e}")
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                annotations=[
                    dict(
                        showarrow=False,
                        x=0,
                        y=0,
                        z=0,
                        text="Cluster Convex Hulls",
                        xanchor="left",
                        xshift=10,
                        opacity=0.7
                    )
                ]
            ),
            title='Mesh Clusters Convex Hulls',
            width=800,
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def visualize_ray_paths(self, paths, max_paths=1000):
        """Visualize ray paths"""
        lines_x = []
        lines_y = []
        lines_z = []
        
        # Sample paths if needed
        n_total_paths = len(paths[0][0])
        if n_total_paths > max_paths:
            selected_indices = torch.randperm(n_total_paths)[:max_paths]
        else:
            selected_indices = torch.arange(n_total_paths)
        
        # Process each bounce
        for bounce_idx in range(len(paths)):
            origins, endpoints, hit_faces = paths[bounce_idx]
            origins = origins[selected_indices].cpu().numpy()
            endpoints = endpoints[selected_indices].cpu().numpy()
            hit_faces = hit_faces[selected_indices].cpu().numpy()
            
            valid_hits = hit_faces >= 0
            for i in range(len(selected_indices)):
                if valid_hits[i]:
                    start = origins[i]
                    end = endpoints[i]
                    lines_x.extend([start[0], end[0], None])
                    lines_y.extend([start[1], end[1], None])
                    lines_z.extend([start[2], end[2], None])
        
        fig = go.Figure(data=[
            # Original mesh
            go.Mesh3d(
                x=self.mesh.vertices[:, 0],
                y=self.mesh.vertices[:, 1],
                z=self.mesh.vertices[:, 2],
                i=self.mesh.faces[:, 0],
                j=self.mesh.faces[:, 1],
                k=self.mesh.faces[:, 2],
                opacity=0.3,
                color='lightgray'
            ),
            # Ray paths
            go.Scatter3d(
                x=lines_x,
                y=lines_y,
                z=lines_z,
                mode='lines',
                line=dict(color='red', width=1),
                name='Ray Paths'
            )
        ])
        
        fig.update_layout(
            scene=dict(aspectmode='data'),
            title='Ray Path Visualization',
            width=800,
            height=1000
        )
        
        return fig
    
    def generate_all_visualizations(self, paths, initial_partition, refined_partition, final_partition):
        """Generate all visualizations including the new convex hull view"""
        print("\nGenerating visualizations...")
        
        visualizations = {
            'Initial Segmentation': self.visualize_segmentation(initial_partition),
            'Initial Hulls': self.visualize_cluster_hulls(initial_partition),
            'Ray Paths': self.visualize_ray_paths(paths),
            'Refined Segmentation': self.visualize_segmentation(refined_partition),
            'Refined Hulls': self.visualize_cluster_hulls(refined_partition),
            'Final Segmentation': self.visualize_segmentation(final_partition),
            'Final Hulls': self.visualize_cluster_hulls(final_partition)
        }
        
        # Update titles to include partition counts
        num_initial = len(set(initial_partition.values()))
        num_refined = len(set(refined_partition.values()))
        num_final = len(set(final_partition.values()))
        
        visualizations['Initial Segmentation'].update_layout(title=f'Initial Segmentation ({num_initial} partitions)')
        visualizations['Refined Segmentation'].update_layout(title=f'Refined Segmentation ({num_refined} partitions)')
        visualizations['Final Segmentation'].update_layout(title=f'Final Segmentation ({num_final} partitions)')
        
        return tuple(visualizations.values())

    def analyze_and_segment(self, min_partition_size=3):
        """Complete analysis and segmentation workflow"""
        print("\nRunning analysis...")
        paths, total_hits, hits_per_bounce, ray_chains = self.analyze()
        
        print("\nBuilding transition graph...")
        G = self.build_transition_graph(ray_chains)
        
        print("\nPerforming initial clustering...")
        initial_partition = self.cluster_graph(G)
        
        print("\nRefining partitions based on connectivity...")
        refined_partition = self.refine_partition_with_connectivity(initial_partition)
        
        print(f"\nMerging partitions smaller than {min_partition_size} triangles...")
        final_partition = self.merge_small_partitions(refined_partition, min_partition_size)
        
        # Print partition statistics
        num_initial = len(set(initial_partition.values()))
        num_refined = len(set(refined_partition.values()))
        num_final = len(set(final_partition.values()))
        
        print(f"\nPartition counts:")
        print(f"Initial partitions: {num_initial}")
        print(f"After refinement: {num_refined}")
        print(f"After merging small partitions: {num_final}")
        
        return paths, total_hits, hits_per_bounce, initial_partition, refined_partition, final_partition

# Main execution
if __name__ == "__main__":

    #'mesh_path': '/Users/ashtonjenson/Downloads/Dragon.stl',
    # Configuration
    config = {
        'mesh_path': '/Users/ashtonjenson/Downloads/kyle.stl',
        'n_bounces': 3,
        'n_additional_rays': 50,
        'theta_degrees': 50.0,
        'resolution': 10,
        'min_size': 10,
        'edge_threshold': None
    }
    
    # Load mesh and time the process
    print("Loading mesh...")
    start_time = time.time()
    mesh = trimesh.load(config['mesh_path'])
    print(f"Mesh loaded in {time.time() - start_time:.2f} seconds")
    print(f"Mesh statistics:")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Surface area: {mesh.area:.2f}")
    print(f"Volume: {mesh.volume:.2f}")
    
    # Initialize analyzer
    analyzer = MeshBounceAnalyzer(
        mesh,
        n_bounces=config['n_bounces'],
        n_additional_rays=config['n_additional_rays'],
        theta_degrees=config['theta_degrees']
    )
    
    # Run analysis
    results = analyzer.analyze_and_segment(min_partition_size=config['min_size'])
    
    # Generate and display visualizations
    paths, total_hits, hits_per_bounce, initial_partition, refined_partition, final_partition = results
    
    print("\nGenerating and displaying visualizations...")
    
    # Generate visualizations
    initial_seg, initial_hulls, ray_paths, refined_seg, refined_hulls, final_seg, final_hulls = \
        analyzer.generate_all_visualizations(paths, initial_partition, refined_partition, final_partition)
    
    # Create list of all figures
    figures = [
        ("Initial Segmentation", initial_seg),
        ("Initial Convex Hulls", initial_hulls),
        ("Ray Paths", ray_paths),
        ("Refined Segmentation", refined_seg),
        ("Refined Convex Hulls", refined_hulls),
        ("Final Segmentation", final_seg),
        ("Final Convex Hulls", final_hulls)
    ]
    
    # Display each figure
    for name, fig in figures:
        print(f"\nDisplaying {name}...")
        fig.show()

    
    