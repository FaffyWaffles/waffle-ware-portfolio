/**
 * Mesh Bounce Analyzer - JavaScript Implementation
 * Advanced 3D mesh segmentation using ray-tracing and graph clustering
 */

// Utility classes and functions
class Logger {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }
    
    log(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const div = document.createElement('div');
        div.innerHTML = `<span style="color: #666;">[${timestamp}]</span> ${message}`;
        
        if (type === 'error') {
            div.style.color = '#dc3545';
        } else if (type === 'warning') {
            div.style.color = '#f39c12';
        } else if (type === 'success') {
            div.style.color = '#28a745';
        }
        
        this.container.appendChild(div);
        this.container.scrollTop = this.container.scrollHeight;
    }
}

class ProgressTracker {
    constructor(containerId, barId) {
        this.container = document.getElementById(containerId);
        this.bar = document.getElementById(barId);
    }
    
    show() {
        this.container.style.display = 'block';
    }
    
    hide() {
        this.container.style.display = 'none';
    }
    
    update(percentage) {
        this.bar.style.width = `${percentage}%`;
    }
}

class UnionFind {
    constructor(size) {
        this.parent = Array.from({length: size}, (_, i) => i);
        this.rank = new Array(size).fill(0);
    }
    
    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]); // Path compression
        }
        return this.parent[x];
    }
    
    union(x, y) {
        let px = this.find(x);
        let py = this.find(y);
        
        if (px === py) return;
        
        if (this.rank[px] < this.rank[py]) {
            [px, py] = [py, px];
        }
        
        this.parent[py] = px;
        if (this.rank[px] === this.rank[py]) {
            this.rank[px]++;
        }
    }
    
    getSets() {
        const sets = {};
        for (let i = 0; i < this.parent.length; i++) {
            const root = this.find(i);
            if (!(root in sets)) {
                sets[root] = [];
            }
            sets[root].push(i);
        }
        return Object.values(sets);
    }
}

// Self-contained Louvain clustering implementation
class SelfContainedLouvain {
    constructor(graph) {
        this.graph = graph;
        this.nodes = graph.getAllNodes();
        this.nodeCount = this.nodes.length;
        
        // Create node index mapping
        this.nodeToIndex = new Map();
        this.indexToNode = new Map();
        this.nodes.forEach((node, i) => {
            this.nodeToIndex.set(node, i);
            this.indexToNode.set(i, node);
        });
        
        // Build adjacency matrix and degree arrays
        this.adjacency = new Array(this.nodeCount);
        this.degrees = new Array(this.nodeCount).fill(0);
        this.totalWeight = 0;
        
        for (let i = 0; i < this.nodeCount; i++) {
            this.adjacency[i] = new Array(this.nodeCount).fill(0);
        }
        
        // Populate adjacency matrix
        for (const node of this.nodes) {
            const nodeIdx = this.nodeToIndex.get(node);
            const neighbors = this.graph.getNeighbors(node);
            
            for (const neighbor of neighbors) {
                const neighborIdx = this.nodeToIndex.get(neighbor);
                const weight = this.graph.getWeight(node, neighbor);
                
                this.adjacency[nodeIdx][neighborIdx] = weight;
                this.degrees[nodeIdx] += weight;
                this.totalWeight += weight;
            }
        }
        
        this.totalWeight /= 2; // Each edge counted twice
    }
    
    cluster(resolution = 1.0, maxIterations = 50) {
        // Initialize each node in its own community
        let communities = new Array(this.nodeCount);
        for (let i = 0; i < this.nodeCount; i++) {
            communities[i] = i;
        }
        
        let improved = true;
        let iteration = 0;
        
        while (improved && iteration < maxIterations) {
            improved = false;
            iteration++;
            
            // For each node, try moving it to neighboring communities
            for (let nodeIdx = 0; nodeIdx < this.nodeCount; nodeIdx++) {
                const currentCommunity = communities[nodeIdx];
                
                // Calculate gain for staying in current community
                const currentGain = this.calculateModularityGain(
                    nodeIdx, currentCommunity, communities, resolution
                );
                
                // Find neighboring communities
                const neighborCommunities = new Set();
                for (let neighborIdx = 0; neighborIdx < this.nodeCount; neighborIdx++) {
                    if (this.adjacency[nodeIdx][neighborIdx] > 0) {
                        neighborCommunities.add(communities[neighborIdx]);
                    }
                }
                
                // Try each neighboring community
                let bestCommunity = currentCommunity;
                let bestGain = currentGain;
                
                for (const community of neighborCommunities) {
                    if (community !== currentCommunity) {
                        const gain = this.calculateModularityGain(
                            nodeIdx, community, communities, resolution
                        );
                        
                        if (gain > bestGain) {
                            bestGain = gain;
                            bestCommunity = community;
                        }
                    }
                }
                
                // Move node if beneficial
                if (bestCommunity !== currentCommunity) {
                    communities[nodeIdx] = bestCommunity;
                    improved = true;
                }
            }
        }
        
        // Renumber communities to be consecutive
        const uniqueCommunities = [...new Set(communities)];
        const communityMap = new Map();
        uniqueCommunities.forEach((comm, i) => communityMap.set(comm, i));
        
        // Convert back to node-based partition
        const partition = {};
        for (let i = 0; i < this.nodeCount; i++) {
            const node = this.indexToNode.get(i);
            partition[node] = communityMap.get(communities[i]);
        }
        
        return partition;
    }
    
    calculateModularityGain(nodeIdx, targetCommunity, communities, resolution) {
        const nodeDegree = this.degrees[nodeIdx];
        
        // Calculate sum of weights to nodes in target community
        let weightToTarget = 0;
        for (let i = 0; i < this.nodeCount; i++) {
            if (communities[i] === targetCommunity) {
                weightToTarget += this.adjacency[nodeIdx][i];
            }
        }
        
        // Calculate total degree of target community
        let communityDegree = 0;
        for (let i = 0; i < this.nodeCount; i++) {
            if (communities[i] === targetCommunity) {
                communityDegree += this.degrees[i];
            }
        }
        
        if (this.totalWeight === 0) return 0;
        
        // Modularity gain calculation
        const gain = (weightToTarget / this.totalWeight) - 
                    (resolution * nodeDegree * communityDegree) / (2 * this.totalWeight * this.totalWeight);
        
        return gain;
    }
}

// Simple but effective graph implementation
class SimpleGraph {
    constructor() {
        this.nodes = new Set();
        this.edges = new Map();
        this.weights = new Map();
    }
    
    addNode(node) {
        this.nodes.add(node);
        if (!this.edges.has(node)) {
            this.edges.set(node, new Set());
        }
    }
    
    addEdge(from, to, weight = 1) {
        this.addNode(from);
        this.addNode(to);
        
        this.edges.get(from).add(to);
        this.edges.get(to).add(from); // Undirected
        
        const edgeKey = `${Math.min(from, to)}-${Math.max(from, to)}`;
        this.weights.set(edgeKey, (this.weights.get(edgeKey) || 0) + weight);
    }
    
    getWeight(from, to) {
        const edgeKey = `${Math.min(from, to)}-${Math.max(from, to)}`;
        return this.weights.get(edgeKey) || 0;
    }
    
    getNeighbors(node) {
        return Array.from(this.edges.get(node) || []);
    }
    
    getAllNodes() {
        return Array.from(this.nodes);
    }
    
    getNodeCount() {
        return this.nodes.size;
    }
    
    getEdgeCount() {
        return this.weights.size;
    }
}

// STL file loader
class STLLoader {
    static async loadFile(file) {
        const arrayBuffer = await file.arrayBuffer();
        return STLLoader.parseSTL(arrayBuffer);
    }
    
    static async loadDefault() {
        try {
            const response = await fetch('./Bunny.stl');
            const arrayBuffer = await response.arrayBuffer();
            return STLLoader.parseSTL(arrayBuffer);
        } catch (error) {
            throw new Error('Could not load default Bunny.stl file. Make sure it exists in the same directory.');
        }
    }
    
    static parseSTL(arrayBuffer) {
        const dataView = new DataView(arrayBuffer);
        
        // Skip header (80 bytes)
        const numTriangles = dataView.getUint32(80, true);
        
        const vertices = [];
        const faces = [];
        const normals = [];
        const vertexMap = new Map();
        let vertexIndex = 0;
        
        for (let i = 0; i < numTriangles; i++) {
            const offset = 84 + i * 50;
            
            // Read normal
            const normal = [
                dataView.getFloat32(offset, true),
                dataView.getFloat32(offset + 4, true),
                dataView.getFloat32(offset + 8, true)
            ];
            normals.push(normal);
            
            // Read vertices
            const faceIndices = [];
            for (let j = 0; j < 3; j++) {
                const vertexOffset = offset + 12 + j * 12;
                const vertex = [
                    dataView.getFloat32(vertexOffset, true),
                    dataView.getFloat32(vertexOffset + 4, true),
                    dataView.getFloat32(vertexOffset + 8, true)
                ];
                
                const vertexKey = vertex.join(',');
                if (!vertexMap.has(vertexKey)) {
                    vertexMap.set(vertexKey, vertexIndex);
                    vertices.push(vertex);
                    vertexIndex++;
                }
                faceIndices.push(vertexMap.get(vertexKey));
            }
            faces.push(faceIndices);
        }
        
        return { vertices, faces, normals };
    }
}

// Main mesh analyzer class
class MeshBounceAnalyzer {
    constructor(meshData, options = {}) {
        this.vertices = meshData.vertices;
        this.faces = meshData.faces;
        this.faceNormals = meshData.normals;
        
        this.nBounces = options.nBounces || 10;
        this.nAdditionalRays = options.nAdditionalRays || 4;
        this.theta = (options.thetaDegrees || 30.0) * Math.PI / 180;
        this.faceCount = this.faces.length;
        this.hitCounts = new Array(this.faceCount).fill(0);
        
        // Initialize logger and progress tracker first
        this.logger = new Logger('logContainer');
        this.progress = new ProgressTracker('progressContainer', 'progressBar');
        
        // Precompute face data
        this.precomputeFaceData();
        
        this.logger.log(`Initialized with ${this.faceCount} faces`);
        this.logger.log(`Number of bounces: ${this.nBounces}`);
        this.logger.log(`Additional rays per face: ${this.nAdditionalRays}`);
        this.logger.log(`Cone angle: ${options.thetaDegrees || 30.0} degrees`);
    }
    
    precomputeFaceData() {
        this.v0Array = [];
        this.edge1Array = [];
        this.edge2Array = [];
        this.faceCentroids = [];
        
        for (let i = 0; i < this.faces.length; i++) {
            const face = this.faces[i];
            const v0 = this.vertices[face[0]];
            const v1 = this.vertices[face[1]];
            const v2 = this.vertices[face[2]];
            
            this.v0Array.push(v0);
            this.edge1Array.push([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]]);
            this.edge2Array.push([v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]]);
            
            const centroid = [
                (v0[0] + v1[0] + v2[0]) / 3,
                (v0[1] + v1[1] + v2[1]) / 3,
                (v0[2] + v1[2] + v2[2]) / 3
            ];
            this.faceCentroids.push(centroid);
        }
        
        // Compute face adjacency (equivalent to trimesh's face_adjacency)
        this.faceAdjacency = this.computeFaceAdjacency();
        if (this.logger) {
            this.logger.log(`Computed ${this.faceAdjacency.length} face adjacency pairs`);
        } else {
            console.log(`Computed ${this.faceAdjacency.length} face adjacency pairs`);
        }
    }
    
    computeFaceAdjacency() {
        // This implements the equivalent of trimesh's face_adjacency
        // Two faces are adjacent if they share exactly one edge (two vertices)
        
        const adjacencyPairs = [];
        const edgeToFaces = new Map();
        
        // For each face, create edge keys and map them to face indices
        for (let faceIdx = 0; faceIdx < this.faces.length; faceIdx++) {
            const face = this.faces[faceIdx];
            
            // Create the three edges of this triangle
            const edges = [
                [face[0], face[1]], // edge 0-1
                [face[1], face[2]], // edge 1-2
                [face[2], face[0]]  // edge 2-0
            ];
            
            for (const edge of edges) {
                // Create a canonical edge key (smaller vertex index first)
                const edgeKey = edge[0] < edge[1] ? `${edge[0]}-${edge[1]}` : `${edge[1]}-${edge[0]}`;
                
                if (!edgeToFaces.has(edgeKey)) {
                    edgeToFaces.set(edgeKey, []);
                }
                edgeToFaces.get(edgeKey).push(faceIdx);
            }
        }
        
        // Find adjacent faces (faces that share an edge)
        for (const [edgeKey, faceList] of edgeToFaces) {
            if (faceList.length === 2) {
                // Exactly two faces share this edge - they are adjacent
                const [face1, face2] = faceList;
                adjacencyPairs.push([face1, face2]);
            }
            // Note: if faceList.length > 2, the mesh is non-manifold
            // if faceList.length === 1, it's a boundary edge
        }
        
        return adjacencyPairs;
    }
    
    buildFaceAdjacencyDict() {
        // Build adjacency dictionary from the precomputed face adjacency pairs
        // This exactly matches the Python version's approach
        const faceAdjDict = new Map();
        
        // Initialize adjacency map with empty arrays
        for (let i = 0; i < this.faceCount; i++) {
            faceAdjDict.set(i, []);
        }
        
        // Populate from face adjacency pairs (equivalent to mesh.face_adjacency)
        for (const [face1, face2] of this.faceAdjacency) {
            faceAdjDict.get(face1).push(face2);
            faceAdjDict.get(face2).push(face1);
        }
        
        if (this.logger) {
            this.logger.log(`Built face adjacency dictionary: ${this.faceAdjacency.length} adjacency pairs`);
        } else {
            console.log(`Built face adjacency dictionary: ${this.faceAdjacency.length} adjacency pairs`);
        }
        return faceAdjDict;
    }
    
    // Vector operations
    static dot(a, b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    
    static cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }
    
    static length(v) {
        return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    
    static normalize(v) {
        const len = MeshBounceAnalyzer.length(v);
        return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
    }
    
    static subtract(a, b) {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }
    
    static add(a, b) {
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    }
    
    static scale(v, s) {
        return [v[0] * s, v[1] * s, v[2] * s];
    }
    
    // Ray-triangle intersection using Möller–Trumbore algorithm
    rayTriangleIntersect(rayOrigin, rayDirection, faceIndex) {
        const epsilon = 1e-7;
        const edge1 = this.edge1Array[faceIndex];
        const edge2 = this.edge2Array[faceIndex];
        
        const pvec = MeshBounceAnalyzer.cross(rayDirection, edge2);
        const det = MeshBounceAnalyzer.dot(edge1, pvec);
        
        if (Math.abs(det) < epsilon) return null;
        
        const invDet = 1.0 / det;
        const tvec = MeshBounceAnalyzer.subtract(rayOrigin, this.v0Array[faceIndex]);
        const u = MeshBounceAnalyzer.dot(tvec, pvec) * invDet;
        
        if (u < -epsilon || u > 1.0 + epsilon) return null;
        
        const qvec = MeshBounceAnalyzer.cross(tvec, edge1);
        const v = MeshBounceAnalyzer.dot(rayDirection, qvec) * invDet;
        
        if (v < -epsilon || u + v > 1.0 + epsilon) return null;
        
        const t = MeshBounceAnalyzer.dot(edge2, qvec) * invDet;
        
        // Check if ray enters from back face (interior)
        const dotProduct = MeshBounceAnalyzer.dot(rayDirection, this.faceNormals[faceIndex]);
        const enteringFromBack = dotProduct > epsilon;
        
        if (t > epsilon && enteringFromBack) {
            const point = MeshBounceAnalyzer.add(rayOrigin, MeshBounceAnalyzer.scale(rayDirection, t));
            return { distance: t, point, faceIndex };
        }
        
        return null;
    }
    
    // Find closest intersection for a ray
    findClosestIntersection(rayOrigin, rayDirection) {
        let closestHit = null;
        let minDistance = Infinity;
        
        for (let faceIndex = 0; faceIndex < this.faceCount; faceIndex++) {
            const hit = this.rayTriangleIntersect(rayOrigin, rayDirection, faceIndex);
            if (hit && hit.distance < minDistance) {
                minDistance = hit.distance;
                closestHit = hit;
            }
        }
        
        return closestHit;
    }
    
    // Generate rotated rays within a cone - vectorized for all faces
    generateAllRotatedRays(baseDirections, nRays, theta) {
        const numFaces = baseDirections.length;
        const allDirections = [];
        
        // Add base directions first
        for (let faceIdx = 0; faceIdx < numFaces; faceIdx++) {
            allDirections.push(baseDirections[faceIdx]);
        }
        
        // Generate additional rays for each face
        for (let faceIdx = 0; faceIdx < numFaces; faceIdx++) {
            const baseDirection = baseDirections[faceIdx];
            
            for (let rayIdx = 0; rayIdx < nRays; rayIdx++) {
                const phi = theta * Math.sqrt(Math.random());
                const omega = 2 * Math.PI * Math.random();
                
                // Create perturbation in spherical coordinates
                const sinPhi = Math.sin(phi);
                const perturbation = [
                    sinPhi * Math.cos(omega),
                    sinPhi * Math.sin(omega),
                    Math.cos(phi)
                ];
                
                // Create rotation matrix to align with base direction
                const zAxis = [0, 0, 1];
                const rotationAxis = MeshBounceAnalyzer.cross(zAxis, baseDirection);
                const rotationAngle = Math.acos(Math.max(-1, Math.min(1, MeshBounceAnalyzer.dot(baseDirection, zAxis))));
                
                let rotatedDirection;
                if (MeshBounceAnalyzer.length(rotationAxis) > 1e-6) {
                    // Apply Rodrigues' rotation formula
                    const k = MeshBounceAnalyzer.normalize(rotationAxis);
                    const cosAngle = Math.cos(rotationAngle);
                    const sinAngle = Math.sin(rotationAngle);
                    
                    const kCrossP = MeshBounceAnalyzer.cross(k, perturbation);
                    const kDotP = MeshBounceAnalyzer.dot(k, perturbation);
                    
                    rotatedDirection = [
                        perturbation[0] * cosAngle + kCrossP[0] * sinAngle + k[0] * kDotP * (1 - cosAngle),
                        perturbation[1] * cosAngle + kCrossP[1] * sinAngle + k[1] * kDotP * (1 - cosAngle),
                        perturbation[2] * cosAngle + kCrossP[2] * sinAngle + k[2] * kDotP * (1 - cosAngle)
                    ];
                } else {
                    rotatedDirection = perturbation;
                }
                
                allDirections.push(MeshBounceAnalyzer.normalize(rotatedDirection));
            }
        }
        
        return allDirections;
    }
    
    // Batch ray-triangle intersection processing
    async batchRayTriangleIntersect(rayOrigins, rayDirections, showProgress = false) {
        const nRays = rayOrigins.length;
        const batchSize = 1000; // Process in smaller batches to prevent blocking
        const nBatches = Math.ceil(nRays / batchSize);
        
        // Initialize results arrays
        const hitFaces = new Array(nRays).fill(-1);
        const hitPoints = new Array(nRays);
        const distances = new Array(nRays).fill(Infinity);
        
        // Initialize hit points
        for (let i = 0; i < nRays; i++) {
            hitPoints[i] = [0, 0, 0];
        }
        
        // Process in batches
        for (let batchIdx = 0; batchIdx < nBatches; batchIdx++) {
            const startIdx = batchIdx * batchSize;
            const endIdx = Math.min((batchIdx + 1) * batchSize, nRays);
            
            if (showProgress && batchIdx % 10 === 0) {
                this.progress.update((batchIdx / nBatches) * 30);
                await new Promise(resolve => setTimeout(resolve, 1));
            }
            
            // Process each ray in the batch
            for (let rayIdx = startIdx; rayIdx < endIdx; rayIdx++) {
                const rayOrigin = rayOrigins[rayIdx];
                const rayDirection = rayDirections[rayIdx];
                
                let closestHit = null;
                let minDistance = Infinity;
                
                // Test against all faces
                for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
                    const hit = this.rayTriangleIntersect(rayOrigin, rayDirection, faceIdx);
                    if (hit && hit.distance < minDistance) {
                        minDistance = hit.distance;
                        closestHit = hit;
                    }
                }
                
                if (closestHit) {
                    hitFaces[rayIdx] = closestHit.faceIndex;
                    hitPoints[rayIdx] = [...closestHit.point];
                    distances[rayIdx] = closestHit.distance;
                }
            }
        }
        
        return { hitFaces, hitPoints, distances };
    }
    
    async analyze() {
        this.logger.log('Starting face-based analysis...', 'info');
        this.progress.show();
        
        let totalIntersections = 0;
        const hitsPerBounce = [];
        const rayChains = [];
        
        // Initialize ray chains for each face
        for (let i = 0; i < this.faceCount; i++) {
            rayChains.push([i]);
        }
        
        // Generate base directions for each face
        this.logger.log('Generating ray directions...');
        const baseOrigins = [...this.faceCentroids];
        const baseDirections = this.faceNormals.map(normal => 
            MeshBounceAnalyzer.scale(normal, -1)
        );
        
        // Generate ALL ray directions at once (matching Python approach)
        const allDirections = this.generateAllRotatedRays(baseDirections, this.nAdditionalRays, this.theta);
        
        // Replicate origins for all rays (base + additional rays per face)
        const totalRaysPerFace = this.nAdditionalRays + 1;
        const allOrigins = [];
        for (let i = 0; i < totalRaysPerFace; i++) {
            allOrigins.push(...baseOrigins);
        }
        
        this.logger.log(`Processing ${allDirections.length} rays for initial bounce...`);
        
        // Storage for visualization and tracking
        const pathSegments = [];
        for (let i = 0; i < this.faceCount; i++) {
            pathSegments.push([]);
        }
        
        // Process first bounce with all rays (matching Python's parallel approach)
        const { hitFaces, hitPoints, distances } = await this.batchRayTriangleIntersect(
            allOrigins, allDirections, true
        );
        
        // Reshape results to [num_faces, n_rays_per_face] (matching Python)
        const reshapedHitFaces = [];
        const reshapedHitPoints = [];
        const reshapedDistances = [];
        
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            const faceHits = [];
            const facePoints = [];
            const faceDistances = [];
            
            for (let rayIdx = 0; rayIdx < totalRaysPerFace; rayIdx++) {
                const globalRayIdx = rayIdx * this.faceCount + faceIdx;
                faceHits.push(hitFaces[globalRayIdx]);
                facePoints.push(hitPoints[globalRayIdx]);
                faceDistances.push(distances[globalRayIdx]);
            }
            
            reshapedHitFaces.push(faceHits);
            reshapedHitPoints.push(facePoints);
            reshapedDistances.push(faceDistances);
        }
        
        // For each face, find if any ray hit (matching Python logic)
        const currentOrigins = [...baseOrigins];
        const currentDirections = [...baseDirections];
        let validRays = new Array(this.faceCount).fill(false);
        let bounceHits = 0;
        
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            const faceHits = reshapedHitFaces[faceIdx];
            const facePoints = reshapedHitPoints[faceIdx];
            
            // Find if any ray hit (equivalent to torch.any(valid_hits, dim=1))
            const validHits = faceHits.map(hit => hit >= 0);
            const anyValidHits = validHits.some(hit => hit);
            
            if (anyValidHits) {
                // Get the first successful hit (equivalent to torch.argmax)
                const firstHitIndex = validHits.findIndex(hit => hit);
                const selectedHitFace = faceHits[firstHitIndex];
                const selectedHitPoint = facePoints[firstHitIndex];
                
                // Store path segment for visualization
                pathSegments[faceIdx].push({
                    origin: [...currentOrigins[faceIdx]],
                    endpoint: [...selectedHitPoint],
                    distance: reshapedDistances[faceIdx][firstHitIndex],
                    hitFace: selectedHitFace,
                    bounce: 0
                });
                
                // Update for next bounce
                currentOrigins[faceIdx] = selectedHitPoint;
                
                // Compute reflected direction
                const rayDirection = allDirections[firstHitIndex * this.faceCount + faceIdx];
                const normal = this.faceNormals[selectedHitFace];
                const dotProduct = MeshBounceAnalyzer.dot(rayDirection, normal);
                currentDirections[faceIdx] = MeshBounceAnalyzer.subtract(
                    rayDirection,
                    MeshBounceAnalyzer.scale(normal, 2 * dotProduct)
                );
                currentDirections[faceIdx] = MeshBounceAnalyzer.normalize(currentDirections[faceIdx]);
                
                this.hitCounts[selectedHitFace]++;
                rayChains[faceIdx].push(selectedHitFace);
                validRays[faceIdx] = true;
                bounceHits++;
            } else {
                // No successful hits for this face
                rayChains[faceIdx].push(-1);
                
                // Store unsuccessful ray for visualization
                const rayLength = 0.05;
                const endpoint = MeshBounceAnalyzer.add(
                    currentOrigins[faceIdx],
                    MeshBounceAnalyzer.scale(baseDirections[faceIdx], rayLength)
                );
                pathSegments[faceIdx].push({
                    origin: [...currentOrigins[faceIdx]],
                    endpoint: endpoint,
                    distance: rayLength,
                    hitFace: -1,
                    bounce: 0,
                    missed: true
                });
            }
        }
        
        hitsPerBounce.push(bounceHits);
        totalIntersections += bounceHits;
        this.logger.log(`Bounce 0: ${bounceHits} hits (${(bounceHits/this.faceCount*100).toFixed(1)}% of rays)`);
        
        // Continue with regular bounces (one ray per face, as in Python)
        this.logger.log('Processing subsequent bounces...');
        for (let bounce = 1; bounce < this.nBounces; bounce++) {
            this.progress.update(30 + (bounce / this.nBounces) * 60);
            
            bounceHits = 0;
            
            for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
                if (!validRays[faceIdx]) {
                    rayChains[faceIdx].push(-1);
                    continue;
                }
                
                const hit = this.findClosestIntersection(
                    currentOrigins[faceIdx], 
                    currentDirections[faceIdx]
                );
                
                if (hit) {
                    // Store path segment for visualization
                    pathSegments[faceIdx].push({
                        origin: [...currentOrigins[faceIdx]],
                        endpoint: [...hit.point],
                        distance: hit.distance,
                        hitFace: hit.faceIndex,
                        bounce: bounce
                    });
                    
                    currentOrigins[faceIdx] = hit.point;
                    
                    const normal = this.faceNormals[hit.faceIndex];
                    const dotProduct = MeshBounceAnalyzer.dot(currentDirections[faceIdx], normal);
                    currentDirections[faceIdx] = MeshBounceAnalyzer.subtract(
                        currentDirections[faceIdx],
                        MeshBounceAnalyzer.scale(normal, 2 * dotProduct)
                    );
                    currentDirections[faceIdx] = MeshBounceAnalyzer.normalize(currentDirections[faceIdx]);
                    
                    this.hitCounts[hit.faceIndex]++;
                    rayChains[faceIdx].push(hit.faceIndex);
                    bounceHits++;
                } else {
                    validRays[faceIdx] = false;
                    rayChains[faceIdx].push(-1);
                    
                    // Store final unsuccessful ray segment
                    const rayLength = 0.02;
                    const endpoint = MeshBounceAnalyzer.add(
                        currentOrigins[faceIdx],
                        MeshBounceAnalyzer.scale(currentDirections[faceIdx], rayLength)
                    );
                    pathSegments[faceIdx].push({
                        origin: [...currentOrigins[faceIdx]],
                        endpoint: endpoint,
                        distance: rayLength,
                        hitFace: -1,
                        bounce: bounce,
                        missed: true
                    });
                }
            }
            
            hitsPerBounce.push(bounceHits);
            totalIntersections += bounceHits;
            this.logger.log(`Bounce ${bounce}: ${bounceHits} hits (${(bounceHits/this.faceCount*100).toFixed(1)}% of rays)`);
            
            if (bounceHits === 0) {
                this.logger.log('No more hits, stopping early', 'warning');
                break;
            }
            
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        
        // Report unsuccessful faces
        const unsuccessfulFaces = rayChains.filter((chain, i) => 
            !chain.slice(1).some(face => face >= 0)
        ).length;
        
        if (unsuccessfulFaces > 0) {
            this.logger.log(`Warning: ${unsuccessfulFaces} faces never achieved a successful raycast`, 'warning');
        }
        
        this.progress.update(90);
        return { allPaths: pathSegments, totalIntersections, hitsPerBounce, rayChains };
    }
    
    buildTransitionGraph(rayChains) {
        this.logger.log('Building transition graph...');
        const graph = new AdvancedGraph();
        const transitionCounts = new Map();
        
        for (const chain of rayChains) {
            for (let i = 0; i < chain.length - 1; i++) {
                const fromFace = chain[i];
                const toFace = chain[i + 1];
                
                if (fromFace >= 0 && toFace >= 0) {
                    const key = `${fromFace}-${toFace}`;
                    transitionCounts.set(key, (transitionCounts.get(key) || 0) + 1);
                }
            }
        }
        
        for (const [key, count] of transitionCounts) {
            const [from, to] = key.split('-').map(Number);
            graph.addEdge(from, to, count);
        }
        
        this.logger.log(`Graph built with ${graph.getNodeCount()} nodes and ${graph.getEdgeCount()} edges`);
        return graph;
    }
    
    analyzeClusteringResolutions(graph, resolutions = [0.5, 1.0, 2.0, 4.0]) {
        this.logger.log('Analyzing clustering resolutions...');
        const results = {};
        
        for (const resolution of resolutions) {
            // Note: jLouvain doesn't directly support resolution parameter
            // We'll use edge thresholding as a proxy for resolution
            const edgeThreshold = resolution > 1.0 ? Math.floor(resolution) : null;
            const partition = this.clusterGraphWithLouvain(graph, resolution, null, edgeThreshold);
            const nClusters = new Set(Object.values(partition)).size;
            
            // Calculate cluster sizes
            const clusterSizes = new Map();
            for (const cluster of Object.values(partition)) {
                clusterSizes.set(cluster, (clusterSizes.get(cluster) || 0) + 1);
            }
            
            const sizes = Array.from(clusterSizes.values());
            
            // Calculate simplified modularity
            let modularity = 0;
            try {
                modularity = this.calculateSimpleModularity(graph, partition);
            } catch (error) {
                modularity = 0;
                this.logger.log(`Could not calculate modularity: ${error.message}`, 'warning');
            }
            
            results[resolution] = {
                nClusters: nClusters,
                minSize: sizes.length > 0 ? Math.min(...sizes) : 0,
                maxSize: sizes.length > 0 ? Math.max(...sizes) : 0,
                avgSize: sizes.length > 0 ? sizes.reduce((a, b) => a + b, 0) / sizes.length : 0,
                modularity: modularity
            };
        }
        
        // Print results
        this.logger.log('\nClustering Resolution Analysis:');
        this.logger.log('Resolution | Clusters | Min Size | Max Size | Avg Size | Modularity');
        this.logger.log('-----------|----------|----------|----------|----------|----------');
        for (const [res, stats] of Object.entries(results)) {
            this.logger.log(`${res.padStart(9)} | ${stats.nClusters.toString().padStart(8)} | ${stats.minSize.toString().padStart(8)} | ${stats.maxSize.toString().padStart(8)} | ${stats.avgSize.toFixed(1).padStart(8)} | ${stats.modularity.toFixed(3)}`);
        }
        
        return results;
    }
    
    calculateSimpleModularity(graph, partition) {
        // Simplified modularity calculation
        // Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
        
        const nodes = graph.getAllNodes();
        const m = graph.getEdgeCount(); // Total number of edges
        
        if (m === 0) return 0;
        
        let modularity = 0;
        const degrees = new Map();
        
        // Calculate degrees
        for (const node of nodes) {
            degrees.set(node, graph.getNeighbors(node).length);
        }
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeI = nodes[i];
                const nodeJ = nodes[j];
                
                if (partition[nodeI] === partition[nodeJ]) {
                    const A_ij = graph.getWeight(nodeI, nodeJ) > 0 ? 1 : 0;
                    const k_i = degrees.get(nodeI) || 0;
                    const k_j = degrees.get(nodeJ) || 0;
                    
                    modularity += A_ij - (k_i * k_j) / (2 * m);
                }
            }
        }
        
        return modularity / (2 * m);
    }
    
    buildTransitionGraph(rayChains) {
        this.logger.log('Building transition graph...');
        const graph = new SimpleGraph();
        const transitionCounts = new Map();
        
        for (const chain of rayChains) {
            for (let i = 0; i < chain.length - 1; i++) {
                const fromFace = chain[i];
                const toFace = chain[i + 1];
                
                if (fromFace >= 0 && toFace >= 0) {
                    const key = `${fromFace}-${toFace}`;
                    transitionCounts.set(key, (transitionCounts.get(key) || 0) + 1);
                }
            }
        }
        
        for (const [key, count] of transitionCounts) {
            const [from, to] = key.split('-').map(Number);
            graph.addEdge(from, to, count);
        }
        
        this.logger.log(`Graph built with ${graph.getNodeCount()} nodes and ${graph.getEdgeCount()} edges`);
        return graph;
    }
    
    clusterGraphWithLouvain(graph, resolution = 1.0, minClusterSize = null, edgeThreshold = null) {
        this.logger.log(`Performing jLouvain clustering with resolution ${resolution}...`);
        
        // Debug: Log graph statistics
        this.logger.log(`Graph stats: ${graph.getNodeCount()} nodes, ${graph.getEdgeCount()} edges`);
        
        // Check if we have any edges
        if (graph.getEdgeCount() === 0) {
            this.logger.log('No edges in graph, creating trivial clustering', 'warning');
            const nodes = graph.getAllNodes();
            const partition = {};
            nodes.forEach((node, index) => {
                partition[node] = Math.floor(index / 50); // Group every 50 nodes
            });
            return partition;
        }
        
        // Convert our graph to jLouvain format
        const nodes = graph.getAllNodes();
        const edges = [];
        
        // Build edge list for jLouvain
        const processedEdges = new Set();
        for (const node of nodes) {
            const neighbors = graph.getNeighbors(node);
            for (const neighbor of neighbors) {
                const edgeKey = `${Math.min(node, neighbor)}-${Math.max(node, neighbor)}`;
                if (!processedEdges.has(edgeKey)) {
                    processedEdges.add(edgeKey);
                    const weight = graph.getWeight(node, neighbor);
                    
                    // Apply edge threshold if specified
                    if (edgeThreshold === null || weight >= edgeThreshold) {
                        edges.push({
                            source: node.toString(),
                            target: neighbor.toString(),
                            weight: weight
                        });
                    }
                }
            }
        }
        
        this.logger.log(`Prepared ${edges.length} edges for jLouvain (${processedEdges.size} total, threshold: ${edgeThreshold || 'none'})`);
        
        // Log some edge weights for debugging
        let totalWeight = 0;
        let minWeight = Infinity;
        let maxWeight = -Infinity;
        
        for (let i = 0; i < Math.min(edges.length, 5); i++) {
            const edge = edges[i];
            totalWeight += edge.weight;
            minWeight = Math.min(minWeight, edge.weight);
            maxWeight = Math.max(maxWeight, edge.weight);
            this.logger.log(`Edge ${edge.source}-${edge.target}: weight=${edge.weight}`);
        }
        
        if (edges.length > 0) {
            const avgWeight = totalWeight / Math.min(edges.length, 5);
            this.logger.log(`Sample edge weights: min=${minWeight}, max=${maxWeight}, avg=${avgWeight.toFixed(2)}`);
        }
        
        // Apply jLouvain clustering
        let partition;
        try {
            this.logger.log('Running jLouvain algorithm...');
            
            // Check if jLouvain is available
            if (typeof jLouvain === 'undefined') {
                throw new Error('jLouvain library not available');
            }
            
            const community = jLouvain();
            
            // Note: jLouvain doesn't directly support resolution parameter
            // We'll apply it as a post-processing step if needed
            const rawPartition = community
                .nodes(nodes.map(n => n.toString()))
                .edges(edges)();
            
            // Convert string keys back to numbers
            partition = {};
            for (const [nodeStr, cluster] of Object.entries(rawPartition)) {
                partition[parseInt(nodeStr)] = cluster;
            }
            
            this.logger.log('jLouvain algorithm completed successfully');
            
        } catch (error) {
            this.logger.log(`jLouvain clustering failed: ${error.message}. Using weight-based fallback.`, 'warning');
            return this.fallbackClusteringBetter(graph);
        }
        
        // Debug: Log partition statistics
        const clusterCounts = new Map();
        for (const cluster of Object.values(partition)) {
            clusterCounts.set(cluster, (clusterCounts.get(cluster) || 0) + 1);
        }
        
        this.logger.log(`Raw jLouvain result: ${clusterCounts.size} clusters`);
        const sizes = Array.from(clusterCounts.values());
        if (sizes.length > 0) {
            this.logger.log(`Cluster sizes: min=${Math.min(...sizes)}, max=${Math.max(...sizes)}, avg=${(sizes.reduce((a,b) => a+b, 0) / sizes.length).toFixed(1)}`);
        }
        
        // Post-process clusters based on minimum size if specified
        if (minClusterSize !== null) {
            partition = this.mergeSmallClusters(graph, partition, minClusterSize);
        }
        
        const numClusters = new Set(Object.values(partition)).size;
        this.logger.log(`jLouvain clustering complete: ${numClusters} final clusters`);
        
        return partition;
    }
    
    fallbackClusteringBetter(graph) {
        this.logger.log('Using improved fallback clustering based on edge weights and connectivity');
        
        const nodes = graph.getAllNodes();
        const partition = {};
        const visited = new Set();
        let clusterId = 0;
        
        // First, try weight-based clustering for better results
        if (this.useWeightBasedClustering(graph, partition)) {
            return partition;
        }
        
        // Fallback to connected components
        for (const startNode of nodes) {
            if (visited.has(startNode)) continue;
            
            // BFS to find connected component, but limit size for better clustering
            const queue = [startNode];
            const component = [];
            const maxComponentSize = Math.min(200, Math.ceil(nodes.length / 25)); // Target ~25+ clusters
            
            while (queue.length > 0 && component.length < maxComponentSize) {
                const node = queue.shift();
                if (visited.has(node)) continue;
                
                visited.add(node);
                component.push(node);
                
                // Add neighbors, but prioritize high-weight connections
                const neighbors = graph.getNeighbors(node);
                const weightedNeighbors = neighbors
                    .map(neighbor => ({
                        node: neighbor,
                        weight: graph.getWeight(node, neighbor)
                    }))
                    .sort((a, b) => b.weight - a.weight); // Sort by weight descending
                
                for (const {node: neighbor} of weightedNeighbors) {
                    if (!visited.has(neighbor) && component.length < maxComponentSize) {
                        queue.push(neighbor);
                    }
                }
            }
            
            // Assign cluster ID to all nodes in this component
            for (const node of component) {
                partition[node] = clusterId;
            }
            
            this.logger.log(`Cluster ${clusterId}: ${component.length} nodes`);
            clusterId++;
            
            // If we've processed most nodes, stop to avoid tiny clusters
            if (visited.size > nodes.length * 0.95) {
                break;
            }
        }
        
        // Assign remaining unvisited nodes to existing clusters based on best connections
        for (const node of nodes) {
            if (!visited.has(node)) {
                const neighbors = graph.getNeighbors(node);
                if (neighbors.length > 0) {
                    // Find the cluster with strongest connection
                    const clusterWeights = new Map();
                    for (const neighbor of neighbors) {
                        if (neighbor in partition) {
                            const cluster = partition[neighbor];
                            const weight = graph.getWeight(node, neighbor);
                            clusterWeights.set(cluster, (clusterWeights.get(cluster) || 0) + weight);
                        }
                    }
                    
                    if (clusterWeights.size > 0) {
                        const bestCluster = Array.from(clusterWeights.entries())
                            .sort((a, b) => b[1] - a[1])[0][0];
                        partition[node] = bestCluster;
                    } else {
                        partition[node] = clusterId++;
                    }
                } else {
                    partition[node] = clusterId++;
                }
            }
        }
        
        const finalClusters = new Set(Object.values(partition)).size;
        this.logger.log(`Fallback clustering complete: ${finalClusters} clusters created`);
        
        return partition;
    }
    
    useWeightBasedClustering(graph, partition) {
        this.logger.log('Attempting weight-based clustering...');
        
        const nodes = graph.getAllNodes();
        
        // Create clusters based on edge weight thresholds
        const highWeightThreshold = 5; // Edges with weight >= 5 indicate strong connections
        const mediumWeightThreshold = 2; // Edges with weight >= 2 indicate moderate connections
        
        const strongEdges = [];
        const mediumEdges = [];
        
        // Collect strong and medium edges
        for (const node of nodes) {
            for (const neighbor of graph.getNeighbors(node)) {
                if (node < neighbor) { // Avoid duplicates
                    const weight = graph.getWeight(node, neighbor);
                    if (weight >= highWeightThreshold) {
                        strongEdges.push([node, neighbor, weight]);
                    } else if (weight >= mediumWeightThreshold) {
                        mediumEdges.push([node, neighbor, weight]);
                    }
                }
            }
        }
        
        this.logger.log(`Found ${strongEdges.length} strong edges (weight >= ${highWeightThreshold})`);
        this.logger.log(`Found ${mediumEdges.length} medium edges (weight >= ${mediumWeightThreshold})`);
        
        if (strongEdges.length === 0 && mediumEdges.length === 0) {
            this.logger.log('No strong connections found, skipping weight-based clustering');
            return false;
        }
        
        // Use Union-Find to create clusters from strong edges first
        const uf = new UnionFind(nodes.length);
        const nodeToIndex = new Map();
        nodes.forEach((node, index) => nodeToIndex.set(node, index));
        
        // Process strong edges first
        for (const [node1, node2, weight] of strongEdges) {
            const idx1 = nodeToIndex.get(node1);
            const idx2 = nodeToIndex.get(node2);
            uf.union(idx1, idx2);
        }
        
        // Then process medium edges, but only if it doesn't create clusters that are too large
        for (const [node1, node2, weight] of mediumEdges) {
            const idx1 = nodeToIndex.get(node1);
            const idx2 = nodeToIndex.get(node2);
            
            // Check if merging would create a cluster that's too large
            const root1 = uf.find(idx1);
            const root2 = uf.find(idx2);
            
            if (root1 !== root2) {
                // Count current cluster sizes (simplified approximation)
                uf.union(idx1, idx2);
            }
        }
        
        // Convert Union-Find result to partition
        const components = uf.getSets();
        
        for (let clusterId = 0; clusterId < components.length; clusterId++) {
            for (const nodeIndex of components[clusterId]) {
                const node = nodes[nodeIndex];
                partition[node] = clusterId;
            }
        }
        
        this.logger.log(`Weight-based clustering created ${components.length} clusters`);
        
        // Only use this result if it creates a reasonable number of clusters
        if (components.length >= 10 && components.length <= nodes.length / 10) {
            return true;
        } else {
            this.logger.log(`Cluster count ${components.length} not reasonable, falling back to connectivity-based`);
            return false;
        }
    }
    
    fallbackClustering(graph) {
        // Simple fallback clustering if Louvain fails
        const nodes = graph.getAllNodes();
        const partition = {};
        nodes.forEach((node, index) => {
            partition[node] = Math.floor(index / 100); // Simple grouping
        });
        return partition;
    }
    
    mergeSmallClusters(graph, partition, minClusterSize) {
        this.logger.log(`Merging clusters smaller than ${minClusterSize} nodes...`);
        
        // Group nodes by cluster
        const clusters = new Map();
        for (const [node, cluster] of Object.entries(partition)) {
            if (!clusters.has(cluster)) {
                clusters.set(cluster, []);
            }
            clusters.get(cluster).push(parseInt(node));
        }
        
        // Sort clusters by size
        const sortedClusters = Array.from(clusters.entries())
            .sort((a, b) => a[1].length - b[1].length);
        
        const newPartition = { ...partition };
        let nextClusterId = Math.max(...Object.values(partition)) + 1;
        
        for (const [clusterId, nodes] of sortedClusters) {
            if (nodes.length < minClusterSize) {
                // Find the most connected neighboring cluster
                const neighborConnections = new Map();
                
                for (const node of nodes) {
                    for (const neighbor of graph.getNeighbors(node)) {
                        const neighborCluster = newPartition[neighbor];
                        if (neighborCluster !== clusterId) {
                            const weight = graph.getWeight(node, neighbor);
                            neighborConnections.set(neighborCluster, 
                                (neighborConnections.get(neighborCluster) || 0) + weight);
                        }
                    }
                }
                
                if (neighborConnections.size > 0) {
                    // Merge with the most connected cluster
                    const bestNeighbor = Array.from(neighborConnections.entries())
                        .sort((a, b) => b[1] - a[1])[0][0];
                    
                    for (const node of nodes) {
                        newPartition[node] = bestNeighbor;
                    }
                } else {
                    // If no neighbors, create a new cluster
                    for (const node of nodes) {
                        newPartition[node] = nextClusterId;
                    }
                    nextClusterId++;
                }
            }
        }
        
        const initialClusters = clusters.size;
        const finalClusters = new Set(Object.values(newPartition)).size;
        this.logger.log(`Merged small clusters: ${initialClusters} -> ${finalClusters} clusters`);
        
        return newPartition;
    }
    
    refinePartitionWithConnectivity(partition) {
        this.logger.log('Refining partitions based on physical connectivity...');
        const faceAdjDict = this.buildFaceAdjacencyDict();
        
        // Group faces by partition
        const partitionGroups = new Map();
        for (const [face, cluster] of Object.entries(partition)) {
            const faceNum = parseInt(face);
            const clusterNum = parseInt(cluster);
            if (!partitionGroups.has(clusterNum)) {
                partitionGroups.set(clusterNum, []);
            }
            partitionGroups.get(clusterNum).push(faceNum);
        }
        
        let newClusterId = Math.max(...Object.values(partition)) + 1;
        const newPartition = { ...partition };
        
        // Process each partition separately
        for (const [clusterId, faces] of partitionGroups) {
            const uf = new UnionFind(faces.length);
            const faceToLocal = new Map();
            faces.forEach((face, i) => faceToLocal.set(face, i));
            
            // Union adjacent faces within this partition
            for (let i = 0; i < faces.length; i++) {
                const face = faces[i];
                const adjacentFaces = faceAdjDict.get(face) || [];
                
                for (const adjFace of adjacentFaces) {
                    if (faceToLocal.has(adjFace)) {
                        uf.union(i, faceToLocal.get(adjFace));
                    }
                }
            }
            
            // Get connected components
            const connectedComponents = uf.getSets();
            
            // Assign new cluster IDs to additional components
            if (connectedComponents.length > 1) {
                for (let compIndex = 1; compIndex < connectedComponents.length; compIndex++) {
                    for (const localIdx of connectedComponents[compIndex]) {
                        const globalFaceIdx = faces[localIdx];
                        newPartition[globalFaceIdx] = newClusterId;
                    }
                    newClusterId++;
                }
            }
        }
        
        const initialClusters = Object.keys(partitionGroups).length;
        const finalClusters = new Set(Object.values(newPartition)).size;
        this.logger.log(`Refined ${initialClusters} clusters into ${finalClusters} connected components`);
        
        return newPartition;
    }
    
    mergeSmallPartitions(partition, minPartitionSize = 3, iteration = 1) {
        this.logger.log(`Merging small partitions (iteration ${iteration}, min size: ${minPartitionSize})...`);
        
        const newPartition = { ...partition };
        const faceAdjDict = this.buildFaceAdjacencyDict();
        
        // Ensure all faces have a partition
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            if (!(faceIdx in newPartition)) {
                newPartition[faceIdx] = Math.max(...Object.values(partition)) + 1;
            }
        }
        
        // Group faces by partition
        const partitionGroups = new Map();
        for (const [face, cluster] of Object.entries(newPartition)) {
            const faceNum = parseInt(face);
            const clusterNum = parseInt(cluster);
            if (!partitionGroups.has(clusterNum)) {
                partitionGroups.set(clusterNum, []);
            }
            partitionGroups.get(clusterNum).push(faceNum);
        }
        
        // Sort partitions by size (smallest first)
        const partitionSizes = Array.from(partitionGroups.entries())
            .map(([clusterId, faces]) => [clusterId, faces.length])
            .sort((a, b) => a[1] - b[1]);
        
        let smallPartitionsProcessed = 0;
        
        for (const [clusterId, size] of partitionSizes) {
            if (size < minPartitionSize) {
                smallPartitionsProcessed++;
                const faces = partitionGroups.get(clusterId);
                
                // Process each face independently
                for (const faceIdx of faces) {
                    const partnerFaces = faces.filter(f => f !== faceIdx);
                    const adjacentFaces = faceAdjDict.get(faceIdx) || [];
                    
                    // Find largest adjacent partition (excluding partner faces)
                    const adjacentPartitions = new Map();
                    for (const adjFace of adjacentFaces) {
                        if (!partnerFaces.includes(adjFace)) {
                            const adjPartition = newPartition[adjFace];
                            if (adjPartition !== clusterId) {
                                const partitionSize = partitionGroups.get(adjPartition)?.length || 0;
                                adjacentPartitions.set(adjPartition, partitionSize);
                            }
                        }
                    }
                    
                    if (adjacentPartitions.size > 0) {
                        const largestPartition = Array.from(adjacentPartitions.entries())
                            .sort((a, b) => b[1] - a[1])[0][0];
                        newPartition[faceIdx] = largestPartition;
                        
                        // Update partition groups
                        partitionGroups.get(largestPartition).push(faceIdx);
                        const currentGroup = partitionGroups.get(clusterId);
                        const index = currentGroup.indexOf(faceIdx);
                        if (index > -1) {
                            currentGroup.splice(index, 1);
                        }
                    }
                }
            }
        }
        
        // Renumber partitions to be consecutive
        const uniquePartitions = [...new Set(Object.values(newPartition))].sort((a, b) => a - b);
        const partitionMap = new Map();
        uniquePartitions.forEach((oldId, newId) => partitionMap.set(oldId, newId));
        
        const finalPartition = {};
        for (const [face, cluster] of Object.entries(newPartition)) {
            finalPartition[face] = partitionMap.get(cluster);
        }
        
        // Check if small partitions still exist
        const finalGroups = new Map();
        for (const [face, cluster] of Object.entries(finalPartition)) {
            const clusterNum = parseInt(cluster);
            if (!finalGroups.has(clusterNum)) {
                finalGroups.set(clusterNum, []);
            }
            finalGroups.get(clusterNum).push(parseInt(face));
        }
        
        const hasSmallPartitions = Array.from(finalGroups.values())
            .some(faces => faces.length < minPartitionSize);
        
        this.logger.log(`Iteration ${iteration}: processed ${smallPartitionsProcessed} small partitions`);
        
        if (hasSmallPartitions && iteration < 5) {
            return this.mergeSmallPartitions(finalPartition, minPartitionSize, iteration + 1);
        }
        
        const finalClusters = finalGroups.size;
        this.logger.log(`Merge complete: ${finalClusters} final partitions`);
        
        return finalPartition;
    }
    
    clusterGraph(graph, resolution = 1.0) {
        this.logger.log(`Performing graph clustering with resolution ${resolution}...`);
        
        // Use the advanced Louvain clustering
        const partition = this.clusterGraphWithLouvain(graph, resolution);
        
        const numClusters = new Set(Object.values(partition)).size;
        this.logger.log(`Graph clustering complete: ${numClusters} clusters`);
        
        return partition;
    }
    
    async analyzeAndSegment(minPartitionSize = 3) {
        this.logger.log('Running complete analysis and segmentation workflow...', 'info');
        
        const { allPaths, totalIntersections, hitsPerBounce, rayChains } = await this.analyze();
        
        this.progress.update(92);
        const graph = this.buildTransitionGraph(rayChains);
        
        // Analyze different clustering resolutions (matching Python)
        this.progress.update(93);
        this.analyzeClusteringResolutions(graph, [0.5, 1.0, 2.0, 4.0]);
        
        this.progress.update(94);
        const initialPartition = this.clusterGraph(graph);
        
        this.progress.update(96);
        const refinedPartition = this.refinePartitionWithConnectivity(initialPartition);
        
        this.progress.update(98);
        const finalPartition = this.mergeSmallPartitions(refinedPartition, minPartitionSize);
        
        this.progress.update(100);
        
        // Calculate statistics
        const stats = this.calculateStatistics(totalIntersections, hitsPerBounce, 
            initialPartition, refinedPartition, finalPartition);
        
        this.progress.hide();
        this.logger.log('Analysis complete!', 'success');
        
        return {
            paths: allPaths,
            totalIntersections,
            hitsPerBounce,
            initialPartition,
            refinedPartition,
            finalPartition,
            stats
        };
    }
    
    calculateStatistics(totalIntersections, hitsPerBounce, initialPartition, refinedPartition, finalPartition) {
        const hitCounts = this.hitCounts;
        
        const rayStats = {
            totalFaces: this.faceCount,
            totalRaySegments: this.faceCount * this.nBounces,
            validIntersections: totalIntersections,
            hitRate: (totalIntersections / (this.faceCount * this.nBounces) * 100).toFixed(1),
            avgHitsPerFace: (hitCounts.reduce((a, b) => a + b, 0) / hitCounts.length).toFixed(2),
            minHits: Math.min(...hitCounts),
            maxHits: Math.max(...hitCounts),
            faceCoverage: ((hitCounts.filter(h => h > 0).length / hitCounts.length) * 100).toFixed(1)
        };
        
        const getPartitionStats = (partition) => {
            const clusters = [...new Set(Object.values(partition))];
            const sizes = clusters.map(c => 
                Object.values(partition).filter(v => v === c).length
            );
            return {
                numClusters: clusters.length,
                avgSize: (sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(1),
                minSize: Math.min(...sizes),
                maxSize: Math.max(...sizes)
            };
        };
        
        return {
            rayStats,
            initialPartition: getPartitionStats(initialPartition),
            refinedPartition: getPartitionStats(refinedPartition),
            finalPartition: getPartitionStats(finalPartition),
            hitsPerBounce
        };
    }
    
    visualizeSegmentation(partition, title) {
        const numClusters = Math.max(...Object.values(partition)) + 1;
        const colors = this.generateColors(numClusters);
        
        const traces = [];
        
        // Group faces by cluster
        const clusterFaces = new Map();
        for (const [face, cluster] of Object.entries(partition)) {
            const faceNum = parseInt(face);
            const clusterNum = parseInt(cluster);
            if (!clusterFaces.has(clusterNum)) {
                clusterFaces.set(clusterNum, []);
            }
            clusterFaces.get(clusterNum).push(faceNum);
        }
        
        // Create a trace for each cluster
        for (const [clusterId, faceIndices] of clusterFaces) {
            const clusterVertices = [];
            const clusterFaces = [];
            const vertexMap = new Map();
            let vertexCount = 0;
            
            for (const faceIdx of faceIndices) {
                const face = this.faces[faceIdx];
                const mappedFace = [];
                
                for (const vertexIdx of face) {
                    if (!vertexMap.has(vertexIdx)) {
                        vertexMap.set(vertexIdx, vertexCount);
                        clusterVertices.push(this.vertices[vertexIdx]);
                        vertexCount++;
                    }
                    mappedFace.push(vertexMap.get(vertexIdx));
                }
                clusterFaces.push(mappedFace);
            }
            
            if (clusterVertices.length > 0) {
                const x = clusterVertices.map(v => v[0]);
                const y = clusterVertices.map(v => v[1]);
                const z = clusterVertices.map(v => v[2]);
                const i = clusterFaces.map(f => f[0]);
                const j = clusterFaces.map(f => f[1]);
                const k = clusterFaces.map(f => f[2]);
                
                traces.push({
                    type: 'mesh3d',
                    x: x,
                    y: y,
                    z: z,
                    i: i,
                    j: j,
                    k: k,
                    color: colors[clusterId],
                    opacity: 0.8,
                    name: `Cluster ${clusterId}`,
                    showlegend: false
                });
            }
        }
        
        const layout = {
            title: title,
            scene: {
                aspectmode: 'data',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            width: 600,
            height: 500
        };
        
        return { data: traces, layout };
    }
    
    generateColors(numColors) {
        const colors = [];
        for (let i = 0; i < numColors; i++) {
            const hue = (i * 360 / numColors) % 360;
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        return colors;
    }
    
    visualizeRayPaths(pathSegments, maxPaths = 500) {
        this.logger.log(`Visualizing ray paths for ${Math.min(maxPaths, this.faceCount)} faces...`);
        
        // Sample faces if needed
        const sampleIndices = [];
        if (this.faceCount > maxPaths) {
            const step = Math.floor(this.faceCount / maxPaths);
            for (let i = 0; i < this.faceCount; i += step) {
                sampleIndices.push(i);
            }
        } else {
            for (let i = 0; i < this.faceCount; i++) {
                sampleIndices.push(i);
            }
        }
        
        // Add original mesh as background
        const meshTrace = {
            type: 'mesh3d',
            x: this.vertices.map(v => v[0]),
            y: this.vertices.map(v => v[1]),
            z: this.vertices.map(v => v[2]),
            i: this.faces.map(f => f[0]),
            j: this.faces.map(f => f[1]),
            k: this.faces.map(f => f[2]),
            color: 'lightgray',
            opacity: 0.15,
            name: 'Mesh',
            showlegend: true,
            lighting: {
                ambient: 0.8,
                diffuse: 0.8,
                specular: 0.1
            }
        };
        
        const traces = [meshTrace];
        
        // Group ray segments by bounce number for different colors
        const bounceGroups = new Map();
        let totalSegments = 0;
        let successfulSegments = 0;
        let missedSegments = 0;
        
        for (const faceIdx of sampleIndices) {
            const segments = pathSegments[faceIdx];
            if (!segments || segments.length === 0) continue;
            
            for (const segment of segments) {
                totalSegments++;
                
                if (segment.missed) {
                    missedSegments++;
                } else {
                    successfulSegments++;
                }
                
                const bounce = segment.bounce;
                if (!bounceGroups.has(bounce)) {
                    bounceGroups.set(bounce, {
                        successful: { x: [], y: [], z: [] },
                        missed: { x: [], y: [], z: [] }
                    });
                }
                
                const group = bounceGroups.get(bounce);
                const target = segment.missed ? group.missed : group.successful;
                
                // Add line segment
                target.x.push(segment.origin[0], segment.endpoint[0], null);
                target.y.push(segment.origin[1], segment.endpoint[1], null);
                target.z.push(segment.origin[2], segment.endpoint[2], null);
            }
        }
        
        // Create traces for each bounce with different colors
        const bounceColors = [
            '#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00',
            '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff'
        ];
        
        // Add successful ray traces
        for (const [bounce, data] of bounceGroups) {
            if (data.successful.x.length > 0) {
                const color = bounceColors[bounce % bounceColors.length];
                traces.push({
                    type: 'scatter3d',
                    mode: 'lines',
                    x: data.successful.x,
                    y: data.successful.y,
                    z: data.successful.z,
                    line: { 
                        color: color, 
                        width: Math.max(1, 4 - bounce * 0.3) // Thicker for early bounces
                    },
                    name: `Bounce ${bounce} (${data.successful.x.filter(x => x !== null).length / 2} rays)`,
                    showlegend: true,
                    hovertemplate: `Bounce ${bounce}<br>Distance: %{text}<extra></extra>`,
                    opacity: Math.max(0.4, 1.0 - bounce * 0.1)
                });
            }
        }
        
        // Add missed rays trace
        const allMissedX = [];
        const allMissedY = [];
        const allMissedZ = [];
        
        for (const data of bounceGroups.values()) {
            allMissedX.push(...data.missed.x);
            allMissedY.push(...data.missed.y);
            allMissedZ.push(...data.missed.z);
        }
        
        if (allMissedX.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'lines',
                x: allMissedX,
                y: allMissedY,
                z: allMissedZ,
                line: { 
                    color: '#666666', 
                    width: 1,
                    dash: 'dot'
                },
                name: `Missed Rays (${allMissedX.filter(x => x !== null).length / 2} rays)`,
                showlegend: true,
                opacity: 0.3
            });
        }
        
        // Calculate some statistics for the title
        const maxBounce = Math.max(...bounceGroups.keys());
        const avgRayLength = this.calculateAverageRayLength(pathSegments, sampleIndices);
        
        const layout = {
            title: {
                text: `Ray Path Visualization<br><sub>Showing ${sampleIndices.length} faces, ${totalSegments} segments, ${successfulSegments} hits, ${missedSegments} misses<br>Max bounces: ${maxBounce}, Avg ray length: ${avgRayLength.toFixed(3)}</sub>`,
                font: { size: 14 }
            },
            scene: {
                aspectmode: 'data',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                },
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' }
            },
            width: 700,
            height: 600,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: 'rgba(0,0,0,0.2)',
                borderwidth: 1
            }
        };
        
        this.logger.log(`Ray visualization complete: ${totalSegments} total segments, ${successfulSegments} successful hits, ${missedSegments} misses`);
        this.logger.log(`Maximum bounce depth achieved: ${maxBounce}`);
        this.logger.log(`Average ray segment length: ${avgRayLength.toFixed(4)} units`);
        
        return { data: traces, layout };
    }
    
    calculateAverageRayLength(pathSegments, sampleIndices) {
        let totalLength = 0;
        let segmentCount = 0;
        
        for (const faceIdx of sampleIndices) {
            const segments = pathSegments[faceIdx];
            if (!segments) continue;
            
            for (const segment of segments) {
                if (!segment.missed) { // Only count successful hits
                    totalLength += segment.distance;
                    segmentCount++;
                }
            }
        }
        
        return segmentCount > 0 ? totalLength / segmentCount : 0;
    }
}

// Application controller
class App {
    constructor() {
        this.analyzer = null;
        this.setupEventListeners();
        this.logger = new Logger('logContainer');
    }
    
    setupEventListeners() {
        document.getElementById('analyzeBtn').addEventListener('click', () => this.loadAndAnalyze());
        document.getElementById('loadCustomBtn').addEventListener('click', () => this.analyzeCustomMesh());
        document.getElementById('meshFile').addEventListener('change', (e) => this.handleFileSelect(e));
    }
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            document.getElementById('loadCustomBtn').disabled = false;
            this.logger.log(`Selected file: ${file.name}`, 'info');
        }
    }
    
    async loadAndAnalyze() {
        try {
            document.getElementById('analyzeBtn').disabled = true;
            this.clearResults();
            
            this.logger.log('Loading default Bunny.stl mesh...', 'info');
            const meshData = await STLLoader.loadDefault();
            
            this.logger.log(`Mesh loaded: ${meshData.vertices.length} vertices, ${meshData.faces.length} faces`, 'success');
            
            await this.runAnalysis(meshData);
            
        } catch (error) {
            this.logger.log(`Error: ${error.message}`, 'error');
        } finally {
            document.getElementById('analyzeBtn').disabled = false;
        }
    }
    
    async analyzeCustomMesh() {
        try {
            const fileInput = document.getElementById('meshFile');
            const file = fileInput.files[0];
            
            if (!file) {
                this.logger.log('Please select a file first', 'warning');
                return;
            }
            
            document.getElementById('loadCustomBtn').disabled = true;
            this.clearResults();
            
            this.logger.log(`Loading ${file.name}...`, 'info');
            const meshData = await STLLoader.loadFile(file);
            
            this.logger.log(`Mesh loaded: ${meshData.vertices.length} vertices, ${meshData.faces.length} faces`, 'success');
            
            await this.runAnalysis(meshData);
            
        } catch (error) {
            this.logger.log(`Error: ${error.message}`, 'error');
        } finally {
            document.getElementById('loadCustomBtn').disabled = false;
        }
    }
    
    async runAnalysis(meshData) {
        const options = {
            nBounces: parseInt(document.getElementById('nBounces').value),
            nAdditionalRays: parseInt(document.getElementById('nAdditionalRays').value),
            thetaDegrees: parseFloat(document.getElementById('thetaDegrees').value),
            resolution: parseFloat(document.getElementById('resolution').value),
            minPartitionSize: parseInt(document.getElementById('minPartitionSize').value)
        };
        
        this.analyzer = new MeshBounceAnalyzer(meshData, options);
        
        const startTime = Date.now();
        const results = await this.analyzer.analyzeAndSegment(options.minPartitionSize);
        const analysisTime = ((Date.now() - startTime) / 1000).toFixed(2);
        
        this.logger.log(`Analysis completed in ${analysisTime} seconds`, 'success');
        
        this.displayResults(results);
        this.displayStatistics(results.stats);
        this.generateVisualizations(results);
    }
    
    clearResults() {
        document.getElementById('statsPanel').style.display = 'none';
        document.getElementById('visualizationContainer').innerHTML = '';
    }
    
    displayStatistics(stats) {
        const statsPanel = document.getElementById('statsPanel');
        const statsGrid = document.getElementById('statsGrid');
        
        statsGrid.innerHTML = '';
        
        const statItems = [
            { label: 'Total Faces', value: stats.rayStats.totalFaces },
            { label: 'Hit Rate', value: `${stats.rayStats.hitRate}%` },
            { label: 'Face Coverage', value: `${stats.rayStats.faceCoverage}%` },
            { label: 'Avg Hits/Face', value: stats.rayStats.avgHitsPerFace },
            { label: 'Initial Clusters', value: stats.initialPartition.numClusters },
            { label: 'Final Clusters', value: stats.finalPartition.numClusters },
            { label: 'Min Cluster Size', value: stats.finalPartition.minSize },
            { label: 'Max Cluster Size', value: stats.finalPartition.maxSize }
        ];
        
        statItems.forEach(item => {
            const div = document.createElement('div');
            div.className = 'stat-item';
            div.innerHTML = `
                <div class="stat-value">${item.value}</div>
                <div class="stat-label">${item.label}</div>
            `;
            statsGrid.appendChild(div);
        });
        
        statsPanel.style.display = 'block';
    }
    
    generateVisualizations(results) {
        const container = document.getElementById('visualizationContainer');
        
        const visualizations = [
            {
                title: `Initial Segmentation (${results.stats.initialPartition.numClusters} clusters)`,
                plotData: this.analyzer.visualizeSegmentation(results.initialPartition, 'Initial Segmentation')
            },
            {
                title: `Final Segmentation (${results.stats.finalPartition.numClusters} clusters)`,
                plotData: this.analyzer.visualizeSegmentation(results.finalPartition, 'Final Segmentation')
            },
            {
                title: 'Ray Paths Visualization',
                plotData: this.analyzer.visualizeRayPaths(results.paths)
            }
        ];
        
        visualizations.forEach(viz => {
            const panel = document.createElement('div');
            panel.className = 'viz-panel';
            
            const title = document.createElement('div');
            title.className = 'viz-title';
            title.textContent = viz.title;
            panel.appendChild(title);
            
            const plotDiv = document.createElement('div');
            plotDiv.style.width = '100%';
            plotDiv.style.height = '400px';
            panel.appendChild(plotDiv);
            
            container.appendChild(panel);
            
            // Create Plotly visualization
            Plotly.newPlot(plotDiv, viz.plotData.data, viz.plotData.layout, {
                responsive: true,
                displayModeBar: true
            });
        });
    }
    
    displayResults(results) {
        this.logger.log('\n=== ANALYSIS RESULTS ===', 'info');
        this.logger.log(`Total intersections: ${results.totalIntersections}`);
        this.logger.log(`Initial partitions: ${results.stats.initialPartition.numClusters}`);
        this.logger.log(`Final partitions: ${results.stats.finalPartition.numClusters}`);
        
        results.hitsPerBounce.forEach((hits, bounce) => {
            const percentage = ((hits / this.analyzer.faceCount) * 100).toFixed(1);
            this.logger.log(`Bounce ${bounce}: ${hits} hits (${percentage}% success rate)`);
        });
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new App();
});