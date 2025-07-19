// === MESH RAY-TRACING ANALYSIS ALGORITHM ===
// Advanced mesh segmentation using ray-tracing and graph clustering

// === UTILITY CLASSES ===

class Vector3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    static fromArray(arr) {
        return new Vector3(arr[0], arr[1], arr[2]);
    }
    
    toArray() {
        return [this.x, this.y, this.z];
    }
    
    add(v) {
        return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
    }
    
    subtract(v) {
        return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
    }
    
    multiply(scalar) {
        return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
    }
    
    dot(v) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }
    
    cross(v) {
        return new Vector3(
            this.y * v.z - this.z * v.y,
            this.z * v.x - this.x * v.z,
            this.x * v.y - this.y * v.x
        );
    }
    
    magnitude() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }
    
    normalize() {
        const mag = this.magnitude();
        if (mag > 0) {
            return new Vector3(this.x / mag, this.y / mag, this.z / mag);
        }
        return new Vector3(0, 0, 0);
    }
    
    clone() {
        return new Vector3(this.x, this.y, this.z);
    }
}

class UnionFind {
    constructor(size) {
        this.parent = Array.from({ length: size }, (_, i) => i);
        this.rank = new Array(size).fill(0);
    }
    
    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]); // Path compression
        }
        return this.parent[x];
    }
    
    union(x, y) {
        const px = this.find(x);
        const py = this.find(y);
        
        if (px === py) return;
        
        if (this.rank[px] < this.rank[py]) {
            this.parent[px] = py;
        } else if (this.rank[px] > this.rank[py]) {
            this.parent[py] = px;
        } else {
            this.parent[py] = px;
            this.rank[px]++;
        }
    }
    
    getSets() {
        const sets = {};
        for (let i = 0; i < this.parent.length; i++) {
            const root = this.find(i);
            if (!sets[root]) {
                sets[root] = [];
            }
            sets[root].push(i);
        }
        return Object.values(sets);
    }
}

// === MAIN MESH ANALYSIS ENGINE ===

class MeshBounceAnalyzer {
    constructor(meshData, options = {}) {
        this.meshData = meshData;
        this.options = {
            nBounces: options.nBounces || 30,
            nAdditionalRays: options.nAdditionalRays || 1,
            thetaDegrees: options.thetaDegrees || 30,
            batchSize: options.batchSize || 100,
            resolution: options.resolution || 1.0,
            minClusterSize: options.minClusterSize || 20,
            ...options
        };
        
        this.vertices = meshData.vertices.map(v => Vector3.fromArray(v));
        this.faces = meshData.faces;
        this.faceCount = this.faces.length;
        this.hitCounts = new Array(this.faceCount).fill(0);
        this.isAnalyzing = false;
        this.shouldStop = false;
        
        this.precomputeMeshData();
    }
    
    precomputeMeshData() {
        this.faceNormals = [];
        this.faceCentroids = [];
        this.faceAreas = [];
        this.faceAdjacency = {};
        
        // Precompute face data
        for (let i = 0; i < this.faceCount; i++) {
            const face = this.faces[i];
            const v0 = this.vertices[face[0]];
            const v1 = this.vertices[face[1]];
            const v2 = this.vertices[face[2]];
            
            // Face normal
            const edge1 = v1.subtract(v0);
            const edge2 = v2.subtract(v0);
            const normal = edge1.cross(edge2).normalize();
            this.faceNormals.push(normal);
            
            // Face centroid
            const centroid = v0.add(v1).add(v2).multiply(1/3);
            this.faceCentroids.push(centroid);
            
            // Face area
            const area = edge1.cross(edge2).magnitude() * 0.5;
            this.faceAreas.push(area);
            
            // Initialize adjacency
            this.faceAdjacency[i] = [];
        }
        
        // Build face adjacency
        this.buildFaceAdjacency();
        
        // Precompute triangle data for intersection testing
        this.v0Array = [];
        this.edge1Array = [];
        this.edge2Array = [];
        
        for (let i = 0; i < this.faceCount; i++) {
            const face = this.faces[i];
            const v0 = this.vertices[face[0]];
            const v1 = this.vertices[face[1]];
            const v2 = this.vertices[face[2]];
            
            this.v0Array.push(v0);
            this.edge1Array.push(v1.subtract(v0));
            this.edge2Array.push(v2.subtract(v0));
        }
    }
    
    buildFaceAdjacency() {
        const edgeToFaces = {};
        
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            const face = this.faces[faceIdx];
            const edges = [
                [face[0], face[1]].sort((a, b) => a - b),
                [face[1], face[2]].sort((a, b) => a - b),
                [face[2], face[0]].sort((a, b) => a - b)
            ];
            
            for (const edge of edges) {
                const edgeKey = `${edge[0]},${edge[1]}`;
                if (!edgeToFaces[edgeKey]) {
                    edgeToFaces[edgeKey] = [];
                }
                edgeToFaces[edgeKey].push(faceIdx);
            }
        }
        
        // Build adjacency from shared edges
        for (const faces of Object.values(edgeToFaces)) {
            if (faces.length === 2) {
                const [face1, face2] = faces;
                this.faceAdjacency[face1].push(face2);
                this.faceAdjacency[face2].push(face1);
            }
        }
    }
    
    // Möller–Trumbore ray-triangle intersection - FIXED VERSION
    rayTriangleIntersect(rayOrigin, rayDirection, faceIndex) {
        const epsilon = 1e-7; // Match Python epsilon
        const v0 = this.v0Array[faceIndex];
        const edge1 = this.edge1Array[faceIndex];
        const edge2 = this.edge2Array[faceIndex];
        
        const h = rayDirection.cross(edge2);
        const a = edge1.dot(h);
        
        if (Math.abs(a) < epsilon) {
            return null; // Ray is parallel to triangle
        }
        
        const f = 1.0 / a;
        const s = rayOrigin.subtract(v0);
        const u = f * s.dot(h);
        
        if (u < -epsilon || u > 1.0 + epsilon) {
            return null;
        }
        
        const q = s.cross(edge1);
        const v = f * rayDirection.dot(q);
        
        if (v < -epsilon || u + v > 1.0 + epsilon) {
            return null;
        }
        
        const t = f * edge2.dot(q);
        
        if (t > epsilon) {
            // CRITICAL FIX: Check if ray enters from back face (interior) 
            // This matches the Python implementation exactly
            const normal = this.faceNormals[faceIndex];
            const enteringFromBack = rayDirection.dot(normal) > epsilon;
            
            if (enteringFromBack) {
                const hitPoint = rayOrigin.add(rayDirection.multiply(t));
                return { t, hitPoint, faceIndex };
            }
        }
        
        return null;
    }
    
    findClosestIntersection(rayOrigin, rayDirection, excludeFace = -1) {
        let closestHit = null;
        let closestT = Infinity;
        
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            if (faceIdx === excludeFace) continue;
            
            const hit = this.rayTriangleIntersect(rayOrigin, rayDirection, faceIdx);
            if (hit && hit.t < closestT) {
                closestHit = hit;
                closestT = hit.t;
            }
        }
        
        return closestHit;
    }
    
    // FIXED: Proper cone ray generation matching Python implementation
    generateConeRays(baseDirection, nRays, thetaRadians) {
        const rays = [baseDirection.clone()]; // Start with base direction
        
        if (nRays === 0) return rays;
        
        const zAxis = new Vector3(0, 0, 1);
        
        for (let i = 0; i < nRays; i++) {
            // CRITICAL: Use Python's exact random distribution
            const phi = thetaRadians * Math.sqrt(Math.random());
            const omega = 2 * Math.PI * Math.random();
            
            // Convert to cartesian coordinates (relative to z-axis)
            const sinPhi = Math.sin(phi);
            const perturbation = new Vector3(
                sinPhi * Math.cos(omega),
                sinPhi * Math.sin(omega),
                Math.cos(phi)
            );
            
            // CRITICAL FIX: Use proper rotation matrix calculation
            const rotatedDirection = this.applyRodriguesRotation(perturbation, zAxis, baseDirection);
            rays.push(rotatedDirection.normalize());
        }
        
        return rays;
    }
    
    // FIXED: Proper Rodrigues Rotation (matching Python exactly)
    applyRodriguesRotation(vector, fromAxis, toAxis) {
        const dot = fromAxis.dot(toAxis);
        
        // Handle parallel vectors
        if (Math.abs(dot) > 0.9999) {
            if (dot > 0) {
                return vector.clone(); // Same direction
            } else {
                return vector.multiply(-1); // Opposite direction
            }
        }
        
        // Calculate rotation axis and angle
        const rotationAxis = fromAxis.cross(toAxis).normalize();
        const rotationAngle = Math.acos(Math.max(-1, Math.min(1, dot)));
        
        const cosAngle = Math.cos(rotationAngle);
        const sinAngle = Math.sin(rotationAngle);
        
        // Rodrigues' rotation formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        const axisDotVector = rotationAxis.dot(vector);
        const axisCrossVector = rotationAxis.cross(vector);
        
        return vector.multiply(cosAngle)
            .add(axisCrossVector.multiply(sinAngle))
            .add(rotationAxis.multiply(axisDotVector * (1 - cosAngle)));
    }
    
    reflectRay(rayDirection, normal) {
        const dot = rayDirection.dot(normal);
        return rayDirection.subtract(normal.multiply(2 * dot)).normalize();
    }
    
    // CRITICAL FIX: Complete rewrite to match Python's approach exactly
    async analyzeWithProgress(progressCallback) {
        this.isAnalyzing = true;
        this.shouldStop = false;
        
        const startTime = Date.now();
        const thetaRadians = this.options.thetaDegrees * Math.PI / 180;
        
        // Reset hit counts
        this.hitCounts.fill(0);
        
        const rayChains = [];
        const transitionGraph = {};
        let totalIntersections = 0;
        
        progressCallback?.({ stage: 'initialization', progress: 0 });
        
        // CRITICAL FIX: Process all faces with multiple rays per face, find first successful ray
        const allRayOrigins = [];
        const allRayDirections = [];
        const rayToFaceMap = [];
        const rayToTypeMap = []; // 0 = base ray, 1+ = additional rays
        
        // Generate ALL rays first (matching Python's approach)
        for (let faceIdx = 0; faceIdx < this.faceCount; faceIdx++) {
            const origin = this.faceCentroids[faceIdx];
            const baseDirection = this.faceNormals[faceIdx].multiply(-1); // Inward ray
            
            // Generate rays in cone (including base ray)
            const rays = this.generateConeRays(
                baseDirection, 
                this.options.nAdditionalRays, 
                thetaRadians
            );
            
            // Add all rays for this face
            for (let rayIdx = 0; rayIdx < rays.length; rayIdx++) {
                allRayOrigins.push(origin);
                allRayDirections.push(rays[rayIdx]);
                rayToFaceMap.push(faceIdx);
                rayToTypeMap.push(rayIdx);
            }
        }
        
        console.log(`Generated ${allRayOrigins.length} total rays for ${this.faceCount} faces`);
        
        // CRITICAL FIX: Find first successful intersection for each face
        const faceFirstHits = new Array(this.faceCount).fill(null);
        const faceFirstRayDirections = new Array(this.faceCount).fill(null);
        
        // Test all rays and find first success per face
        for (let rayIdx = 0; rayIdx < allRayOrigins.length && !this.shouldStop; rayIdx++) {
            const faceIdx = rayToFaceMap[rayIdx];
            
            // Skip if this face already has a successful hit
            if (faceFirstHits[faceIdx] !== null) continue;
            
            const origin = allRayOrigins[rayIdx];
            const direction = allRayDirections[rayIdx];
            
            const hit = this.findClosestIntersection(origin, direction, faceIdx);
            
            if (hit) {
                faceFirstHits[faceIdx] = hit;
                faceFirstRayDirections[faceIdx] = direction.clone();
            }
            
            // Progress update
            if (rayIdx % 100 === 0) {
                const progress = (rayIdx / allRayOrigins.length) * 40; // 40% for initial ray testing
                progressCallback?.({ 
                    stage: 'ray_testing', 
                    progress, 
                    tested: rayIdx,
                    totalRays: allRayOrigins.length
                });
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        if (this.shouldStop) return null;
        
        // Count successful faces
        const successfulFaces = faceFirstHits.filter(hit => hit !== null).length;
        console.log(`${successfulFaces}/${this.faceCount} faces achieved successful raycast`);
        
        progressCallback?.({ stage: 'ray_tracing', progress: 50 });
        
        // CRITICAL FIX: Build proper ray chains (one per face, using first successful ray)
        let processedBounces = 0;
        let transitionCount = 0; // DEBUG: Count transitions
        
        for (let faceIdx = 0; faceIdx < this.faceCount && !this.shouldStop; faceIdx++) {
            const firstHit = faceFirstHits[faceIdx];
            const firstDirection = faceFirstRayDirections[faceIdx];
            
            if (!firstHit || !firstDirection) {
                // No successful raycast for this face
                rayChains.push([faceIdx]); // Chain with just the starting face
                continue;
            }
            
            // Build chain starting with successful hit
            const chain = [faceIdx, firstHit.faceIndex];
            
            // Record first transition
            if (!transitionGraph[faceIdx]) transitionGraph[faceIdx] = {};
            if (!transitionGraph[faceIdx][firstHit.faceIndex]) {
                transitionGraph[faceIdx][firstHit.faceIndex] = 0;
            }
            transitionGraph[faceIdx][firstHit.faceIndex]++;
            transitionCount++; // DEBUG
            
            // Update hit count
            this.hitCounts[firstHit.faceIndex]++;
            totalIntersections++;
            
            // Continue bouncing
            let currentOrigin = firstHit.hitPoint.add(this.faceNormals[firstHit.faceIndex].multiply(1e-6));
            let currentDirection = this.reflectRay(firstDirection, this.faceNormals[firstHit.faceIndex]);
            let previousFace = firstHit.faceIndex;
            
            for (let bounce = 1; bounce < this.options.nBounces; bounce++) {
                const hit = this.findClosestIntersection(currentOrigin, currentDirection, previousFace);
                
                if (hit) {
                    chain.push(hit.faceIndex);
                    this.hitCounts[hit.faceIndex]++;
                    totalIntersections++;
                    
                    // Record transition
                    if (!transitionGraph[previousFace]) transitionGraph[previousFace] = {};
                    if (!transitionGraph[previousFace][hit.faceIndex]) {
                        transitionGraph[previousFace][hit.faceIndex] = 0;
                    }
                    transitionGraph[previousFace][hit.faceIndex]++;
                    transitionCount++; // DEBUG
                    
                    // Prepare for next bounce
                    if (bounce < this.options.nBounces - 1) {
                        const hitNormal = this.faceNormals[hit.faceIndex];
                        currentOrigin = hit.hitPoint.add(hitNormal.multiply(1e-6));
                        currentDirection = this.reflectRay(currentDirection, hitNormal);
                        previousFace = hit.faceIndex;
                    }
                } else {
                    break; // No more intersections
                }
            }
            
            rayChains.push(chain);
            processedBounces++;
            
            // Progress update
            if (processedBounces % 10 === 0) {
                const progress = 50 + (processedBounces / this.faceCount) * 30; // 30% for bounce processing
                progressCallback?.({ 
                    stage: 'ray_bouncing', 
                    progress, 
                    face: processedBounces,
                    totalFaces: this.faceCount,
                    intersections: totalIntersections
                });
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        // DEBUG: Log transition statistics
        console.log(`Built ${transitionCount} transitions in transition graph`);
        const avgChainLength = rayChains.reduce((sum, chain) => sum + chain.length, 0) / rayChains.length;
        console.log(`Average ray chain length: ${avgChainLength.toFixed(2)}`);
        
        if (this.shouldStop) return null;
        
        progressCallback?.({ stage: 'clustering', progress: 85 });
        
        // DEBUG: Quick check of transition graph before clustering
        const transitionKeys = Object.keys(transitionGraph);
        if (transitionKeys.length === 0) {
            console.log('ERROR: Empty transition graph! All faces will remain in separate clusters.');
            // Return each face as its own cluster
            const emptyClusters = {};
            for (let i = 0; i < this.faceCount; i++) {
                emptyClusters[i] = i;
            }
            return {
                hitCounts: this.hitCounts,
                transitionGraph,
                rayChains,
                clusters: emptyClusters,
                totalIntersections,
                processingTime: (Date.now() - startTime) / 1000,
                faceCount: this.faceCount,
                paths: this.generatePathsForVisualization(rayChains)
            };
        }
        
        // Perform clustering
        const initialClusters = await this.performClustering(transitionGraph);
        
        // DEBUG: Add explicit logging to progress callback
        const initialClusterCount = new Set(Object.values(initialClusters)).size;
        progressCallback?.({ 
            stage: 'clustering_complete', 
            progress: 87,
            clusters: initialClusterCount,
            transitions: transitionCount
        });
        
        progressCallback?.({ stage: 'refinement', progress: 90 });
        
        // Refine clusters
        const refinedClusters = this.refinePartitionWithConnectivity(initialClusters);
        
        // DEBUG: Check refinement impact
        const refinedCount = new Set(Object.values(refinedClusters)).size;
        console.log(`DEBUG: Refinement changed clusters from ${initialClusterCount} to ${refinedCount}`);
        
        progressCallback?.({ stage: 'merging', progress: 95 });
        
        const finalClusters = this.mergeSmallPartitions(refinedClusters);
        
        // DEBUG: Check merging impact
        const finalCount = new Set(Object.values(finalClusters)).size;
        console.log(`DEBUG: Merging changed clusters from ${refinedCount} to ${finalCount}`);
        
        const processingTime = (Date.now() - startTime) / 1000;
        
        this.isAnalyzing = false;
        
        return {
            hitCounts: this.hitCounts,
            transitionGraph,
            rayChains,
            clusters: finalClusters,
            initialClusters: initialClusters,    // NEW: Store initial clustering result
            refinedClusters: refinedClusters,    // NEW: Store refined clustering result
            totalIntersections,
            processingTime,
            faceCount: this.faceCount,
            paths: this.generatePathsForVisualization(rayChains)
        };
    }
    
    // NEW: Generate paths for visualization like Python version
    generatePathsForVisualization(rayChains) {
        const paths = [];
        
        // Group chains by bounce to create path segments
        const maxBounces = Math.max(...rayChains.map(chain => chain.length - 1));
        
        for (let bounce = 0; bounce < maxBounces; bounce++) {
            const origins = [];
            const endpoints = [];
            const hitFaces = [];
            
            for (const chain of rayChains) {
                if (bounce < chain.length - 1) {
                    const fromFace = chain[bounce];
                    const toFace = chain[bounce + 1];
                    
                    origins.push(this.faceCentroids[fromFace]);
                    endpoints.push(this.faceCentroids[toFace]);
                    hitFaces.push(toFace);
                } else {
                    // Pad with invalid data for consistent structure
                    origins.push(new Vector3(0, 0, 0));
                    endpoints.push(new Vector3(0, 0, 0));
                    hitFaces.push(-1);
                }
            }
            
            paths.push({ origins, endpoints, hitFaces });
        }
        
        return paths;
    }
    
    // LIBRARY: Use jLouvain library (same algorithm as Python)
    async performClustering(transitionGraph) {
        console.log('Using jLouvain library for clustering...');
        
        // Check if jLouvain is available
        if (typeof jLouvain === 'undefined') {
            console.log('jLouvain library not available, falling back to simple clustering');
            return this.improvedLouvainClustering(transitionGraph);
        }
        
        // Convert transition graph to jLouvain format
        const nodes = [];
        const edges = [];
        
        // Add all nodes
        for (let i = 0; i < this.faceCount; i++) {
            nodes.push(i.toString());
        }
        
        // Add edges with weights
        for (const [fromStr, toMap] of Object.entries(transitionGraph)) {
            const from = fromStr;
            for (const [toStr, weight] of Object.entries(toMap)) {
                const to = toStr;
                
                if (from !== to && weight > 0) {
                    // jLouvain expects undirected edges, so add both directions with combined weight
                    const reverseWeight = (transitionGraph[to] && transitionGraph[to][from]) || 0;
                    const totalWeight = weight + reverseWeight;
                    
                    // Add edge (jLouvain will handle duplicates)
                    edges.push({
                        source: from,
                        target: to,
                        weight: totalWeight
                    });
                }
            }
        }
        
        console.log(`Prepared ${nodes.length} nodes and ${edges.length} edges for jLouvain`);
        
        // Configure jLouvain with resolution parameter
        const community = jLouvain()
            .nodes(nodes)
            .edges(edges)
            .resolution(this.options.resolution); // Use same resolution as Python
        
        // Run clustering
        const result = community();
        
        // Convert jLouvain result back to our format
        const clusters = {};
        for (const [nodeStr, clusterId] of Object.entries(result)) {
            clusters[parseInt(nodeStr)] = clusterId;
        }
        
        const clusterCount = new Set(Object.values(clusters)).size;
        console.log(`jLouvain clustering completed: ${clusterCount} clusters`);
        
        return clusters;
    }
    
    // SIMPLE: Basic clustering focused on performance, not perfect Louvain
    async improvedLouvainClustering(transitionGraph) {
        console.log('Starting simple weight-based clustering...');
        
        // Initialize each face as its own cluster
        const clusters = {};
        for (let i = 0; i < this.faceCount; i++) {
            clusters[i] = i;
        }
        
        // Build adjacency with weights
        const adjacency = {};
        const edgeWeights = {};
        
        for (let i = 0; i < this.faceCount; i++) {
            adjacency[i] = [];
        }
        
        // Build graph from transitions
        for (const [fromStr, toMap] of Object.entries(transitionGraph)) {
            const from = parseInt(fromStr);
            for (const [toStr, weight] of Object.entries(toMap)) {
                const to = parseInt(toStr);
                
                if (from !== to && weight > 0) {
                    if (!adjacency[from].includes(to)) {
                        adjacency[from].push(to);
                        adjacency[to].push(from);
                        edgeWeights[`${Math.min(from, to)}-${Math.max(from, to)}`] = weight;
                    }
                }
            }
        }
        
        // SIMPLE: Greedy clustering - merge highest weight edges first
        const edges = [];
        for (const [edgeKey, weight] of Object.entries(edgeWeights)) {
            const [from, to] = edgeKey.split('-').map(Number);
            edges.push({ from, to, weight });
        }
        
        // Sort edges by weight (highest first)
        edges.sort((a, b) => b.weight - a.weight);
        
        console.log(`Processing ${edges.length} edges for clustering...`);
        
        // Process top edges to merge clusters
        const maxEdgesToProcess = Math.min(edges.length, this.faceCount); // Limit processing
        
        for (let i = 0; i < maxEdgesToProcess; i++) {
            const { from, to, weight } = edges[i];
            
            const clusterFrom = clusters[from];
            const clusterTo = clusters[to];
            
            if (clusterFrom !== clusterTo) {
                // Simple merge: smaller cluster joins larger cluster
                const sizeFrom = Object.values(clusters).filter(c => c === clusterFrom).length;
                const sizeTo = Object.values(clusters).filter(c => c === clusterTo).length;
                
                const sourceCluster = sizeFrom < sizeTo ? clusterFrom : clusterTo;
                const targetCluster = sizeFrom < sizeTo ? clusterTo : clusterFrom;
                
                // Apply resolution factor - higher resolution = less merging
                const mergeThreshold = weight / (this.options.resolution * Math.max(sizeFrom, sizeTo));
                
                if (mergeThreshold > 1.0) { // Only merge if weight is high enough
                    // Merge source cluster into target cluster
                    for (let j = 0; j < this.faceCount; j++) {
                        if (clusters[j] === sourceCluster) {
                            clusters[j] = targetCluster;
                        }
                    }
                }
            }
            
            // Yield every 100 edges to prevent hanging
            if (i % 100 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        const finalClusters = new Set(Object.values(clusters));
        console.log(`Simple clustering completed: ${finalClusters.size} clusters`);
        return clusters;
    }
    
    // FAST: Simplified refinement with connectivity
    refinePartitionWithConnectivity(partition) {
        console.log('Starting fast refinement...');
        
        // Group faces by partition
        const partitionGroups = {};
        for (const [faceIdx, clusterId] of Object.entries(partition)) {
            if (!partitionGroups[clusterId]) {
                partitionGroups[clusterId] = [];
            }
            partitionGroups[clusterId].push(parseInt(faceIdx));
        }
        
        let newClusterId = Math.max(...Object.keys(partitionGroups).map(Number)) + 1;
        const newPartition = { ...partition };
        
        // Process only larger partitions to save time
        let processedPartitions = 0;
        for (const [clusterId, faces] of Object.entries(partitionGroups)) {
            if (faces.length <= 1 || faces.length > 1000) continue; // Skip very small or very large
            
            // Quick connectivity check using UnionFind
            const uf = new UnionFind(faces.length);
            const faceToLocal = {};
            
            faces.forEach((face, index) => {
                faceToLocal[face] = index;
            });
            
            // Union adjacent faces within this partition
            for (let i = 0; i < Math.min(faces.length, 500); i++) { // Limit processing
                const face = faces[i];
                const adjacentFaces = this.faceAdjacency[face] || [];
                
                for (const adjFace of adjacentFaces) {
                    if (faceToLocal.hasOwnProperty(adjFace)) {
                        uf.union(i, faceToLocal[adjFace]);
                    }
                }
            }
            
            // Get connected components
            const connectedComponents = uf.getSets();
            
            // Assign new cluster IDs only if we have multiple components
            if (connectedComponents.length > 1) {
                for (let componentIdx = 1; componentIdx < connectedComponents.length; componentIdx++) {
                    const component = connectedComponents[componentIdx];
                    for (const localIdx of component) {
                        const globalFaceIdx = faces[localIdx];
                        newPartition[globalFaceIdx] = newClusterId;
                    }
                    newClusterId++;
                }
            }
            
            processedPartitions++;
            if (processedPartitions % 10 === 0) {
                console.log(`Refined ${processedPartitions} partitions...`);
            }
        }
        
        console.log('Fast refinement completed');
        return newPartition;
    }
    
    // DEBUG: Skip merging to test if that's the issue
    mergeSmallPartitions(partition) {
        console.log('DEBUGGING: Skipping merging to test clustering...');
        
        // Just renumber clusters to be consecutive
        const uniqueClusters = [...new Set(Object.values(partition))].sort((a, b) => a - b);
        const clusterMap = {};
        uniqueClusters.forEach((clusterId, index) => {
            clusterMap[clusterId] = index;
        });
        
        const finalPartition = {};
        for (const [faceIdx, clusterId] of Object.entries(partition)) {
            finalPartition[faceIdx] = clusterMap[clusterId];
        }
        
        console.log(`DEBUG: Skipped merging, kept ${uniqueClusters.length} clusters`);
        return finalPartition;
    }
    
    stop() {
        this.shouldStop = true;
    }
    
    // FIXED: Generate proper ray paths for visualization
    visualizeRayPaths(paths, maxPaths = 1000) {
        const lines = [];
        let pathCount = 0;
        
        // Extract ray segments from ray chains
        for (const chain of this.rayChains) {
            if (pathCount >= maxPaths) break;
            
            // Each chain represents bounces between face centroids
            for (let i = 0; i < chain.length - 1; i++) {
                if (pathCount >= maxPaths) break;
                
                const fromFace = chain[i];
                const toFace = chain[i + 1];
                
                if (fromFace >= 0 && toFace >= 0) {
                    lines.push({
                        start: this.faceCentroids[fromFace].toArray(),
                        end: this.faceCentroids[toFace].toArray()
                    });
                    pathCount++;
                }
            }
        }
        
        return lines;
    }
    
    // NEW: Detailed statistics display like Python
    displayDetailedStats(results) {
        const { hitCounts, initialClusters, refinedClusters, clusters, rayChains } = results;
        
        // Ray-tracing statistics
        const totalHits = hitCounts.reduce((sum, count) => sum + count, 0);
        const minHits = Math.min(...hitCounts);
        const maxHits = Math.max(...hitCounts);
        const avgHits = totalHits / this.faceCount;
        const hitVariance = hitCounts.reduce((sum, count) => sum + Math.pow(count - avgHits, 2), 0) / this.faceCount;
        const hitStdDev = Math.sqrt(hitVariance);
        const faceCoverage = (hitCounts.filter(count => count > 0).length / this.faceCount) * 100;
        
        // Segmentation statistics
        const getClusterStats = (clusters) => {
            const sizes = {};
            Object.values(clusters).forEach(clusterId => {
                sizes[clusterId] = (sizes[clusterId] || 0) + 1;
            });
            const sizeValues = Object.values(sizes);
            return {
                count: Object.keys(sizes).length,
                avg: sizeValues.reduce((sum, size) => sum + size, 0) / sizeValues.length,
                min: Math.min(...sizeValues),
                max: Math.max(...sizeValues),
                stdDev: Math.sqrt(sizeValues.reduce((sum, size) => sum + Math.pow(size - (sizeValues.reduce((s, sz) => s + sz, 0) / sizeValues.length), 2), 0) / sizeValues.length)
            };
        };
        
        const initialStats = getClusterStats(initialClusters);
        const refinedStats = getClusterStats(refinedClusters);
        const finalStats = getClusterStats(clusters);
        
        // Per-bounce statistics
        const perBounceStats = [];
        const maxChainLength = Math.max(...rayChains.map(chain => chain.length));
        for (let bounce = 0; bounce < maxChainLength - 1; bounce++) {
            let hitsAtBounce = 0;
            for (const chain of rayChains) {
                if (chain.length > bounce + 1) {
                    hitsAtBounce++;
                }
            }
            perBounceStats.push(hitsAtBounce);
        }
        
        // Display comprehensive stats
        window.log('\n=== DETAILED ANALYSIS STATISTICS ===');
        window.log('\nRay-tracing Statistics:');
        window.log(`Total faces analyzed: ${this.faceCount}`);
        window.log(`Total ray segments: ${this.faceCount * this.options.nBounces}`);
        window.log(`Valid intersections: ${totalHits}`);
        window.log(`Hit rate: ${(totalHits / (this.faceCount * this.options.nBounces) * 100).toFixed(1)}%`);
        window.log(`Average hits per face: ${avgHits.toFixed(2)}`);
        window.log(`Min hits on a face: ${minHits}`);
        window.log(`Max hits on a face: ${maxHits}`);
        window.log(`Std dev of hits: ${hitStdDev.toFixed(2)}`);
        window.log(`Face coverage: ${faceCoverage.toFixed(1)}%`);
        
        window.log('\nSegmentation Statistics:');
        window.log(`Initial segmentation:`);
        window.log(`  Number of clusters: ${initialStats.count}`);
        window.log(`  Average cluster size: ${initialStats.avg.toFixed(1)}`);
        window.log(`  Minimum cluster size: ${initialStats.min}`);
        window.log(`  Maximum cluster size: ${initialStats.max}`);
        window.log(`  Cluster size std dev: ${initialStats.stdDev.toFixed(1)}`);
        
        window.log(`Refined segmentation:`);
        window.log(`  Number of clusters: ${refinedStats.count}`);
        window.log(`  Average cluster size: ${refinedStats.avg.toFixed(1)}`);
        window.log(`  Minimum cluster size: ${refinedStats.min}`);
        window.log(`  Maximum cluster size: ${refinedStats.max}`);
        window.log(`  Cluster size std dev: ${refinedStats.stdDev.toFixed(1)}`);
        
        window.log(`Final segmentation:`);
        window.log(`  Number of clusters: ${finalStats.count}`);
        window.log(`  Average cluster size: ${finalStats.avg.toFixed(1)}`);
        window.log(`  Minimum cluster size: ${finalStats.min}`);
        window.log(`  Maximum cluster size: ${finalStats.max}`);
        window.log(`  Cluster size std dev: ${finalStats.stdDev.toFixed(1)}`);
        
        window.log('\nPer-bounce Statistics:');
        perBounceStats.forEach((hits, bounce) => {
            const percentage = (hits / this.faceCount * 100).toFixed(1);
            window.log(`  Bounce ${bounce}: ${hits} hits (${percentage}% of rays)`);
        });
    }
}

// === INTEGRATION WITH EXISTING UI ===

// Function to connect the algorithm to the existing UI
function connectMeshAlgorithm() {
    // This function will be called from the main HTML file
    // It integrates the algorithm with the existing UI components
    
    // Override the existing analyze button behavior
    document.getElementById('analyzeBtn').onclick = async function() {
        // Get the current mesh data from the global scope
        if (!window.currentMeshData) {
            window.showError('Please load a mesh file first');
            return;
        }
        
        // Get analysis parameters from UI
        const options = {
            nBounces: parseInt(document.getElementById('bounces').value),
            nAdditionalRays: parseInt(document.getElementById('additionalRays').value),
            thetaDegrees: parseFloat(document.getElementById('coneAngle').value),
            batchSize: parseInt(document.getElementById('batchSize').value),
            resolution: parseFloat(document.getElementById('resolution').value),
            minClusterSize: parseInt(document.getElementById('minClusterSize').value)
        };
        
        // Validate mesh data has reasonable size
        if (window.currentMeshData.faces.length > 10000) {
            const proceed = confirm(
                `This mesh has ${window.currentMeshData.faces.length} faces. ` +
                `Analysis may take several minutes. Continue?`
            );
            if (!proceed) return;
        }
        
        // Create analyzer
        const analyzer = new MeshBounceAnalyzer(window.currentMeshData, options);
        
        // Store analyzer reference globally for stop functionality
        window.currentAnalyzer = analyzer;
        
        // Update UI for analysis start
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        
        window.log('Starting mesh analysis...');
        window.log(`Parameters: ${options.nBounces} bounces, ${options.nAdditionalRays} additional rays, ${options.thetaDegrees}° cone`);
        
        try {
            // Run analysis with progress updates
            const results = await analyzer.analyzeWithProgress((progress) => {
                updateAnalysisProgress(progress);
            });
            
            if (results) {
                displayAnalysisResults(results);
                
                // Update 3D visualization with clusters
                if (window.meshVisualization) {
                    window.meshVisualization.displayMesh(window.currentMeshData, results.clusters);
                }
                
                // Store results globally for potential export/further analysis
                window.analysisResults = results;
                
                window.log(`Analysis completed successfully in ${results.processingTime.toFixed(2)}s`);
                window.log(`Found ${new Set(Object.values(results.clusters)).size} clusters`);
                
                // FORCE show visualization controls - simple test
                const visualDiv = document.getElementById('visualizationControls');
                if (visualDiv) {
                    visualDiv.style.display = 'block';
                    window.log('✅ Visualization controls should now be visible!');
                } else {
                    window.log('❌ ERROR: visualizationControls div not found in DOM');
                }
                
                // Add visualization controls with all stages - TRY CATCH for debugging
                try {
                    window.log('🔄 About to call addVisualizationControls...');
                    addVisualizationControls(results);
                    window.log('✅ addVisualizationControls completed');
                } catch (error) {
                    window.log(`❌ Error in addVisualizationControls: ${error.message}`);
                    console.error('Visualization controls error:', error);
                }
                
            } else {
                window.log('Analysis was stopped by user');
            }
            
        } catch (error) {
            window.showError(`Analysis failed: ${error.message}`);
            window.log(`Analysis error: ${error.message}`);
            console.error('Full error:', error);
        } finally {
            // Reset UI
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            window.currentAnalyzer = null;
        }
    };
    
    // Add stop button functionality
    document.getElementById('stopBtn').onclick = function() {
        if (window.currentAnalyzer) {
            window.currentAnalyzer.stop();
            window.log('Stopping analysis...');
        }
    };
}

// Helper functions for UI integration
function updateAnalysisProgress(progress) {
    const { stage, progress: percent, face, totalFaces, intersections, tested, totalRays, clusters, transitions } = progress;
    
    document.getElementById('progressFill').style.width = `${percent}%`;
    
    let statusText = '';
    switch (stage) {
        case 'initialization':
            statusText = 'Initializing analysis...';
            break;
        case 'ray_testing':
            statusText = `Testing rays: ${tested}/${totalRays}`;
            break;
        case 'ray_tracing':
        case 'ray_bouncing':
            statusText = `Ray tracing: ${face}/${totalFaces} faces (${intersections || 0} hits)`;
            break;
        case 'clustering':
            statusText = 'Performing clustering...';
            break;
        case 'clustering_complete':
            statusText = `Clustering complete: ${clusters} clusters from ${transitions} transitions`;
            window.log(`DEBUG: Clustering produced ${clusters} clusters from ${transitions} transitions`);
            break;
        case 'refinement':
            statusText = 'Refining clusters...';
            break;
        case 'merging':
            statusText = 'Merging small clusters...';
            break;
        default:
            statusText = `Processing: ${percent.toFixed(1)}%`;
    }
    
    document.getElementById('progressText').textContent = statusText;
}

function displayAnalysisResults(results) {
    const { hitCounts, clusters, totalIntersections, processingTime, faceCount } = results;
    
    // Calculate statistics
    const totalHits = hitCounts.reduce((sum, count) => sum + count, 0);
    const numClusters = new Set(Object.values(clusters)).size;
    const avgRaysPerFace = totalHits / faceCount;
    const hitRate = (totalHits / (faceCount * (results.paths ? results.paths.length : 1))) * 100;
    
    // Calculate segmentation quality metrics
    const clusterSizes = {};
    Object.values(clusters).forEach(clusterId => {
        clusterSizes[clusterId] = (clusterSizes[clusterId] || 0) + 1;
    });
    
    const clusterSizeValues = Object.values(clusterSizes);
    const avgClusterSize = clusterSizeValues.reduce((sum, size) => sum + size, 0) / clusterSizeValues.length;
    const minClusterSize = Math.min(...clusterSizeValues);
    const maxClusterSize = Math.max(...clusterSizeValues);
    
    // Calculate segmentation quality score
    const idealClusterSize = Math.sqrt(faceCount);
    const sizeVariance = clusterSizeValues.reduce((sum, size) => sum + Math.pow(size - avgClusterSize, 2), 0) / clusterSizeValues.length;
    const normalizedVariance = sizeVariance / (avgClusterSize * avgClusterSize);
    const segmentationQuality = Math.max(0, 100 * (1 - normalizedVariance));
    
    // Update results display
    document.getElementById('totalFaces').textContent = faceCount.toLocaleString();
    document.getElementById('totalClusters').textContent = numClusters.toLocaleString();
    document.getElementById('rayHits').textContent = totalHits.toLocaleString();
    document.getElementById('hitRate').textContent = avgRaysPerFace.toFixed(1);
    document.getElementById('processingTime').textContent = processingTime.toFixed(2) + 's';
    document.getElementById('segmentationQuality').textContent = segmentationQuality.toFixed(1) + '%';
    
    document.getElementById('resultsSection').style.display = 'block';
    
    // Log detailed results
    window.log(`\n=== ANALYSIS RESULTS ===`);
    window.log(`Total faces: ${faceCount}`);
    window.log(`Total clusters: ${numClusters}`);
    window.log(`Ray hits: ${totalHits} (${avgRaysPerFace.toFixed(1)} avg per face)`);
    window.log(`Processing time: ${processingTime.toFixed(2)}s`);
    window.log(`Segmentation quality: ${segmentationQuality.toFixed(1)}%`);
    
    // Cluster size distribution
    const sizeCounts = {};
    clusterSizeValues.forEach(size => {
        sizeCounts[size] = (sizeCounts[size] || 0) + 1;
    });
    
    window.log(`\nCluster size distribution:`);
    window.log(`  Min: ${minClusterSize}, Max: ${maxClusterSize}, Avg: ${avgClusterSize.toFixed(1)}`);
    
    const sortedSizes = Object.entries(sizeCounts).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    const maxToShow = 10;
    for (let i = 0; i < Math.min(sortedSizes.length, maxToShow); i++) {
        const [size, count] = sortedSizes[i];
        window.log(`  Size ${size}: ${count} clusters`);
    }
    if (sortedSizes.length > maxToShow) {
        window.log(`  ... and ${sortedSizes.length - maxToShow} more size categories`);
    }
}

// NEW: Add visualization controls for different views
function addVisualizationControls(results) {
    // Check if controls already exist
    if (document.getElementById('visualizationControls')) {
        return;
    }
    
    const controlsHtml = `
        <div id="visualizationControls" class="control-section">
            <h3>Visualization Options</h3>
            <div class="button-group">
                <button class="btn btn-secondary" id="showClustersBtn">Show Clusters</button>
                <button class="btn btn-secondary" id="showOriginalBtn">Show Original</button>
            </div>
            <div class="button-group">
                <button class="btn btn-secondary" id="showRayPathsBtn" disabled>Show Ray Paths</button>
                <button class="btn btn-secondary" id="exportResultsBtn">Export Results</button>
            </div>
        </div>
    `;
    
    // Insert controls after the processing options
    const processingSection = document.querySelector('.controls-panel').lastElementChild.previousElementSibling;
    processingSection.insertAdjacentHTML('afterend', controlsHtml);
    
    // Add event listeners
    document.getElementById('showClustersBtn').onclick = () => {
        if (window.meshVisualization && window.currentMeshData) {
            window.meshVisualization.displayMesh(window.currentMeshData, results.clusters);
            window.log('Switched to cluster view');
        }
    };
    
    document.getElementById('showOriginalBtn').onclick = () => {
        if (window.meshVisualization && window.currentMeshData) {
            window.meshVisualization.displayMesh(window.currentMeshData);
            window.log('Switched to original mesh view');
        }
    };
    
    document.getElementById('showRayPathsBtn').onclick = () => {
        // TODO: Implement ray path visualization
        window.log('Ray path visualization not yet implemented');
    };
    
    document.getElementById('exportResultsBtn').onclick = () => {
        exportAnalysisResults(results);
    };
}

// NEW: Export analysis results
function exportAnalysisResults(results) {
    try {
        const exportData = {
            timestamp: new Date().toISOString(),
            faceCount: results.faceCount,
            clusters: results.clusters,
            hitCounts: results.hitCounts,
            totalIntersections: results.totalIntersections,
            processingTime: results.processingTime,
            clusterStatistics: calculateExportStatistics(results.clusters)
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `mesh_analysis_${Date.now()}.json`;
        link.click();
        
        window.log('Analysis results exported successfully');
        
    } catch (error) {
        window.showError(`Export failed: ${error.message}`);
        window.log(`Export error: ${error.message}`);
    }
}

function calculateExportStatistics(clusters) {
    const clusterSizes = {};
    Object.values(clusters).forEach(clusterId => {
        clusterSizes[clusterId] = (clusterSizes[clusterId] || 0) + 1;
    });
    
    const sizes = Object.values(clusterSizes);
    return {
        numClusters: Object.keys(clusterSizes).length,
        minSize: Math.min(...sizes),
        maxSize: Math.max(...sizes),
        avgSize: sizes.reduce((sum, size) => sum + size, 0) / sizes.length,
        sizeDistribution: clusterSizes
    };
}

// NEW: Enhanced error handling and validation
function validateMeshData(meshData) {
    if (!meshData || !meshData.vertices || !meshData.faces) {
        throw new Error('Invalid mesh data structure');
    }
    
    if (meshData.vertices.length === 0) {
        throw new Error('Mesh has no vertices');
    }
    
    if (meshData.faces.length === 0) {
        throw new Error('Mesh has no faces');
    }
    
    // Check for valid face indices
    const maxVertexIndex = meshData.vertices.length - 1;
    for (const face of meshData.faces) {
        if (face.length !== 3) {
            throw new Error('All faces must be triangles');
        }
        
        for (const vertexIndex of face) {
            if (vertexIndex < 0 || vertexIndex > maxVertexIndex) {
                throw new Error(`Invalid vertex index: ${vertexIndex}`);
            }
        }
    }
    
    return true;
}

// Auto-connect when the script loads
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a moment for the main script to load
        setTimeout(() => {
            try {
                connectMeshAlgorithm();
                console.log('Mesh algorithm connected successfully!');
                
                // Add global validation
                const originalLoadMesh = window.loadMeshFile;
                if (originalLoadMesh) {
                    window.loadMeshFile = function(file) {
                        try {
                            return originalLoadMesh(file);
                        } catch (error) {
                            window.showError(`Mesh loading failed: ${error.message}`);
                            throw error;
                        }
                    };
                }
                
            } catch (error) {
                console.error('Failed to connect mesh algorithm:', error);
            }
        }, 100);
    });
}