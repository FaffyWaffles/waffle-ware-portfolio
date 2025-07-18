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
            nBounces: options.nBounces || 3,
            nAdditionalRays: options.nAdditionalRays || 10,
            thetaDegrees: options.thetaDegrees || 30,
            batchSize: options.batchSize || 100,
            resolution: options.resolution || 1.0,
            minClusterSize: options.minClusterSize || 10,
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
    
    // Möller–Trumbore ray-triangle intersection
    rayTriangleIntersect(rayOrigin, rayDirection, faceIndex) {
        const epsilon = 1e-8;
        const v0 = this.v0Array[faceIndex];
        const edge1 = this.edge1Array[faceIndex];
        const edge2 = this.edge2Array[faceIndex];
        
        const h = rayDirection.cross(edge2);
        const a = edge1.dot(h);
        
        if (a > -epsilon && a < epsilon) {
            return null; // Ray is parallel to triangle
        }
        
        const f = 1.0 / a;
        const s = rayOrigin.subtract(v0);
        const u = f * s.dot(h);
        
        if (u < 0.0 || u > 1.0) {
            return null;
        }
        
        const q = s.cross(edge1);
        const v = f * rayDirection.dot(q);
        
        if (v < 0.0 || u + v > 1.0) {
            return null;
        }
        
        const t = f * edge2.dot(q);
        
        if (t > epsilon) {
            // Check if ray enters from back face (interior)
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
    
    generateConeRays(baseDirection, nRays, thetaRadians) {
        const rays = [baseDirection];
        
        for (let i = 0; i < nRays; i++) {
            // Generate random point on cone
            const phi = thetaRadians * Math.sqrt(Math.random());
            const omega = 2 * Math.PI * Math.random();
            
            const x = Math.sin(phi) * Math.cos(omega);
            const y = Math.sin(phi) * Math.sin(omega);
            const z = Math.cos(phi);
            
            const perturbation = new Vector3(x, y, z);
            const zAxis = new Vector3(0, 0, 1);
            let rotatedRay;
            
            if (Math.abs(baseDirection.dot(zAxis)) > 0.9999) {
                // Base direction is nearly parallel to z-axis
                rotatedRay = perturbation;
                if (baseDirection.z < 0) {
                    rotatedRay = rotatedRay.multiply(-1);
                }
            } else {
                // Use Rodrigues' rotation formula
                const axis = zAxis.cross(baseDirection).normalize();
                const angle = Math.acos(Math.max(-1, Math.min(1, zAxis.dot(baseDirection))));
                
                const cosAngle = Math.cos(angle);
                const sinAngle = Math.sin(angle);
                
                rotatedRay = perturbation.multiply(cosAngle)
                    .add(axis.cross(perturbation).multiply(sinAngle))
                    .add(axis.multiply(axis.dot(perturbation)).multiply(1 - cosAngle));
            }
            
            rays.push(rotatedRay.normalize());
        }
        
        return rays;
    }
    
    reflectRay(rayDirection, normal) {
        const dot = rayDirection.dot(normal);
        return rayDirection.subtract(normal.multiply(2 * dot)).normalize();
    }
    
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
        
        // Process faces in batches
        const batchSize = this.options.batchSize;
        const totalBatches = Math.ceil(this.faceCount / batchSize);
        
        for (let batchIdx = 0; batchIdx < totalBatches && !this.shouldStop; batchIdx++) {
            const startFace = batchIdx * batchSize;
            const endFace = Math.min(startFace + batchSize, this.faceCount);
            
            const batchResults = await this.processBatch(startFace, endFace, thetaRadians);
            
            totalIntersections += batchResults.intersections;
            rayChains.push(...batchResults.chains);
            
            // Merge transition graph
            for (const [from, to] of batchResults.transitions) {
                if (!transitionGraph[from]) transitionGraph[from] = {};
                if (!transitionGraph[from][to]) transitionGraph[from][to] = 0;
                transitionGraph[from][to]++;
            }
            
            // Update progress
            const progress = ((batchIdx + 1) / totalBatches) * 80; // 80% for ray tracing
            progressCallback?.({ 
                stage: 'ray_tracing', 
                progress, 
                batch: batchIdx + 1,
                totalBatches,
                intersections: totalIntersections
            });
            
            // Yield control to prevent blocking
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        if (this.shouldStop) {
            this.isAnalyzing = false;
            return null;
        }
        
        progressCallback?.({ stage: 'clustering', progress: 85 });
        
        // Perform clustering
        const clusters = await this.performClustering(transitionGraph);
        
        progressCallback?.({ stage: 'refinement', progress: 95 });
        
        // Refine clusters
        const refinedClusters = this.refinePartitionWithConnectivity(clusters);
        const finalClusters = this.mergeSmallPartitions(refinedClusters);
        
        const processingTime = (Date.now() - startTime) / 1000;
        
        this.isAnalyzing = false;
        
        return {
            hitCounts: this.hitCounts,
            transitionGraph,
            rayChains,
            clusters: finalClusters,
            totalIntersections,
            processingTime,
            faceCount: this.faceCount
        };
    }
    
    async processBatch(startFace, endFace, thetaRadians) {
        const batchResults = {
            intersections: 0,
            chains: [],
            transitions: []
        };
        
        for (let faceIdx = startFace; faceIdx < endFace; faceIdx++) {
            const origin = this.faceCentroids[faceIdx];
            const baseDirection = this.faceNormals[faceIdx].multiply(-1); // Inward ray
            
            // Generate rays in cone
            const rays = this.generateConeRays(
                baseDirection, 
                this.options.nAdditionalRays, 
                thetaRadians
            );
            
            for (const rayDirection of rays) {
                const chain = [faceIdx];
                let currentOrigin = origin.clone();
                let currentDirection = rayDirection.clone();
                let previousFace = faceIdx;
                
                // Trace ray through bounces
                for (let bounce = 0; bounce < this.options.nBounces; bounce++) {
                    const hit = this.findClosestIntersection(
                        currentOrigin, 
                        currentDirection, 
                        previousFace
                    );
                    
                    if (hit) {
                        this.hitCounts[hit.faceIndex]++;
                        batchResults.intersections++;
                        
                        chain.push(hit.faceIndex);
                        batchResults.transitions.push([previousFace, hit.faceIndex]);
                        
                        // Prepare for next bounce
                        if (bounce < this.options.nBounces - 1) {
                            currentOrigin = hit.hitPoint.clone();
                            const hitNormal = this.faceNormals[hit.faceIndex];
                            currentDirection = this.reflectRay(currentDirection, hitNormal);
                            previousFace = hit.faceIndex;
                        }
                    } else {
                        break; // No more intersections
                    }
                }
                
                if (chain.length > 1) {
                    batchResults.chains.push(chain);
                }
            }
        }
        
        return batchResults;
    }
    
    async performClustering(transitionGraph) {
        // Convert transition graph to adjacency list
        const graph = {};
        const edgeWeights = {};
        
        // Initialize graph
        for (let i = 0; i < this.faceCount; i++) {
            graph[i] = [];
        }
        
        // Build undirected graph with weights
        for (const [from, toMap] of Object.entries(transitionGraph)) {
            for (const [to, weight] of Object.entries(toMap)) {
                const fromIdx = parseInt(from);
                const toIdx = parseInt(to);
                
                if (!graph[fromIdx].includes(toIdx)) {
                    graph[fromIdx].push(toIdx);
                    graph[toIdx].push(fromIdx);
                    edgeWeights[`${Math.min(fromIdx, toIdx)},${Math.max(fromIdx, toIdx)}`] = weight;
                }
            }
        }
        
        // Simple Louvain-style clustering
        return this.louvainClustering(graph, edgeWeights);
    }
    
    louvainClustering(graph, edgeWeights) {
        const clusters = {};
        
        // Initialize each node as its own cluster
        for (let i = 0; i < this.faceCount; i++) {
            clusters[i] = i;
        }
        
        let improved = true;
        let iterations = 0;
        const maxIterations = 10;
        
        while (improved && iterations < maxIterations) {
            improved = false;
            
            // For each node, try to find a better cluster
            for (let nodeId = 0; nodeId < this.faceCount; nodeId++) {
                const currentCluster = clusters[nodeId];
                let bestCluster = currentCluster;
                let bestModularity = this.calculateNodeModularity(nodeId, currentCluster, clusters, graph, edgeWeights);
                
                // Check neighboring clusters
                const neighborClusters = new Set();
                for (const neighbor of graph[nodeId] || []) {
                    neighborClusters.add(clusters[neighbor]);
                }
                
                for (const neighborCluster of neighborClusters) {
                    if (neighborCluster !== currentCluster) {
                        const modularity = this.calculateNodeModularity(nodeId, neighborCluster, clusters, graph, edgeWeights);
                        if (modularity > bestModularity) {
                            bestModularity = modularity;
                            bestCluster = neighborCluster;
                            improved = true;
                        }
                    }
                }
                
                clusters[nodeId] = bestCluster;
            }
            
            iterations++;
        }
        
        return clusters;
    }
    
    calculateNodeModularity(nodeId, clusterId, clusters, graph, edgeWeights) {
        let internalWeight = 0;
        let totalWeight = 0;
        
        for (const neighbor of graph[nodeId] || []) {
            const edgeKey = `${Math.min(nodeId, neighbor)},${Math.max(nodeId, neighbor)}`;
            const weight = edgeWeights[edgeKey] || 1;
            
            if (clusters[neighbor] === clusterId) {
                internalWeight += weight;
            }
            totalWeight += weight;
        }
        
        return totalWeight > 0 ? internalWeight / totalWeight : 0;
    }
    
    refinePartitionWithConnectivity(partition) {
        const partitionGroups = {};
        
        // Group faces by partition
        for (const [faceIdx, clusterId] of Object.entries(partition)) {
            if (!partitionGroups[clusterId]) {
                partitionGroups[clusterId] = [];
            }
            partitionGroups[clusterId].push(parseInt(faceIdx));
        }
        
        let newClusterId = Math.max(...Object.keys(partitionGroups).map(Number)) + 1;
        const newPartition = { ...partition };
        
        // Process each partition separately
        for (const [clusterId, faces] of Object.entries(partitionGroups)) {
            if (faces.length <= 1) continue;
            
            // Create UnionFind for this partition
            const uf = new UnionFind(faces.length);
            const faceToLocal = {};
            
            faces.forEach((face, index) => {
                faceToLocal[face] = index;
            });
            
            // Union adjacent faces within this partition
            for (let i = 0; i < faces.length; i++) {
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
            
            // Assign new cluster IDs to additional components
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
        }
        
        return newPartition;
    }
    
    mergeSmallPartitions(partition) {
        const partitionSizes = {};
        
        // Calculate partition sizes
        for (const clusterId of Object.values(partition)) {
            partitionSizes[clusterId] = (partitionSizes[clusterId] || 0) + 1;
        }
        
        const newPartition = { ...partition };
        
        // Merge small partitions
        for (const [clusterId, size] of Object.entries(partitionSizes)) {
            if (size < this.options.minClusterSize) {
                // Find faces in this small partition
                const facesInPartition = [];
                for (const [faceIdx, faceCluster] of Object.entries(partition)) {
                    if (faceCluster == clusterId) {
                        facesInPartition.push(parseInt(faceIdx));
                    }
                }
                
                // Find the largest adjacent partition
                const neighborClusters = {};
                for (const faceIdx of facesInPartition) {
                    const adjacentFaces = this.faceAdjacency[faceIdx] || [];
                    for (const adjFace of adjacentFaces) {
                        const adjCluster = partition[adjFace];
                        if (adjCluster != clusterId) {
                            neighborClusters[adjCluster] = (neighborClusters[adjCluster] || 0) + 1;
                        }
                    }
                }
                
                // Find the most connected neighbor cluster
                let bestCluster = null;
                let bestCount = 0;
                for (const [neighborCluster, count] of Object.entries(neighborClusters)) {
                    if (count > bestCount) {
                        bestCount = count;
                        bestCluster = neighborCluster;
                    }
                }
                
                // Merge with best neighbor or largest partition
                if (bestCluster) {
                    for (const faceIdx of facesInPartition) {
                        newPartition[faceIdx] = bestCluster;
                    }
                } else {
                    // Merge with largest partition
                    const largestPartition = Object.entries(partitionSizes)
                        .reduce((a, b) => partitionSizes[a[0]] > partitionSizes[b[0]] ? a : b)[0];
                    
                    for (const faceIdx of facesInPartition) {
                        newPartition[faceIdx] = largestPartition;
                    }
                }
            }
        }
        
        return newPartition;
    }
    
    stop() {
        this.shouldStop = true;
    }
}

// === INTEGRATION WITH EXISTING UI ===

// Function to connect the algorithm to the existing UI
function connectMeshAlgorithm() {
    // This function will be called from the main HTML file
    // It integrates the algorithm with the existing UI components
    
    // Override the existing analyze button behavior
    const originalAnalyzeHandler = document.getElementById('analyzeBtn').onclick;
    
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
        
        // Create analyzer
        const analyzer = new MeshBounceAnalyzer(window.currentMeshData, options);
        
        // Update UI for analysis start
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        
        window.log('Starting mesh analysis...');
        
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
                
                window.log(`Analysis completed successfully in ${results.processingTime.toFixed(2)}s`);
            } else {
                window.log('Analysis was stopped by user');
            }
            
        } catch (error) {
            window.showError(`Analysis failed: ${error.message}`);
            window.log(`Analysis error: ${error.message}`);
        } finally {
            // Reset UI
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('progressSection').style.display = 'none';
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
    const { stage, progress: percent, batch, totalBatches, intersections } = progress;
    
    document.getElementById('progressFill').style.width = `${percent}%`;
    
    let statusText = '';
    switch (stage) {
        case 'initialization':
            statusText = 'Initializing analysis...';
            break;
        case 'ray_tracing':
            statusText = `Ray tracing: ${batch}/${totalBatches} batches (${intersections} hits)`;
            break;
        case 'clustering':
            statusText = 'Performing clustering...';
            break;
        case 'refinement':
            statusText = 'Refining clusters...';
            break;
        default:
            statusText = `Processing: ${percent.toFixed(1)}%`;
    }
    
    document.getElementById('progressText').textContent = statusText;
    window.log(statusText);
}

function displayAnalysisResults(results) {
    const { hitCounts, clusters, totalIntersections, processingTime, faceCount } = results;
    
    // Calculate statistics
    const totalHits = hitCounts.reduce((sum, count) => sum + count, 0);
    const numClusters = new Set(Object.values(clusters)).size;
    const maxPossibleHits = faceCount * results.clusters ? Object.keys(results.clusters).length : faceCount;
    const hitRate = (totalHits / maxPossibleHits) * 100;
    
    // Calculate segmentation quality
    const clusterSizes = {};
    Object.values(clusters).forEach(clusterId => {
        clusterSizes[clusterId] = (clusterSizes[clusterId] || 0) + 1;
    });
    
    const avgClusterSize = Object.values(clusterSizes).reduce((sum, size) => sum + size, 0) / Object.keys(clusterSizes).length;
    const idealClusterSize = Math.sqrt(faceCount);
    const segmentationQuality = Math.max(0, 100 - Math.abs(avgClusterSize - idealClusterSize) / idealClusterSize * 100);
    
    // Update results display
    document.getElementById('totalFaces').textContent = faceCount.toLocaleString();
    document.getElementById('totalClusters').textContent = numClusters.toLocaleString();
    document.getElementById('rayHits').textContent = totalHits.toLocaleString();
    document.getElementById('hitRate').textContent = hitRate.toFixed(1) + '%';
    document.getElementById('processingTime').textContent = processingTime.toFixed(2) + 's';
    document.getElementById('segmentationQuality').textContent = segmentationQuality.toFixed(1) + '%';
    
    document.getElementById('resultsSection').style.display = 'block';
    
    // Log detailed results
    window.log(`\n=== ANALYSIS RESULTS ===`);
    window.log(`Total faces: ${faceCount}`);
    window.log(`Total clusters: ${numClusters}`);
    window.log(`Ray hits: ${totalHits} (${hitRate.toFixed(1)}% hit rate)`);
    window.log(`Processing time: ${processingTime.toFixed(2)}s`);
    window.log(`Segmentation quality: ${segmentationQuality.toFixed(1)}%`);
    
    // Cluster size distribution
    const sizeCounts = {};
    Object.values(clusterSizes).forEach(size => {
        sizeCounts[size] = (sizeCounts[size] || 0) + 1;
    });
    
    window.log(`\nCluster size distribution:`);
    Object.entries(sizeCounts).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).forEach(([size, count]) => {
        window.log(`  Size ${size}: ${count} clusters`);
    });
}

// Auto-connect when the script loads
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a moment for the main script to load
        setTimeout(() => {
            connectMeshAlgorithm();
            console.log('Mesh algorithm connected successfully!');
        }, 100);
    });
}