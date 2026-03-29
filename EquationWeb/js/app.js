// ═══════════════════════════════════════════════════════════════════════════════
// EquationWeb — D3.js Force-Directed Graph Visualization
// ═══════════════════════════════════════════════════════════════════════════════

(function () {
    "use strict";

    // ─── State ───────────────────────────────────────────────────────────────
    let simulation, svg, g, linkGroup, nodeGroup, labelGroup;
    let allNodes = [], allLinks = [];
    let activeFilters = new Set(Object.keys(FIELD_INFO));
    let viewMode = "all"; // "all" | "constants" | "equations"
    let sizingMode = "fixed";
    let selectedNode = null;
    let selectedNodes = new Set();   // multi-select: set of selected node IDs
    let pathHighlight = null;        // { nodeIds: Set, linkKeys: Set } for shortest-path display
    let searchTerm = "";
    let width, height;
    let currentZoom = 1;
    let formMode = "standard";     // "standard" | "differential" | "integral" | "vector" | "tensor"
    let showAccuracy = true;        // show accuracy level indicators
    let showSupersession = true;    // show "generalizes" links between equations
    let showVariables = true;       // show variable nodes
    let showLabels = true;          // master label toggle
    let showCurves = true;          // curved vs straight edges
    let showHierarchy = true;       // hierarchical derivation layout
    let connectionMode = "variables"; // "variables" = via variable nodes, "direct" = equation↔equation
    let timelineYear = 2030;        // timeline cutoff year (max = show all)
    let timelinePlaying = false;    // animation state
    let timelineInterval = null;    // animation interval handle

    // Simulation physics tuning (multipliers relative to defaults)
    let simRepulsion = 1.0;         // charge strength multiplier
    let simLinkDist = 1.0;          // link distance multiplier
    let simLinkStrength = 1.0;      // link strength multiplier
    let simGravity = 0.015;         // centering force strength
    let simDecay = 0.35;            // velocity decay

    // ─── Build Graph from Data ───────────────────────────────────────────────

    function buildGraph() {
        const nodes = [];
        const links = [];
        const nodeMap = new Map();

        // Add constant nodes
        CONSTANTS.forEach(c => {
            const node = {
                id: c.id,
                name: c.name,
                symbol: c.symbol,
                type: "constant",
                value: c.value,
                unit: c.unit,
                description: c.description,
                connections: 0
            };
            nodes.push(node);
            nodeMap.set(c.id, node);
        });

        // Add variable nodes
        VARIABLES.forEach(v => {
            const node = {
                id: v.id,
                name: v.name,
                symbol: v.symbol,
                type: "variable",
                unit: v.unit,
                connections: 0
            };
            nodes.push(node);
            nodeMap.set(v.id, node);
        });

        // Add equation nodes and build links
        EQUATIONS.forEach(eq => {
            const meta = (typeof EQUATION_META !== 'undefined' && EQUATION_META[eq.id]) || {};
            const node = {
                id: eq.id,
                name: eq.name,
                type: "equation",
                field: eq.field,
                equation: eq.equation,
                description: eq.description,
                year: eq.year,
                discoverer: eq.discoverer,
                uses: eq.uses,
                connections: eq.uses.length,
                // Taxonomy metadata
                status: meta.status || "derived",
                conditions: meta.conditions || null,
                statusNote: meta.statusNote || null,
                supersededBy: meta.supersededBy || null,
                equivalentTo: meta.equivalentTo || null,
                incompatibleWith: meta.incompatibleWith || null,
                derivesFromData: meta.derivesFrom || null,
                componentOf: meta.componentOf || null,
                forms: meta.forms || null,
                formNotes: meta.formNotes || null
            };
            nodes.push(node);
            nodeMap.set(eq.id, node);

            // Create links from equation to each variable/constant it uses
            eq.uses.forEach(varId => {
                const target = nodeMap.get(varId);
                if (target) {
                    target.connections++;
                    links.push({
                        source: eq.id,
                        target: varId,
                        type: target.type === "constant" ? "constant" : "variable"
                    });
                }
            });
        });

        // Create supersession links (more general → less general)
        nodes.forEach(node => {
            if (node.supersededBy && nodeMap.has(node.supersededBy)) {
                links.push({
                    source: node.supersededBy,
                    target: node.id,
                    type: "generalizes"
                });
            }
        });

        // Create equivalent links (same physics, different formulation)
        const equivSeen = new Set();
        nodes.forEach(node => {
            if (node.equivalentTo) {
                node.equivalentTo.forEach(otherId => {
                    const key = [node.id, otherId].sort().join('|');
                    if (!equivSeen.has(key) && nodeMap.has(otherId)) {
                        equivSeen.add(key);
                        links.push({ source: node.id, target: otherId, type: "equivalent" });
                    }
                });
            }
        });

        // Create incompatible links (mutually exclusive domains)
        const incompSeen = new Set();
        nodes.forEach(node => {
            if (node.incompatibleWith) {
                node.incompatibleWith.forEach(otherId => {
                    const key = [node.id, otherId].sort().join('|');
                    if (!incompSeen.has(key) && nodeMap.has(otherId)) {
                        incompSeen.add(key);
                        links.push({ source: node.id, target: otherId, type: "incompatible" });
                    }
                });
            }
        });

        // Create derivesFrom links (logical derivation chains)
        nodes.forEach(node => {
            if (node.derivesFromData) {
                node.derivesFromData.forEach(d => {
                    if (nodeMap.has(d.eq)) {
                        links.push({ source: node.id, target: d.eq, type: "derivesFrom", assuming: d.assuming });
                    }
                });
            }
        });

        // Create componentOf links (equation is part of a system)
        const systemMembers = {};
        nodes.forEach(node => {
            if (node.componentOf) {
                if (!systemMembers[node.componentOf]) systemMembers[node.componentOf] = [];
                systemMembers[node.componentOf].push(node.id);
            }
        });
        // Link all members of the same system to each other
        const compSeen = new Set();
        Object.values(systemMembers).forEach(members => {
            for (let i = 0; i < members.length; i++) {
                for (let j = i + 1; j < members.length; j++) {
                    const key = [members[i], members[j]].sort().join('|');
                    if (!compSeen.has(key)) {
                        compSeen.add(key);
                        links.push({ source: members[i], target: members[j], type: "componentOf" });
                    }
                }
            }
        });

        return { nodes, links, nodeMap };
    }

    // ─── Rebuild Graph for Connection Mode Switch ─────────────────────────────

    function rebuildGraphMode() {
        // Preserve current node positions
        const posMap = {};
        allNodes.forEach(n => { if (n.x != null) posMap[n.id] = { x: n.x, y: n.y, vx: n.vx, vy: n.vy }; });

        // Rebuild base graph
        const graphData = buildGraph();

        if (connectionMode === "direct") {
            // Remove variable-type and constant-type links, build direct equation↔equation links
            const eqNodes = graphData.nodes.filter(n => n.type === "equation");
            const relationLinks = graphData.links.filter(l =>
                l.type === "generalizes" || l.type === "equivalent" ||
                l.type === "incompatible" || l.type === "derivesFrom" ||
                l.type === "componentOf"
            );

            // Build equation↔equation links from shared variables/constants
            const directLinks = [];
            const pairSeen = new Set();
            for (let i = 0; i < eqNodes.length; i++) {
                for (let j = i + 1; j < eqNodes.length; j++) {
                    const a = eqNodes[i], b = eqNodes[j];
                    const sharedVars = (a.uses || []).filter(u => (b.uses || []).includes(u));
                    if (sharedVars.length > 0) {
                        const key = [a.id, b.id].sort().join('|');
                        if (!pairSeen.has(key)) {
                            pairSeen.add(key);
                            const hasConst = sharedVars.some(v => v.startsWith("const_"));
                            directLinks.push({
                                source: a.id,
                                target: b.id,
                                type: hasConst ? "direct-const" : "direct-var",
                                sharedVars: sharedVars,
                                sharedCount: sharedVars.length
                            });
                        }
                    }
                }
            }

            // Keep equations + constants (constants as reference nodes), drop variables
            graphData.nodes = graphData.nodes.filter(n => n.type !== "variable");
            graphData.links = [...directLinks, ...relationLinks];

            // Update connection counts
            const countMap = {};
            directLinks.forEach(l => {
                countMap[l.source] = (countMap[l.source] || 0) + 1;
                countMap[l.target] = (countMap[l.target] || 0) + 1;
            });
            graphData.nodes.forEach(n => {
                if (n.type === "equation") n.connections = countMap[n.id] || 0;
            });
        }

        // Set data
        allNodes = graphData.nodes;
        allLinks = graphData.links;

        // Restore positions
        allNodes.forEach(n => {
            if (posMap[n.id]) {
                n.x = posMap[n.id].x;
                n.y = posMap[n.id].y;
                n.vx = posMap[n.id].vx;
                n.vy = posMap[n.id].vy;
            }
        });

        // Recompute edge bundling
        computeBundleOffsets(allLinks);
        computeDerivationDepths(allNodes, allLinks);

        // Update simulation
        simulation.nodes(allNodes);

        // Apply physics using slider-aware function (no reheat yet)
        applySimulationPhysics(false);
        simulation.force("collision", d3.forceCollide().radius(d => nodeRadius(d) + (connectionMode === "direct" ? 8 : 12)));

        // Re-render
        linkGroup.selectAll("*").remove();
        nodeGroup.selectAll("*").remove();
        renderGraph();

        // Reheat
        simulation.alpha(0.8).restart();
    }

    // ─── Apply Simulation Physics (slider-driven) ─────────────────────────────

    function applySimulationPhysics(reheat) {
        if (!simulation) return;

        // Link force
        if (connectionMode === "direct") {
            simulation.force("link", d3.forceLink(allLinks).id(d => d.id).distance(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 150;
                else if (d.type === "componentOf") base = 80;
                else base = d.sharedCount > 2 ? 60 : d.sharedCount > 1 ? 90 : 120;
                return base * simLinkDist;
            }).strength(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 0.08;
                else if (d.type === "componentOf") base = 0.12;
                else base = d.sharedCount > 2 ? 0.2 : d.sharedCount > 1 ? 0.12 : 0.06;
                return Math.min(1, base * simLinkStrength);
            }));
            simulation.force("charge", d3.forceManyBody()
                .strength(-250 * simRepulsion)
                .distanceMax(600));
        } else {
            simulation.force("link", d3.forceLink(allLinks).id(d => d.id).distance(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 150;
                else if (d.type === "componentOf") base = 80;
                else base = d.type === "constant" ? 140 : 110;
                return base * simLinkDist;
            }).strength(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 0.08;
                else if (d.type === "componentOf") base = 0.12;
                else base = 0.15;
                return Math.min(1, base * simLinkStrength);
            }));
            simulation.force("charge", d3.forceManyBody()
                .strength(d => {
                    let base;
                    if (d.type === "equation") base = -400;
                    else if (d.type === "constant") base = -600;
                    else base = -200;
                    return base * simRepulsion;
                })
                .distanceMax(800));
        }

        // Centering / gravity
        simulation.force("x", d3.forceX(width / 2).strength(simGravity));
        simulation.force("y", d3.forceY(height / 2).strength(simGravity));

        // Velocity decay
        simulation.velocityDecay(simDecay);

        if (reheat) simulation.alpha(0.6).restart();
    }

    // ─── Edge Bundling: Compute Curvature Offsets ──────────────────────────────
    // Groups links by shared endpoints and assigns offset indices so edges
    // from the same node fan out as curved bundles instead of overlapping.

    function computeBundleOffsets(links) {
        // Group links by the sorted pair of endpoints
        const pairGroups = {};
        links.forEach((l, i) => {
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            const key = [sid, tid].sort().join('|');
            if (!pairGroups[key]) pairGroups[key] = [];
            pairGroups[key].push(i);
        });

        // Assign offset for parallel links between the same pair
        Object.values(pairGroups).forEach(group => {
            const n = group.length;
            group.forEach((idx, i) => {
                links[idx].bundleOffset = n > 1 ? (i - (n - 1) / 2) : 0;
                links[idx].pairCount = n;
            });
        });

        // For links without parallel duplicates, compute a secondary offset
        // based on link index at the more-connected endpoint to create fanning
        const nodeLinkIndices = {};
        links.forEach((l, i) => {
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            if (!nodeLinkIndices[sid]) nodeLinkIndices[sid] = [];
            nodeLinkIndices[sid].push(i);
            if (!nodeLinkIndices[tid]) nodeLinkIndices[tid] = [];
            nodeLinkIndices[tid].push(i);
        });

        links.forEach((l, i) => {
            if (l.pairCount > 1) return; // already offset
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            // Use the more-connected endpoint for fanning
            const sGroup = nodeLinkIndices[sid] || [];
            const tGroup = nodeLinkIndices[tid] || [];
            const group = sGroup.length >= tGroup.length ? sGroup : tGroup;
            const idx = group.indexOf(i);
            const n = group.length;
            l.bundleOffset = n > 2 ? (idx - (n - 1) / 2) / n : 0;
        });
    }

    // ─── Hierarchical Layout: Derivation Depth ───────────────────────────────
    // Computes each equation's depth in the derivation DAG (including both
    // derivesFrom and generalizes chains). Root/postulate equations get depth 0;
    // equations that derive from them get depth 1, etc.

    let maxDerivDepth = 0;

    function computeDerivationDepths(nodes, links) {
        // Build parent→child adjacency from directed links
        const childrenOf = {};  // parentId → [childIds]
        const hasParent = new Set();

        links.forEach(l => {
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            if (l.type === "derivesFrom") {
                // source derives from target: target is parent, source is child
                if (!childrenOf[tid]) childrenOf[tid] = [];
                childrenOf[tid].push(sid);
                hasParent.add(sid);
            } else if (l.type === "generalizes") {
                // source (more general) generalizes target (less general)
                // source is parent, target is child
                if (!childrenOf[sid]) childrenOf[sid] = [];
                childrenOf[sid].push(tid);
                hasParent.add(tid);
            }
        });

        // BFS from root equations (no parents)
        const depths = {};
        const queue = [];
        nodes.forEach(n => {
            if (n.type === "equation" && !hasParent.has(n.id)) {
                depths[n.id] = 0;
                queue.push(n.id);
            }
        });

        while (queue.length > 0) {
            const id = queue.shift();
            const d = depths[id];
            (childrenOf[id] || []).forEach(childId => {
                if (depths[childId] === undefined || depths[childId] < d + 1) {
                    depths[childId] = d + 1;
                    queue.push(childId);
                }
            });
        }

        maxDerivDepth = Math.max(1, ...Object.values(depths));

        // Assign to nodes (-1 = not part of any derivation chain)
        nodes.forEach(n => {
            n.derivationDepth = depths[n.id] !== undefined ? depths[n.id] : -1;
        });

        return maxDerivDepth;
    }

    // ─── Link Path Generator (Curved Bézier) ─────────────────────────────────

    function linkPath(d) {
        const sx = d.source.x, sy = d.source.y;
        const tx = d.target.x, ty = d.target.y;
        if (sx == null || sy == null || tx == null || ty == null) return "M0,0";

        // Straight lines when curves are disabled
        if (!showCurves) return `M${sx},${sy} L${tx},${ty}`;

        const dx = tx - sx, dy = ty - sy;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        // Perpendicular unit normal
        const nx = -dy / dist, ny = dx / dist;

        // Curvature: all links get a base curve; bundled links fan further
        const isRelLink = d.type === "generalizes" || d.type === "equivalent" ||
                          d.type === "incompatible" || d.type === "derivesFrom" ||
                          d.type === "componentOf";

        // Deterministic per-link "jitter" so even unbundled links curve slightly
        // Use a hash of source+target IDs to get a consistent small offset
        const sid = d.source.id || d.source;
        const tid = d.target.id || d.target;
        let hash = 0;
        const hstr = sid + tid;
        for (let i = 0; i < hstr.length; i++) {
            hash = ((hash << 5) - hash + hstr.charCodeAt(i)) | 0;
        }
        const jitter = ((hash % 100) / 100) * 0.6 + 0.2; // 0.2–0.8

        const bundleOff = d.bundleOffset || 0;
        const baseCurve = isRelLink ? 22 : 32;
        const distFactor = Math.min(dist / 150, 1.5);

        // Every link curves: bundled links fan out, solo links get a gentle arc
        const offset = bundleOff !== 0
            ? bundleOff * baseCurve * distFactor
            : jitter * (isRelLink ? 12 : 20) * distFactor * (hash % 2 === 0 ? 1 : -1);

        // Control point at midpoint + perpendicular offset
        const mx = (sx + tx) / 2 + nx * offset;
        const my = (sy + ty) / 2 + ny * offset;

        return `M${sx},${sy} Q${mx},${my} ${tx},${ty}`;
    }

    // ─── Compute Node Radius ─────────────────────────────────────────────────

    function nodeRadius(d) {
        if (sizingMode === "connections") {
            const base = d.type === "equation" ? 6 : d.type === "constant" ? 5 : 3;
            return base + Math.sqrt(d.connections) * 2;
        }
        if (d.type === "equation") return 8;
        if (d.type === "constant") return 10;
        return 5;
    }

    // ─── Node Color ──────────────────────────────────────────────────────────

    function nodeColor(d) {
        if (d.type === "constant") return "#FFD700";
        if (d.type === "variable") return "#778899";
        if (d.field && FIELD_INFO[d.field]) return FIELD_INFO[d.field].color;
        return "#999";
    }

    // ─── Render Symbol with Subscripts/Superscripts ──────────────────────────
    //
    // Converts "k_B" → k + subscript B, "R_∞" → R + subscript ∞,
    // "T_μν" → T + subscript μν, "t₁/₂" → t₁⸝₂ (already unicode),
    // "S/N" stays as-is, "ε₀" stays as-is (already unicode sub).
    //
    // Pattern: looks for _ (LaTeX-style subscript) and ^ (superscript)
    // and renders them as <tspan> with dy shifts.

    function renderSymbolTspans(textSel, symbol) {
        if (!symbol) return;

        // Parse subscript/superscript notation: X_sub or X^sup or X_{sub} or X^{sup}
        const parts = [];
        let i = 0;
        while (i < symbol.length) {
            if ((symbol[i] === '_' || symbol[i] === '^') && i > 0) {
                const mode = symbol[i] === '_' ? 'sub' : 'sup';
                i++;
                let content = '';
                if (symbol[i] === '{') {
                    i++; // skip {
                    while (i < symbol.length && symbol[i] !== '}') {
                        content += symbol[i];
                        i++;
                    }
                    i++; // skip }
                } else {
                    // Grab all remaining chars as subscript (e.g. k_B, m_e, N_A, m_p, R_∞, T_μν, λ_mfp)
                    content = symbol.slice(i);
                    i = symbol.length;
                }
                parts.push({ text: content, mode: mode });
            } else {
                // Accumulate base text
                if (parts.length === 0 || parts[parts.length - 1].mode !== 'base') {
                    parts.push({ text: '', mode: 'base' });
                }
                parts[parts.length - 1].text += symbol[i];
                i++;
            }
        }

        // If no sub/superscript found, just set text directly
        if (parts.length === 1 && parts[0].mode === 'base') {
            textSel.text(symbol);
            return;
        }

        // Build tspan elements
        const fontSize = parseFloat(textSel.style("font-size")) || 10;
        parts.forEach(part => {
            const tspan = textSel.append("tspan").text(part.text);
            if (part.mode === 'sub') {
                tspan.attr("dy", fontSize * 0.35)
                     .attr("font-size", (fontSize * 0.7) + "px");
            } else if (part.mode === 'sup') {
                tspan.attr("dy", -fontSize * 0.35)
                     .attr("font-size", (fontSize * 0.7) + "px");
            }
        });

        // Reset baseline after sub/superscript (add invisible tspan to reset dy)
        const lastPart = parts[parts.length - 1];
        if (lastPart.mode === 'sub') {
            textSel.append("tspan").text("").attr("dy", -fontSize * 0.35);
        } else if (lastPart.mode === 'sup') {
            textSel.append("tspan").text("").attr("dy", fontSize * 0.35);
        }
    }

    // ─── Is Node Visible ─────────────────────────────────────────────────────

    // Helper: does this equation bridge 2+ different fields?
    function isEquationBridge(d) {
        if (d.type !== "equation") return false;
        const myField = d.field;
        const otherFields = new Set();
        (d.uses || []).forEach(uid => {
            allNodes.forEach(n => {
                if (n.type === "equation" && n.id !== d.id && (n.uses || []).includes(uid)) {
                    if (n.field !== myField) otherFields.add(n.field);
                }
            });
        });
        return otherFields.size >= 1;
    }

    function isNodeVisible(d) {
        // Timeline filter (equations only)
        if (d.type === "equation" && timelineYear < 2030) {
            if (!d.year || d.year > timelineYear) return false;
        }

        // Search filter
        if (searchTerm) {
            const term = searchTerm.toLowerCase();
            const match =
                d.name.toLowerCase().includes(term) ||
                (d.symbol && d.symbol.toLowerCase().includes(term)) ||
                (d.description && d.description.toLowerCase().includes(term)) ||
                (d.discoverer && d.discoverer.toLowerCase().includes(term)) ||
                (d.field && FIELD_INFO[d.field] && FIELD_INFO[d.field].name.toLowerCase().includes(term));
            if (!match) {
                // Also show if connected to a matching node
                const matchingIds = new Set(allNodes.filter(n => {
                    return n.name.toLowerCase().includes(term) ||
                        (n.symbol && n.symbol.toLowerCase().includes(term));
                }).map(n => n.id));
                const connected = allLinks.some(l => {
                    const sid = l.source.id || l.source;
                    const tid = l.target.id || l.target;
                    return (sid === d.id && matchingIds.has(tid)) || (tid === d.id && matchingIds.has(sid));
                });
                if (!connected) return false;
            }
        }

        // View mode filter
        if (viewMode === "constants") {
            if (d.type === "variable") return false;
            if (d.type === "equation") {
                // Only show equations connected to constants
                const usesConst = d.uses && d.uses.some(u => u.startsWith("const_"));
                if (!usesConst) return false;
            }
        }
        if (viewMode === "equations") {
            if (d.type === "variable" || d.type === "constant") return false;
        }

        // Bridge view: show only equations that connect 2+ different fields
        if (viewMode === "bridges") {
            if (d.type === "equation") {
                // An equation is a "bridge" if the variables/constants it uses
                // are also used by equations from OTHER fields
                const myField = d.field;
                const otherFields = new Set();
                (d.uses || []).forEach(uid => {
                    allNodes.forEach(n => {
                        if (n.type === "equation" && n.id !== d.id && (n.uses || []).includes(uid)) {
                            if (n.field !== myField) otherFields.add(n.field);
                        }
                    });
                });
                if (otherFields.size < 1) return false;
            }
            // Show variables/constants only if connected to a bridge equation
            if (d.type === "variable" || d.type === "constant") {
                const hasBridgeEq = allNodes.some(n => {
                    if (n.type !== "equation" || !(n.uses || []).includes(d.id)) return false;
                    if (!isEquationBridge(n)) return false;
                    return true;
                });
                if (!hasBridgeEq) return false;
            }
        }

        // Variable nodes toggle / direct connection mode
        if (d.type === "variable" && (!showVariables || connectionMode === "direct")) return false;

        // Field filter (for equations)
        if (d.type === "equation" && d.field && !activeFilters.has(d.field)) return false;

        return true;
    }

    // ─── Is Link Visible ─────────────────────────────────────────────────────

    function isLinkVisible(l) {
        const source = l.source.id ? l.source : allNodes.find(n => n.id === l.source);
        const target = l.target.id ? l.target : allNodes.find(n => n.id === l.target);
        if (!source || !target) return false;
        const isRelationLink = l.type === "generalizes" || l.type === "equivalent" || l.type === "incompatible" || l.type === "derivesFrom" || l.type === "componentOf";
        const isDirectLink = l.type === "direct-var" || l.type === "direct-const";
        if (isRelationLink && !showSupersession) return false;
        if (viewMode === "constants" && l.type !== "constant" && !isRelationLink && !isDirectLink) return false;
        if (viewMode === "equations" && !isRelationLink && !isDirectLink) return false;
        return isNodeVisible(source) && isNodeVisible(target);
    }

    // ─── Initialize SVG & Simulation ─────────────────────────────────────────

    function initGraph() {
        const container = document.getElementById("graph-container");
        width = container.clientWidth;
        height = container.clientHeight;

        svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 12])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
                const newZoom = event.transform.k;
                if (Math.abs(newZoom - currentZoom) > 0.05) {
                    currentZoom = newZoom;
                    updateLabelVisibility();
                }
                updateMinimap();
            });
        svg.call(zoom);
        mainZoom = zoom;

        // Background click to deselect
        svg.on("click", (event) => {
            if (event.target === svg.node()) {
                deselectAll();
            }
        });

        g = svg.append("g");

        // Arrow markers for directional links
        const defs = svg.append("defs");
        // Generalizes arrow (orange)
        defs.append("marker").attr("id", "arrow-generalizes").attr("viewBox", "0 0 10 6").attr("refX", 18).attr("refY", 3)
            .attr("markerWidth", 8).attr("markerHeight", 6).attr("orient", "auto")
            .append("path").attr("d", "M0,0 L10,3 L0,6 Z").attr("fill", "rgba(255,140,0,0.7)");
        // DerivesFrom arrow (green)
        defs.append("marker").attr("id", "arrow-derivesFrom").attr("viewBox", "0 0 10 6").attr("refX", 18).attr("refY", 3)
            .attr("markerWidth", 8).attr("markerHeight", 6).attr("orient", "auto")
            .append("path").attr("d", "M0,0 L10,3 L0,6 Z").attr("fill", "rgba(0,200,120,0.7)");

        // Layers
        linkGroup = g.append("g").attr("class", "links");
        nodeGroup = g.append("g").attr("class", "nodes");
        labelGroup = g.append("g").attr("class", "labels");

        // Build data
        const graphData = buildGraph();
        allNodes = graphData.nodes;
        allLinks = graphData.links;

        // Compute edge bundling offsets
        computeBundleOffsets(allLinks);

        // Compute derivation depths for hierarchical layout
        computeDerivationDepths(allNodes, allLinks);

        // Force simulation
        simulation = d3.forceSimulation(allNodes)
            .force("link", d3.forceLink(allLinks).id(d => d.id).distance(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 150;
                else if (d.type === "componentOf") base = 80;
                else base = d.type === "constant" ? 140 : 110;
                return base * simLinkDist;
            }).strength(d => {
                let base;
                if (d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom") base = 0.08;
                else if (d.type === "componentOf") base = 0.12;
                else base = 0.15;
                return Math.min(1, base * simLinkStrength);
            }))
            .force("charge", d3.forceManyBody().strength(d => {
                let base;
                if (d.type === "equation") base = -400;
                else if (d.type === "constant") base = -600;
                else base = -200;
                return base * simRepulsion;
            }).distanceMax(800))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 12))
            .force("x", d3.forceX(width / 2).strength(simGravity))
            .force("y", d3.forceY(height / 2).strength(simGravity))
            // Hierarchical force: push root/postulate equations upward,
            // derived equations downward, proportional to derivation depth
            .force("derivationY", d3.forceY(d => {
                if (d.derivationDepth >= 0) {
                    return height * 0.12 + (d.derivationDepth / maxDerivDepth) * height * 0.76;
                }
                return height / 2;
            }).strength(d => d.derivationDepth >= 0 ? 0.025 : 0))
            .alphaDecay(0.008)
            .velocityDecay(simDecay);

        renderGraph();

        simulation.on("tick", ticked);

        // Initial zoom to fit
        setTimeout(() => {
            const bounds = g.node().getBBox();
            if (bounds.width > 0 && bounds.height > 0) {
                const scale = 0.8 * Math.min(width / bounds.width, height / bounds.height);
                const tx = width / 2 - scale * (bounds.x + bounds.width / 2);
                const ty = height / 2 - scale * (bounds.y + bounds.height / 2);
                svg.transition().duration(1000)
                    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
            }
        }, 3000);
    }

    // ─── Render Graph ────────────────────────────────────────────────────────

    function renderGraph() {
        // Links
        const linkSel = linkGroup.selectAll("path.link")
            .data(allLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

        linkSel.exit().remove();

        const linkEnter = linkSel.enter().append("path")
            .attr("class", d => `link link-${d.type}`)
            .attr("fill", "none")
            .attr("stroke", d => {
                if (d.type === "generalizes") return "rgba(255,140,0,0.6)";
                if (d.type === "equivalent") return "rgba(0,210,255,0.6)";
                if (d.type === "incompatible") return "rgba(255,60,60,0.55)";
                if (d.type === "derivesFrom") return "rgba(0,200,120,0.5)";
                if (d.type === "componentOf") return "rgba(180,130,255,0.5)";
                if (d.type === "direct-const") return "rgba(255,215,0,0.35)";
                if (d.type === "direct-var") {
                    return d.sharedCount > 2 ? "rgba(150,200,255,0.4)" : d.sharedCount > 1 ? "rgba(150,170,190,0.3)" : "rgba(150,170,190,0.15)";
                }
                if (d.type === "constant") return "rgba(255,215,0,0.25)";
                return "rgba(150,170,190,0.18)";
            })
            .attr("stroke-width", d => {
                if (d.type === "generalizes") return 2.0;
                if (d.type === "equivalent") return 2.0;
                if (d.type === "incompatible") return 1.8;
                if (d.type === "derivesFrom") return 1.8;
                if (d.type === "componentOf") return 1.8;
                if (d.type === "direct-const") return 1.4;
                if (d.type === "direct-var") return d.sharedCount > 2 ? 1.8 : d.sharedCount > 1 ? 1.2 : 0.6;
                if (d.type === "constant") return 1.2;
                return 0.7;
            })
            .attr("stroke-dasharray", d => {
                if (d.type === "generalizes") return "6,3";
                if (d.type === "incompatible") return "3,3";
                if (d.type === "derivesFrom") return "8,3,2,3";
                return null;
            })
            .attr("marker-end", d => {
                if (d.type === "generalizes") return "url(#arrow-generalizes)";
                if (d.type === "derivesFrom") return "url(#arrow-derivesFrom)";
                return null;
            })
            .style("display", d => {
                const isRel = d.type === "generalizes" || d.type === "equivalent" || d.type === "incompatible" || d.type === "derivesFrom" || d.type === "componentOf";
                if (isRel && !showSupersession) return "none";
                return null;
            })
            .style("pointer-events", d => (d.type === "direct-var" || d.type === "direct-const") ? "stroke" : null)
            .on("mouseenter", function(event, d) {
                if (d.type !== "direct-var" && d.type !== "direct-const") return;
                const tooltip = d3.select("#tooltip");
                const nodeMap = {};
                allNodes.forEach(n => nodeMap[n.id] = n);
                const srcName = nodeMap[d.source.id || d.source]?.name || d.source.id || d.source;
                const tgtName = nodeMap[d.target.id || d.target]?.name || d.target.id || d.target;
                const varNames = (d.sharedVars || []).map(vid => {
                    const v = VARIABLES.find(x => x.id === vid) || CONSTANTS.find(x => x.id === vid);
                    return v ? v.name : vid;
                }).join(", ");
                tooltip.style("display", "block")
                    .style("left", (event.pageX + 12) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .html(`<strong>${srcName} ↔ ${tgtName}</strong><br>Shared: ${varNames} (${d.sharedCount})`);
            })
            .on("mouseleave", function() {
                d3.select("#tooltip").style("display", "none");
            });

        // Nodes
        const nodeSel = nodeGroup.selectAll("g.node")
            .data(allNodes, d => d.id);

        nodeSel.exit().remove();

        const nodeEnter = nodeSel.enter().append("g")
            .attr("class", d => `node node-${d.type}`)
            .call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded));

        // Node shapes
        nodeEnter.each(function (d) {
            const el = d3.select(this);
            if (d.type === "constant") {
                // Diamond shape for constants
                const r = nodeRadius(d);
                el.append("polygon")
                    .attr("points", `0,${-r} ${r},0 0,${r} ${-r},0`)
                    .attr("fill", "#FFD700")
                    .attr("stroke", "#FFF8DC")
                    .attr("stroke-width", 1.5)
                    .attr("class", "node-shape");
            } else if (d.type === "equation") {
                el.append("circle")
                    .attr("r", nodeRadius(d))
                    .attr("fill", nodeColor(d))
                    .attr("stroke", "rgba(255,255,255,0.3)")
                    .attr("stroke-width", 1)
                    .attr("class", "node-shape");
            } else {
                el.append("circle")
                    .attr("r", nodeRadius(d))
                    .attr("fill", nodeColor(d))
                    .attr("stroke", "rgba(255,255,255,0.15)")
                    .attr("stroke-width", 0.5)
                    .attr("class", "node-shape");
            }
        });

        // ─── Status-Based Visual Encoding ─────────────────────────────
        // Each equation status gets a distinct border style and badge glyph
        nodeEnter.filter(d => d.type === "equation" && showAccuracy).each(function(d) {
            const el = d3.select(this);
            const r = nodeRadius(d);
            const si = (typeof STATUS_INFO !== 'undefined') ? STATUS_INFO[d.status] : null;
            if (!si) return;

            if (d.status === "postulate") {
                // Thick gold solid border
                el.insert("circle", ".node-shape")
                    .attr("r", r + 3)
                    .attr("fill", "none")
                    .attr("stroke", "#FFD700")
                    .attr("stroke-width", 2.5)
                    .attr("stroke-opacity", 0.7)
                    .attr("class", "status-ring status-postulate");
            } else if (d.status === "solution") {
                // Double green rings
                el.insert("circle", ".node-shape")
                    .attr("r", r + 2)
                    .attr("fill", "none")
                    .attr("stroke", "#00e676")
                    .attr("stroke-width", 1)
                    .attr("stroke-opacity", 0.6)
                    .attr("class", "status-ring status-solution");
                el.insert("circle", ".node-shape")
                    .attr("r", r + 4)
                    .attr("fill", "none")
                    .attr("stroke", "#00e676")
                    .attr("stroke-width", 1)
                    .attr("stroke-opacity", 0.6)
                    .attr("class", "status-ring status-solution");
            } else if (d.status === "approximation") {
                // Orange dashed ring
                el.insert("circle", ".node-shape")
                    .attr("r", r + 2)
                    .attr("fill", "none")
                    .attr("stroke", "rgba(255,140,0,0.6)")
                    .attr("stroke-width", 1.2)
                    .attr("stroke-dasharray", "3,2")
                    .attr("class", "status-ring status-approx");
            } else if (d.status === "empirical") {
                // Pink dotted ring
                el.insert("circle", ".node-shape")
                    .attr("r", r + 2)
                    .attr("fill", "none")
                    .attr("stroke", "rgba(255,107,157,0.6)")
                    .attr("stroke-width", 1.2)
                    .attr("stroke-dasharray", "1.5,2")
                    .attr("class", "status-ring status-empirical");
            } else if (d.status === "definition") {
                // Thin grey solid ring
                el.insert("circle", ".node-shape")
                    .attr("r", r + 2)
                    .attr("fill", "none")
                    .attr("stroke", "rgba(136,146,164,0.4)")
                    .attr("stroke-width", 0.8)
                    .attr("class", "status-ring status-definition");
            }
            // derived: no ring (clean default)

            // Badge glyph for all statuses except derived
            if (d.status !== "derived") {
                el.append("text")
                    .attr("class", "status-badge")
                    .attr("x", r - 1)
                    .attr("y", -r + 3)
                    .text(si.badge)
                    .style("font-size", "9px")
                    .style("fill", si.color)
                    .style("font-weight", "bold")
                    .style("pointer-events", "none");
            }
        });

        // Data quality warning badges (⚠ for incomplete data)
        nodeEnter.filter(d => d.type === "equation" && d.hasDataIssues).each(function(d) {
            const el = d3.select(this);
            const r = nodeRadius(d);
            el.append("text")
                .attr("class", "dq-badge")
                .attr("x", -r + 1)
                .attr("y", -r + 3)
                .text("⚠")
                .style("font-size", "8px")
                .style("fill", "#F1C40F")
                .style("opacity", 0.8)
                .style("pointer-events", "none");
        });

        // ─── Labels ────────────────────────────────────────────────────
        // Constants: always show symbol with proper sub/superscripts
        nodeEnter.filter(d => d.type === "constant").each(function(d) {
            const lbl = d3.select(this).append("text")
                .attr("class", "node-label constant-label always-visible")
                .attr("dy", d => nodeRadius(d) + 14)
                .attr("text-anchor", "middle")
                .style("font-size", "10px")
                .style("fill", "#FFD700")
                .style("font-weight", "bold")
                .style("pointer-events", "none");
            renderSymbolTspans(lbl, d.symbol);
        });

        // Variables: show symbol at medium zoom with proper sub/superscripts
        nodeEnter.filter(d => d.type === "variable").each(function(d) {
            const lbl = d3.select(this).append("text")
                .attr("class", "node-label variable-label zoom-medium")
                .attr("dy", d => nodeRadius(d) + 12)
                .attr("text-anchor", "middle")
                .style("font-size", "9px")
                .style("fill", "rgba(180,195,215,0.9)")
                .style("font-weight", "600")
                .style("pointer-events", "none")
                .style("display", "none");
            renderSymbolTspans(lbl, d.symbol);
        });

        // Equations: KaTeX rendered via foreignObject, shown at low-medium zoom
        nodeEnter.filter(d => d.type === "equation").each(function(d) {
            const el = d3.select(this);
            const fo = el.append("foreignObject")
                .attr("class", "eq-fo zoom-low")
                .attr("width", 200)
                .attr("height", 50)
                .attr("x", -100)
                .attr("y", nodeRadius(d) + 4)
                .style("overflow", "visible")
                .style("pointer-events", "none")
                .style("display", "none");

            const div = fo.append("xhtml:div")
                .attr("class", "eq-label-katex")
                .style("text-align", "center")
                .style("pointer-events", "none");

            // Render KaTeX
            try {
                katex.render(d.equation, div.node(), {
                    throwOnError: false,
                    displayMode: false,
                    output: "html"
                });
            } catch(e) {
                div.text(d.equation);
            }
        });

        // Initial label visibility
        updateLabelVisibility();

        // Tooltip events
        nodeEnter
            .on("mouseenter", handleMouseEnter)
            .on("mouseleave", handleMouseLeave)
            .on("click", handleNodeClick);

        updateVisibility();
    }

    // ─── Tick ────────────────────────────────────────────────────────────────

    function ticked() {
        linkGroup.selectAll("path.link")
            .attr("d", linkPath);

        nodeGroup.selectAll("g.node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        // Throttled minimap update
        if (simulation.alpha() > 0.05 && Math.random() < 0.1) updateMinimap();
    }

    // ─── Drag ────────────────────────────────────────────────────────────────

    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.1).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // ─── Mouse Events ────────────────────────────────────────────────────────

    function handleMouseEnter(event, d) {
        showTooltip(event, d);
        // Don't override highlight when a node is already selected or multi-select path is active
        if (!selectedNode && !pathHighlight) {
            highlightConnections(d, true);
        }
    }

    function handleMouseLeave(event, d) {
        hideTooltip();
        if (!selectedNode && !pathHighlight) {
            highlightConnections(d, false);
        }
    }

    function handleNodeClick(event, d) {
        event.stopPropagation();

        // Shift+click: multi-select for path finding
        if (event.shiftKey) {
            if (selectedNodes.has(d.id)) {
                selectedNodes.delete(d.id);
            } else {
                selectedNodes.add(d.id);
            }

            // If we have 2+ selected, compute shortest path between all pairs
            if (selectedNodes.size >= 2) {
                pathHighlight = computeMultiPath(selectedNodes);
            } else {
                pathHighlight = null;
            }

            applyMultiSelectHighlight();
            // Show detail for the last-clicked node
            showDetailPanel(d);
            return;
        }

        // Normal click: clear multi-select, single-select behavior
        selectedNodes.clear();
        pathHighlight = null;

        if (selectedNode === d) {
            deselectAll();
            return;
        }

        // Deselect previous
        if (selectedNode) highlightConnections(selectedNode, false);

        selectedNode = d;
        highlightConnections(d, true);
        showDetailPanel(d);
    }

    // ─── BFS Shortest Path Between Multi-Selected Nodes ──────────────────

    function computeMultiPath(nodeIdSet) {
        // Build adjacency list from ALL links (not just visible)
        const adj = {};
        allLinks.forEach(l => {
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            if (!adj[sid]) adj[sid] = [];
            if (!adj[tid]) adj[tid] = [];
            adj[sid].push({ neighbor: tid, link: l });
            adj[tid].push({ neighbor: sid, link: l });
        });

        const selectedArr = [...nodeIdSet];
        const allPathNodes = new Set();
        const allPathLinks = new Set();

        // BFS shortest path between every pair
        for (let i = 0; i < selectedArr.length; i++) {
            for (let j = i + 1; j < selectedArr.length; j++) {
                const path = bfsPath(adj, selectedArr[i], selectedArr[j]);
                if (path) {
                    path.nodes.forEach(id => allPathNodes.add(id));
                    path.links.forEach(key => allPathLinks.add(key));
                }
            }
        }

        // Always include selected nodes themselves
        selectedArr.forEach(id => allPathNodes.add(id));

        return { nodeIds: allPathNodes, linkKeys: allPathLinks };
    }

    function bfsPath(adj, startId, endId) {
        if (startId === endId) return { nodes: [startId], links: [] };
        const visited = new Set([startId]);
        const queue = [[startId]];
        const linkMap = {}; // child → link used to reach it

        while (queue.length > 0) {
            const path = queue.shift();
            const current = path[path.length - 1];

            for (const edge of (adj[current] || [])) {
                if (visited.has(edge.neighbor)) continue;
                visited.add(edge.neighbor);

                const newPath = [...path, edge.neighbor];
                const sid = edge.link.source.id || edge.link.source;
                const tid = edge.link.target.id || edge.link.target;
                const linkKey = `${sid}-${tid}`;
                linkMap[edge.neighbor] = linkKey;

                if (edge.neighbor === endId) {
                    // Reconstruct link keys
                    const linkKeys = [];
                    for (let k = 1; k < newPath.length; k++) {
                        // Find the link key for this step
                        linkKeys.push(linkMap[newPath[k]]);
                    }
                    return { nodes: newPath, links: linkKeys };
                }
                queue.push(newPath);
            }
        }
        return null; // no path found
    }

    function applyMultiSelectHighlight() {
        if (selectedNodes.size === 0) {
            deselectAll();
            return;
        }

        const highlightNodes = pathHighlight ? pathHighlight.nodeIds : selectedNodes;
        const highlightLinks = pathHighlight ? pathHighlight.linkKeys : new Set();

        nodeGroup.selectAll("g.node")
            .style("opacity", n => highlightNodes.has(n.id) ? 1 : 0.08)
            .classed("multi-selected", n => selectedNodes.has(n.id));

        linkGroup.selectAll("path.link")
            .style("opacity", l => {
                const sid = l.source.id || l.source;
                const tid = l.target.id || l.target;
                const key = `${sid}-${tid}`;
                return highlightLinks.has(key) ? 1 : 0.02;
            })
            .attr("stroke-width", l => {
                const sid = l.source.id || l.source;
                const tid = l.target.id || l.target;
                const key = `${sid}-${tid}`;
                return highlightLinks.has(key) ? 3.5 : null;
            })
            .attr("stroke", l => {
                const sid = l.source.id || l.source;
                const tid = l.target.id || l.target;
                const key = `${sid}-${tid}`;
                if (highlightLinks.has(key)) return "#FFD700";
                return null;
            });
    }

    function deselectAll() {
        if (selectedNode) highlightConnections(selectedNode, false);
        selectedNode = null;
        selectedNodes.clear();
        pathHighlight = null;
        hideDetailPanel();

        // Reset all opacities
        nodeGroup.selectAll("g.node").style("opacity", 1).classed("multi-selected", false);
        linkGroup.selectAll("path.link").style("opacity", null).attr("stroke-width", null);
    }

    // ─── Highlight Connections ───────────────────────────────────────────────

    function highlightConnections(d, highlight) {
        const connectedIds = new Set();
        connectedIds.add(d.id);

        allLinks.forEach(l => {
            const sid = l.source.id || l.source;
            const tid = l.target.id || l.target;
            if (sid === d.id) connectedIds.add(tid);
            if (tid === d.id) connectedIds.add(sid);
        });

        if (highlight) {
            nodeGroup.selectAll("g.node")
                .style("opacity", n => connectedIds.has(n.id) ? 1 : 0.08);

            linkGroup.selectAll("path.link")
                .style("opacity", l => {
                    const sid = l.source.id || l.source;
                    const tid = l.target.id || l.target;
                    return (sid === d.id || tid === d.id) ? 1 : 0.02;
                })
                .attr("stroke-width", l => {
                    const sid = l.source.id || l.source;
                    const tid = l.target.id || l.target;
                    if (sid === d.id || tid === d.id) {
                        if (l.type === "generalizes" || l.type === "equivalent" || l.type === "incompatible"
                            || l.type === "derivesFrom" || l.type === "componentOf") return 3;
                        return l.type === "constant" ? 2.5 : 1.5;
                    }
                    return null;
                })
                .attr("stroke", l => {
                    const sid = l.source.id || l.source;
                    const tid = l.target.id || l.target;
                    if (sid === d.id || tid === d.id) {
                        if (l.type === "generalizes") return "#FF8C00";
                        if (l.type === "equivalent") return "#00D2FF";
                        if (l.type === "incompatible") return "#FF3C3C";
                        if (l.type === "derivesFrom") return "#00c878";
                        if (l.type === "componentOf") return "#B482FF";
                        return l.type === "constant" ? "#FFD700" : "#aabbcc";
                    }
                    return null;
                });
        } else {
            nodeGroup.selectAll("g.node").style("opacity", 1);
            linkGroup.selectAll("path.link")
                .style("opacity", null)
                .attr("stroke-width", null)
                .attr("stroke", d => {
                    if (d.type === "generalizes") return "rgba(255,140,0,0.6)";
                    if (d.type === "equivalent") return "rgba(0,210,255,0.6)";
                    if (d.type === "incompatible") return "rgba(255,60,60,0.55)";
                    if (d.type === "derivesFrom") return "rgba(0,200,120,0.5)";
                    if (d.type === "componentOf") return "rgba(180,130,255,0.5)";
                    if (d.type === "constant") return "rgba(255,215,0,0.25)";
                    return "rgba(150,170,190,0.18)";
                });
        }
    }

    // ─── Label Visibility by Zoom ─────────────────────────────────────────────
    //
    // Tier 1 (always): constant symbols
    // Tier 2 (zoom >= 1.0): equation TeX labels
    // Tier 3 (zoom >= 2.0): variable symbols

    function updateLabelVisibility() {
        const z = currentZoom;

        // Equation TeX labels
        nodeGroup.selectAll(".eq-fo")
            .style("display", (showLabels && z >= 0.6) ? null : "none");

        // Variable symbol labels — show much earlier
        nodeGroup.selectAll(".variable-label")
            .style("display", (showLabels && z >= 0.4) ? null : "none");
    }

    // ─── Form Display Switching ───────────────────────────────────────────────
    //
    // Switch equation labels between standard, differential, integral, vector, tensor

    function updateFormDisplay() {
        nodeGroup.selectAll(".eq-fo").each(function(d) {
            if (d.type !== "equation") return;
            const div = d3.select(this).select(".eq-label-katex");
            // Use alternative form if available, else fall back to standard
            const tex = (formMode !== "standard" && d.forms && d.forms[formMode])
                ? d.forms[formMode]
                : d.equation;
            div.html("");
            try {
                katex.render(tex, div.node(), {
                    throwOnError: false,
                    displayMode: false,
                    output: "html"
                });
            } catch(e) {
                div.text(tex);
            }
        });
    }

    // ─── Update Visibility ───────────────────────────────────────────────────

    function updateVisibility() {
        nodeGroup.selectAll("g.node")
            .style("display", d => isNodeVisible(d) ? null : "none");

        linkGroup.selectAll("path.link")
            .style("display", d => isLinkVisible(d) ? null : "none");

        // Re-apply zoom-based label visibility
        updateLabelVisibility();

        // Update stats
        const visibleEqs = allNodes.filter(n => n.type === "equation" && isNodeVisible(n)).length;
        const visibleConsts = allNodes.filter(n => n.type === "constant" && isNodeVisible(n)).length;
        const visibleVars = allNodes.filter(n => n.type === "variable" && isNodeVisible(n)).length;
        const visibleLinks = allLinks.filter(l => isLinkVisible(l)).length;

        document.getElementById("stats").innerHTML =
            `<span class="stat">${visibleEqs} equations</span>` +
            `<span class="stat">${visibleConsts + visibleVars} variables</span>` +
            `<span class="stat">${visibleLinks} connections</span>`;

        // Reheat simulation gently
        simulation.alpha(0.3).restart();

        // Refresh minimap
        updateMinimap();
    }

    // ─── Tooltip ─────────────────────────────────────────────────────────────

    function showTooltip(event, d) {
        const tooltip = document.getElementById("tooltip");
        let html = "";

        if (d.type === "equation") {
            const fieldInfo = FIELD_INFO[d.field] || {};
            const si = (typeof STATUS_INFO !== 'undefined' && d.status) ? STATUS_INFO[d.status] : null;
            const statusHtml = si
                ? `<div class="tooltip-level" style="color:${si.color}">${si.badge} ${si.label}</div>`
                : "";
            const condHtml = d.conditions
                ? `<div class="tooltip-domain">Valid: <span id="tooltip-domain-tex"></span></div>`
                : "";
            html = `
                <div class="tooltip-field" style="color:${fieldInfo.color || '#fff'}">
                    ${fieldInfo.icon || ''} ${fieldInfo.name || d.field}
                </div>
                <div class="tooltip-name">${d.name}</div>
                ${statusHtml}
                ${condHtml}
                <div class="tooltip-desc">${d.description}</div>
                <div class="tooltip-meta">${d.discoverer || ''} ${d.year ? '(' + d.year + ')' : ''}</div>
            `;
        } else if (d.type === "constant") {
            html = `
                <div class="tooltip-field" style="color:#FFD700">◆ Fundamental Constant</div>
                <div class="tooltip-name">${d.name} (${d.symbol})</div>
                <div class="tooltip-value">${d.value} ${d.unit}</div>
                <div class="tooltip-desc">${d.description}</div>
                <div class="tooltip-meta">${d.connections} equation connections</div>
            `;
        } else {
            html = `
                <div class="tooltip-field" style="color:#778899">○ Physical Variable</div>
                <div class="tooltip-name">${d.name} (${d.symbol})</div>
                <div class="tooltip-unit">${d.unit ? 'Unit: ' + d.unit : ''}</div>
                <div class="tooltip-meta">${d.connections} equation connections</div>
            `;
        }

        tooltip.innerHTML = html;
        tooltip.classList.add("visible");

        // Position tooltip
        const rect = tooltip.getBoundingClientRect();
        let x = event.pageX + 15;
        let y = event.pageY - 10;
        if (x + rect.width > window.innerWidth - 20) x = event.pageX - rect.width - 15;
        if (y + rect.height > window.innerHeight - 20) y = event.pageY - rect.height - 10;
        tooltip.style.left = x + "px";
        tooltip.style.top = y + "px";

        // Render conditions tex in tooltip if present
        if (d.type === "equation" && d.conditions) {
            const domEl = document.getElementById("tooltip-domain-tex");
            if (domEl) {
                try {
                    katex.render(d.conditions, domEl, { throwOnError: false, displayMode: false });
                } catch(e) {
                    domEl.textContent = d.conditions;
                }
            }
        }
    }

    function hideTooltip() {
        document.getElementById("tooltip").classList.remove("visible");
    }

    // ─── Detail Panel ────────────────────────────────────────────────────────

    function showDetailPanel(d) {
        const panel = document.getElementById("detail-panel");
        const content = document.getElementById("detail-content");
        let html = "";

        if (d.type === "equation") {
            const fieldInfo = FIELD_INFO[d.field] || {};

            // Find connected variables and constants
            const connectedVars = [];
            const connectedConsts = [];
            (d.uses || []).forEach(uid => {
                // In direct mode, variable nodes aren't in allNodes; fall back to source data
                let node = allNodes.find(n => n.id === uid);
                if (!node) node = VARIABLES.find(v => v.id === uid) || CONSTANTS.find(c => c.id === uid);
                if (node) {
                    if (node.type === "constant") connectedConsts.push(node);
                    else connectedVars.push(node);
                }
            });

            // Find related equations (share at least 2 variables/constants)
            const relatedEqs = allNodes.filter(n => {
                if (n.type !== "equation" || n.id === d.id) return false;
                const shared = (n.uses || []).filter(u => (d.uses || []).includes(u));
                return shared.length >= 2;
            }).sort((a, b) => {
                const sa = (a.uses || []).filter(u => (d.uses || []).includes(u)).length;
                const sb = (b.uses || []).filter(u => (d.uses || []).includes(u)).length;
                return sb - sa;
            }).slice(0, 8);

            // Build status & taxonomy section
            const si = (typeof STATUS_INFO !== 'undefined' && d.status) ? STATUS_INFO[d.status] : null;

            let taxonomyHtml = "";
            if (si || d.conditions || d.supersededBy || d.derivesFromData || d.componentOf) {
                // Supersession chain (generalizes)
                let chainHtml = "";
                const chain = [];
                let current = d;
                while (current && current.supersededBy) {
                    const parent = allNodes.find(n => n.id === current.supersededBy);
                    if (parent) chain.push(parent);
                    current = parent;
                }
                const children = allNodes.filter(n => n.supersededBy === d.id);

                if (children.length || chain.length) {
                    chainHtml = `<div class="accuracy-chain"><div class="chain-label">Generalization chain:</div>`;
                    children.forEach(child => {
                        const csi = (typeof STATUS_INFO !== 'undefined' && child.status) ? STATUS_INFO[child.status] : null;
                        chainHtml += `
                            <div class="chain-item chain-approx" data-id="${child.id}">
                                <span class="chain-level" style="color:${csi ? csi.color : '#FF8C00'}">${csi ? csi.badge : '≈'}</span>
                                <span class="chain-name">${child.name}</span>
                                <span class="chain-arrow">→</span>
                            </div>`;
                    });
                    chainHtml += `
                        <div class="chain-item chain-current">
                            <span class="chain-level" style="color:${si ? si.color : '#8892a4'}">${si ? si.badge : '●'}</span>
                            <span class="chain-name"><strong>${d.name}</strong></span>
                            ${chain.length ? '<span class="chain-arrow">→</span>' : ''}
                        </div>`;
                    chain.forEach((parent, i) => {
                        const psi = (typeof STATUS_INFO !== 'undefined' && parent.status) ? STATUS_INFO[parent.status] : null;
                        chainHtml += `
                            <div class="chain-item chain-exact" data-id="${parent.id}">
                                <span class="chain-level" style="color:${psi ? psi.color : '#00e676'}">${psi ? psi.badge : '●'}</span>
                                <span class="chain-name">${parent.name}</span>
                                ${(i < chain.length - 1) ? '<span class="chain-arrow">→</span>' : ''}
                            </div>`;
                    });
                    chainHtml += `</div>`;
                }

                // DerivesFrom section
                let derivesHtml = "";
                if (d.derivesFromData && d.derivesFromData.length) {
                    derivesHtml = `<div class="derives-section"><div class="chain-label">Derives from:</div>`;
                    d.derivesFromData.forEach(df => {
                        const parent = allNodes.find(n => n.id === df.eq);
                        if (parent) {
                            derivesHtml += `
                                <div class="derives-item" data-id="${parent.id}">
                                    <span class="derives-arrow">←</span>
                                    <span class="derives-name">${parent.name}</span>
                                    ${df.assuming ? `<span class="derives-assuming">assuming ${df.assuming}</span>` : ''}
                                </div>`;
                        }
                    });
                    derivesHtml += `</div>`;
                }

                // ComponentOf section
                let compHtml = "";
                if (d.componentOf) {
                    const systemLabel = d.componentOf.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                    const siblings = allNodes.filter(n => n.componentOf === d.componentOf && n.id !== d.id);
                    compHtml = `
                        <div class="component-section">
                            <div class="chain-label">Part of: <strong>${systemLabel}</strong></div>
                            ${siblings.length ? `<div class="component-siblings">${siblings.map(s =>
                                `<span class="chip chip-system" data-id="${s.id}">${s.name}</span>`
                            ).join('')}</div>` : ''}
                        </div>`;
                }

                taxonomyHtml = `
                    <div class="detail-section detail-accuracy">
                        <h3>🔬 Status & Taxonomy</h3>
                        <div class="accuracy-level" style="color:${si ? si.color : '#8892a4'}">
                            ${si ? si.badge + ' ' + si.label : '● Derived'}
                        </div>
                        ${d.conditions ? '<div class="accuracy-domain">Valid when: <span id="detail-domain-tex"></span></div>' : ''}
                        ${d.statusNote ? `<div class="status-note">${d.statusNote}</div>` : ''}
                        ${chainHtml}
                        ${derivesHtml}
                        ${compHtml}
                    </div>`;
            }

            // Build alternative forms section
            let formsHtml = "";
            if (d.forms) {
                const formLabels = {
                    differential: "∂ Differential",
                    integral: "∫ Integral",
                    vector: "→ Vector",
                    tensor: "⊗ Tensor"
                };
                let formItems = "";
                Object.entries(d.forms).forEach(([key, tex]) => {
                    const note = d.formNotes && d.formNotes[key] ? d.formNotes[key] : "";
                    formItems += `
                        <div class="form-item">
                            <div class="form-label">${formLabels[key] || key}</div>
                            <div class="form-equation" id="detail-form-${key}"></div>
                            ${note ? `<div class="form-note">${note}</div>` : ''}
                        </div>`;
                });
                formsHtml = `
                    <div class="detail-section detail-forms">
                        <h3>📝 Alternative Forms</h3>
                        ${formItems}
                    </div>`;
            }

            html = `
                <div class="detail-header" style="border-left: 3px solid ${fieldInfo.color || '#999'}">
                    <span class="detail-field" style="color:${fieldInfo.color || '#fff'}">
                        ${fieldInfo.icon || ''} ${fieldInfo.name || d.field}
                    </span>
                    <h2>${d.name}</h2>
                    <div class="detail-equation" id="detail-eq-render"></div>
                </div>
                <p class="detail-desc">${d.description}</p>
                <div class="detail-meta">
                    ${d.discoverer ? '<span>👤 ' + d.discoverer + '</span>' : ''}
                    ${d.year ? '<span>📅 ' + (d.year < 0 ? Math.abs(d.year) + ' BCE' : d.year) + '</span>' : ''}
                </div>
                ${taxonomyHtml}
                ${formsHtml}
                ${connectedConsts.length ? `
                    <div class="detail-section">
                        <h3>◆ Fundamental Constants Used</h3>
                        <div class="detail-chips">
                            ${connectedConsts.map(c => `
                                <span class="chip chip-constant" data-id="${c.id}" title="${c.value} ${c.unit}">
                                    ${c.symbol} <small>${c.name}</small>
                                </span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                ${connectedVars.length ? `
                    <div class="detail-section">
                        <h3>○ Variables</h3>
                        <div class="detail-chips">
                            ${connectedVars.map(v => `
                                <span class="chip chip-variable" data-id="${v.id}">
                                    ${v.symbol} <small>${v.name}</small>
                                </span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                ${relatedEqs.length ? `
                    <div class="detail-section">
                        <h3>🔗 Related Equations</h3>
                        <div class="detail-related">
                            ${relatedEqs.map(eq => {
                                const shared = (eq.uses || []).filter(u => (d.uses || []).includes(u));
                                const sharedNames = shared.map(s => {
                                    const n = allNodes.find(x => x.id === s) || VARIABLES.find(x => x.id === s) || CONSTANTS.find(x => x.id === s);
                                    return n ? n.symbol : s;
                                }).join(', ');
                                const eqField = FIELD_INFO[eq.field] || {};
                                return `
                                    <div class="related-eq" data-id="${eq.id}">
                                        <span class="related-dot" style="background:${eqField.color || '#999'}"></span>
                                        <span class="related-name">${eq.name}</span>
                                        <span class="related-shared">via ${sharedNames}</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                ` : ''}
            `;
        } else if (d.type === "constant") {
            // Find all equations using this constant
            const eqsUsing = allNodes.filter(n => n.type === "equation" && (n.uses || []).includes(d.id));
            const fieldGroups = {};
            eqsUsing.forEach(eq => {
                const f = eq.field || "other";
                if (!fieldGroups[f]) fieldGroups[f] = [];
                fieldGroups[f].push(eq);
            });

            html = `
                <div class="detail-header" style="border-left: 3px solid #FFD700">
                    <span class="detail-field" style="color:#FFD700">◆ Fundamental Constant</span>
                    <h2>${d.name}</h2>
                    <div class="detail-constant-value">
                        <span class="const-symbol">${d.symbol}</span> = ${d.value} ${d.unit}
                    </div>
                </div>
                <p class="detail-desc">${d.description}</p>
                <div class="detail-section">
                    <h3>Appears in ${eqsUsing.length} equations across ${Object.keys(fieldGroups).length} fields</h3>
                    ${Object.entries(fieldGroups).map(([field, eqs]) => {
                        const fi = FIELD_INFO[field] || {};
                        return `
                            <div class="detail-field-group">
                                <h4 style="color:${fi.color || '#999'}">${fi.icon || ''} ${fi.name || field}</h4>
                                ${eqs.map(eq => `
                                    <div class="related-eq" data-id="${eq.id}">
                                        <span class="related-dot" style="background:${fi.color || '#999'}"></span>
                                        <span class="related-name">${eq.name}</span>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            // Variable
            const eqsUsing = allNodes.filter(n => n.type === "equation" && (n.uses || []).includes(d.id));
            const fieldGroups = {};
            eqsUsing.forEach(eq => {
                const f = eq.field || "other";
                if (!fieldGroups[f]) fieldGroups[f] = [];
                fieldGroups[f].push(eq);
            });

            html = `
                <div class="detail-header" style="border-left: 3px solid #778899">
                    <span class="detail-field" style="color:#778899">○ Physical Variable</span>
                    <h2>${d.name}</h2>
                    <div class="detail-constant-value">
                        <span class="const-symbol">${d.symbol}</span> ${d.unit ? '[' + d.unit + ']' : ''}
                    </div>
                </div>
                <div class="detail-section">
                    <h3>Appears in ${eqsUsing.length} equations across ${Object.keys(fieldGroups).length} fields</h3>
                    ${Object.entries(fieldGroups).map(([field, eqs]) => {
                        const fi = FIELD_INFO[field] || {};
                        return `
                            <div class="detail-field-group">
                                <h4 style="color:${fi.color || '#999'}">${fi.icon || ''} ${fi.name || field}</h4>
                                ${eqs.map(eq => `
                                    <div class="related-eq" data-id="${eq.id}">
                                        <span class="related-dot" style="background:${fi.color || '#999'}"></span>
                                        <span class="related-name">${eq.name}</span>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }

        content.innerHTML = html;
        panel.classList.remove("hidden");

        // Render equation
        if (d.type === "equation" && d.equation) {
            const eqEl = document.getElementById("detail-eq-render");
            if (eqEl) {
                try {
                    katex.render(d.equation, eqEl, { throwOnError: false, displayMode: true });
                } catch (e) {
                    eqEl.textContent = d.equation;
                }
            }

            // Render conditions TeX
            const domainEl = document.getElementById("detail-domain-tex");
            if (domainEl && d.conditions) {
                try {
                    katex.render(d.conditions, domainEl, { throwOnError: false, displayMode: false });
                } catch(e) {
                    domainEl.textContent = d.conditions;
                }
            }

            // Render alternative form equations
            if (d.forms) {
                Object.entries(d.forms).forEach(([key, tex]) => {
                    const formEl = document.getElementById(`detail-form-${key}`);
                    if (formEl) {
                        try {
                            katex.render(tex, formEl, { throwOnError: false, displayMode: true });
                        } catch(e) {
                            formEl.textContent = tex;
                        }
                    }
                });
            }
        }

        // Chip and related equation click handlers
        content.querySelectorAll("[data-id]").forEach(el => {
            el.addEventListener("click", () => {
                const targetNode = allNodes.find(n => n.id === el.dataset.id);
                if (targetNode) {
                    if (selectedNode) highlightConnections(selectedNode, false);
                    selectedNode = targetNode;
                    highlightConnections(targetNode, true);
                    showDetailPanel(targetNode);
                }
            });
            el.style.cursor = "pointer";
        });
    }

    function hideDetailPanel() {
        document.getElementById("detail-panel").classList.add("hidden");
    }

    // ─── Setup Controls ──────────────────────────────────────────────────────

    function setupControls() {
        // Field filters
        const filterContainer = document.getElementById("field-filters");
        Object.entries(FIELD_INFO).forEach(([key, info]) => {
            const count = EQUATIONS.filter(eq => eq.field === key).length;
            const label = document.createElement("label");
            label.className = "filter-label";
            label.innerHTML = `
                <input type="checkbox" value="${key}" checked>
                <span class="filter-swatch" style="background:${info.color}"></span>
                <span class="filter-name">${info.icon} ${info.name}</span>
                <span class="filter-count">${count}</span>
            `;
            label.querySelector("input").addEventListener("change", (e) => {
                if (e.target.checked) activeFilters.add(key);
                else activeFilters.delete(key);
                updateVisibility();
            });
            filterContainer.appendChild(label);
        });

        // View mode
        document.querySelectorAll('input[name="view"]').forEach(radio => {
            radio.addEventListener("change", (e) => {
                viewMode = e.target.value;
                updateVisibility();
            });
        });

        // Node sizing
        document.querySelectorAll('input[name="sizing"]').forEach(radio => {
            radio.addEventListener("change", (e) => {
                sizingMode = e.target.value;
                // Update node sizes
                nodeGroup.selectAll("g.node").each(function (d) {
                    const el = d3.select(this);
                    const r = nodeRadius(d);
                    if (d.type === "constant") {
                        el.select("polygon").attr("points", `0,${-r} ${r},0 0,${r} ${-r},0`);
                    } else {
                        el.select("circle").attr("r", r);
                    }
                });
                simulation.force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 2));
                simulation.alpha(0.3).restart();
            });
        });

        // Search
        const searchInput = document.getElementById("search");
        let searchTimeout;
        searchInput.addEventListener("input", (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                searchTerm = e.target.value.trim();
                updateVisibility();
            }, 200);
        });

        // Close detail panel
        document.getElementById("close-detail").addEventListener("click", () => {
            deselectAll();
        });

        // Top constants list
        buildTopConstants();

        // Data quality panel
        buildDataQualityPanel();

        // Select all / deselect all buttons
        document.getElementById("select-all-fields").addEventListener("click", () => {
            activeFilters = new Set(Object.keys(FIELD_INFO));
            document.querySelectorAll('#field-filters input[type="checkbox"]').forEach(cb => cb.checked = true);
            updateVisibility();
        });
        document.getElementById("deselect-all-fields").addEventListener("click", () => {
            activeFilters.clear();
            document.querySelectorAll('#field-filters input[type="checkbox"]').forEach(cb => cb.checked = false);
            updateVisibility();
        });

        // Equation form mode
        document.querySelectorAll('input[name="form-mode"]').forEach(radio => {
            radio.addEventListener("change", (e) => {
                formMode = e.target.value;
                updateFormDisplay();
            });
        });

        // Accuracy indicators toggle
        const accuracyToggle = document.getElementById("toggle-accuracy");
        if (accuracyToggle) {
            accuracyToggle.addEventListener("change", (e) => {
                showAccuracy = e.target.checked;
                nodeGroup.selectAll(".accuracy-ring, .approx-badge")
                    .style("display", showAccuracy ? null : "none");
            });
        }

        // Supersession links toggle
        const superToggle = document.getElementById("toggle-supersession");
        if (superToggle) {
            superToggle.addEventListener("change", (e) => {
                showSupersession = e.target.checked;
                linkGroup.selectAll(".link-generalizes, .link-equivalent, .link-incompatible")
                    .style("display", showSupersession ? null : "none");
            });
        }

        // Variable nodes toggle
        const varToggle = document.getElementById("toggle-variables");
        if (varToggle) {
            varToggle.addEventListener("change", (e) => {
                showVariables = e.target.checked;
                updateVisibility();
            });
        }

        // Labels toggle
        const labelToggle = document.getElementById("toggle-labels");
        if (labelToggle) {
            labelToggle.addEventListener("change", (e) => {
                showLabels = e.target.checked;
                updateLabelVisibility();
            });
        }

        // Curved edges toggle
        const curveToggle = document.getElementById("toggle-curves");
        if (curveToggle) {
            curveToggle.addEventListener("change", (e) => {
                showCurves = e.target.checked;
                // Redraw all paths immediately
                linkGroup.selectAll("path.link").attr("d", linkPath);
            });
        }

        // Hierarchical layout toggle
        const hierToggle = document.getElementById("toggle-hierarchy");
        if (hierToggle) {
            hierToggle.addEventListener("change", (e) => {
                showHierarchy = e.target.checked;
                if (showHierarchy) {
                    simulation.force("derivationY", d3.forceY(d => {
                        if (d.derivationDepth >= 0) {
                            return height * 0.12 + (d.derivationDepth / maxDerivDepth) * height * 0.76;
                        }
                        return height / 2;
                    }).strength(d => d.derivationDepth >= 0 ? 0.025 : 0));
                } else {
                    simulation.force("derivationY", null);
                }
                simulation.alpha(0.5).restart();
            });
        }

        // Connection mode (variables vs direct)
        document.querySelectorAll('input[name="conn-mode"]').forEach(radio => {
            radio.addEventListener("change", (e) => {
                connectionMode = e.target.value;
                deselectAll();
                rebuildGraphMode();
            });
        });

        // ─── Simulation Physics Sliders ──────────────────────────────────

        function bindSimSlider(id, stateKey, valId, formatter) {
            const el = document.getElementById(id);
            const valEl = document.getElementById(valId);
            if (!el) return;
            el.addEventListener("input", () => {
                const v = parseFloat(el.value);
                if (stateKey === "simRepulsion") simRepulsion = v;
                else if (stateKey === "simLinkDist") simLinkDist = v;
                else if (stateKey === "simLinkStrength") simLinkStrength = v;
                else if (stateKey === "simGravity") simGravity = v;
                else if (stateKey === "simDecay") simDecay = v;
                if (valEl) valEl.textContent = formatter(v);
                applySimulationPhysics(true);
            });
        }

        bindSimSlider("sim-repulsion", "simRepulsion", "sim-repulsion-val", v => v.toFixed(1) + "×");
        bindSimSlider("sim-link-dist", "simLinkDist", "sim-link-dist-val", v => v.toFixed(1) + "×");
        bindSimSlider("sim-link-str", "simLinkStrength", "sim-link-str-val", v => v.toFixed(1) + "×");
        bindSimSlider("sim-gravity", "simGravity", "sim-gravity-val", v => v.toFixed(3));
        bindSimSlider("sim-decay", "simDecay", "sim-decay-val", v => v.toFixed(2));

        const simResetBtn = document.getElementById("sim-reset");
        if (simResetBtn) {
            simResetBtn.addEventListener("click", () => {
                simRepulsion = 1.0; simLinkDist = 1.0; simLinkStrength = 1.0;
                simGravity = 0.015; simDecay = 0.35;
                document.getElementById("sim-repulsion").value = 1;
                document.getElementById("sim-link-dist").value = 1;
                document.getElementById("sim-link-str").value = 1;
                document.getElementById("sim-gravity").value = 0.015;
                document.getElementById("sim-decay").value = 0.35;
                document.getElementById("sim-repulsion-val").textContent = "1.0×";
                document.getElementById("sim-link-dist-val").textContent = "1.0×";
                document.getElementById("sim-link-str-val").textContent = "1.0×";
                document.getElementById("sim-gravity-val").textContent = "0.015";
                document.getElementById("sim-decay-val").textContent = "0.35";
                applySimulationPhysics(true);
            });
        }

        // ─── Timeline Slider ─────────────────────────────────────────────
        const slider = document.getElementById("timeline-slider");
        const yearLabel = document.getElementById("timeline-year");
        const resetBtn = document.getElementById("timeline-reset");
        const playBtn = document.getElementById("timeline-play");

        if (slider) {
            slider.addEventListener("input", (e) => {
                timelineYear = parseInt(e.target.value);
                yearLabel.textContent = timelineYear >= 2030 ? "All Time"
                    : (timelineYear < 0 ? Math.abs(timelineYear) + " BCE" : timelineYear);
                updateVisibility();
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                stopTimeline();
                timelineYear = 2030;
                slider.value = 2030;
                yearLabel.textContent = "All Time";
                updateVisibility();
            });
        }

        if (playBtn) {
            playBtn.addEventListener("click", () => {
                if (timelinePlaying) {
                    stopTimeline();
                } else {
                    startTimeline();
                }
            });
        }
    }

    // ─── Top Constants Panel ─────────────────────────────────────────────────

    function buildTopConstants() {
        const container = document.getElementById("top-constants");
        const sortedConstants = [...CONSTANTS]
            .map(c => {
                const node = allNodes.find(n => n.id === c.id);
                return { ...c, connections: node ? node.connections : 0 };
            })
            .sort((a, b) => b.connections - a.connections);

        sortedConstants.forEach(c => {
            const div = document.createElement("div");
            div.className = "top-constant-item";
            div.innerHTML = `
                <span class="tc-symbol">${c.symbol}</span>
                <span class="tc-name">${c.name}</span>
                <span class="tc-count">${c.connections}</span>
            `;
            div.addEventListener("click", () => {
                const node = allNodes.find(n => n.id === c.id);
                if (node) {
                    if (selectedNode) highlightConnections(selectedNode, false);
                    selectedNode = node;
                    highlightConnections(node, true);
                    showDetailPanel(node);
                }
            });
            container.appendChild(div);
        });
    }

    // ─── Data Quality Audit ──────────────────────────────────────────────────

    function buildDataQualityPanel() {
        const container = document.getElementById("data-quality");
        if (!container) return;

        const eqs = allNodes.filter(n => n.type === "equation");
        const issues = {
            missingYear: [],
            missingDiscoverer: [],
            missingConditions: [],
            missingDerivesFrom: [],
            missingForms: [],
            missingDescription: []
        };

        eqs.forEach(eq => {
            if (!eq.year) issues.missingYear.push(eq);
            if (!eq.discoverer) issues.missingDiscoverer.push(eq);
            if (!eq.conditions && eq.status !== "definition" && eq.status !== "postulate") issues.missingConditions.push(eq);
            if (!eq.derivesFromData && !eq.supersededBy && eq.status !== "postulate" && eq.status !== "definition" && eq.status !== "empirical") issues.missingDerivesFrom.push(eq);
            if (!eq.forms) issues.missingForms.push(eq);
            if (!eq.description || eq.description.length < 10) issues.missingDescription.push(eq);
        });

        const totalIssues = Object.values(issues).reduce((sum, arr) => sum + arr.length, 0);
        const completeness = Math.round((1 - totalIssues / (eqs.length * 6)) * 100);

        const scoreColor = completeness > 80 ? '#00e676' : completeness > 60 ? '#F1C40F' : '#E74C3C';
        let html = `<div class="dq-score" style="color:${scoreColor}">${completeness}% complete</div>`;

        const categories = [
            { key: "missingYear", label: "Missing year", icon: "\u{1F4C5}" },
            { key: "missingDiscoverer", label: "Missing discoverer", icon: "\u{1F464}" },
            { key: "missingConditions", label: "Missing conditions", icon: "\u{1F4D0}" },
            { key: "missingDerivesFrom", label: "No derivation chain", icon: "\u{1F517}" },
            { key: "missingForms", label: "No alt. forms", icon: "\u2202" },
            { key: "missingDescription", label: "Weak description", icon: "\u{1F4DD}" }
        ];

        categories.forEach(cat => {
            const arr = issues[cat.key];
            if (arr.length === 0) return;
            const item = document.createElement("div");
            item.className = "dq-item";
            item.innerHTML = `
                <span class="dq-icon">${cat.icon}</span>
                <span class="dq-label">${cat.label}</span>
                <span class="dq-count">${arr.length}</span>
            `;
            item.title = arr.slice(0, 10).map(e => e.name).join(", ") + (arr.length > 10 ? "\u2026" : "");
            item.addEventListener("click", () => {
                const ids = new Set(arr.map(e => e.id));
                highlightDataQualityIssues(ids);
            });
            container.appendChild(item);
        });

        // Insert score at top
        container.insertAdjacentHTML("afterbegin", html);

        // Mark nodes with incomplete data in the graph
        const incompleteIds = new Set();
        Object.values(issues).forEach(arr => arr.forEach(eq => incompleteIds.add(eq.id)));
        allNodes.forEach(n => { n.hasDataIssues = incompleteIds.has(n.id); });
    }

    function highlightDataQualityIssues(ids) {
        // Dim everything, highlight only problem equations
        nodeGroup.selectAll("g.node")
            .style("opacity", n => ids.has(n.id) ? 1 : 0.06);
        linkGroup.selectAll("path.link")
            .style("opacity", 0.02);
    }

    // ─── Timeline Animation ─────────────────────────────────────────────────

    function startTimeline() {
        const slider = document.getElementById("timeline-slider");
        const yearLabel = document.getElementById("timeline-year");
        const playBtn = document.getElementById("timeline-play");
        if (!slider) return;

        timelinePlaying = true;
        playBtn.textContent = "⏸ Pause";

        // Start from 300 BCE if at the end
        if (timelineYear >= 2025) {
            timelineYear = -300;
            slider.value = -300;
        }

        timelineInterval = setInterval(() => {
            // Adaptive step: bigger jumps in early history, smaller near present
            const step = timelineYear < 1500 ? 25 : timelineYear < 1800 ? 10 : 5;
            timelineYear += step;

            if (timelineYear >= 2030) {
                timelineYear = 2030;
                stopTimeline();
            }

            slider.value = timelineYear;
            yearLabel.textContent = timelineYear >= 2030 ? "All Time"
                : (timelineYear < 0 ? Math.abs(timelineYear) + " BCE" : timelineYear);
            updateVisibility();
        }, 300);
    }

    function stopTimeline() {
        timelinePlaying = false;
        const playBtn = document.getElementById("timeline-play");
        if (playBtn) playBtn.textContent = "▶ Play";
        if (timelineInterval) {
            clearInterval(timelineInterval);
            timelineInterval = null;
        }
    }

    // ─── Minimap ─────────────────────────────────────────────────────────────

    let minimapSvg, minimapViewport, minimapScale;
    let mainZoom;  // store the main zoom behavior for minimap interaction

    function initMinimap() {
        const mmContainer = document.getElementById("minimap");
        if (!mmContainer) return;
        minimapSvg = d3.select("#minimap-svg");
        // We'll update minimap dots on each tick
    }

    function updateMinimap() {
        if (!minimapSvg) return;
        // Calculate bounds of all nodes
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        allNodes.forEach(n => {
            if (n.x == null) return;
            if (n.x < minX) minX = n.x;
            if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.y > maxY) maxY = n.y;
        });

        if (!isFinite(minX)) return;

        const pad = 50;
        const bw = maxX - minX + pad * 2;
        const bh = maxY - minY + pad * 2;

        const mmW = 160, mmH = 120;
        minimapScale = Math.min(mmW / bw, mmH / bh);
        const offX = (mmW - bw * minimapScale) / 2 - (minX - pad) * minimapScale;
        const offY = (mmH - bh * minimapScale) / 2 - (minY - pad) * minimapScale;

        // Clear and redraw
        minimapSvg.selectAll("*").remove();

        const mg = minimapSvg.append("g")
            .attr("transform", `translate(${offX},${offY}) scale(${minimapScale})`);

        // Draw tiny dots for visible nodes
        allNodes.filter(n => n.x != null && isNodeVisible(n)).forEach(n => {
            mg.append("circle")
                .attr("cx", n.x)
                .attr("cy", n.y)
                .attr("r", 2 / minimapScale)
                .attr("fill", nodeColor(n))
                .attr("opacity", 0.6);
        });

        // Draw viewport rectangle
        if (mainZoom) {
            const transform = d3.zoomTransform(svg.node());
            const vx = (-transform.x) / transform.k;
            const vy = (-transform.y) / transform.k;
            const vw = width / transform.k;
            const vh = height / transform.k;

            mg.append("rect")
                .attr("class", "minimap-viewport")
                .attr("x", vx)
                .attr("y", vy)
                .attr("width", vw)
                .attr("height", vh);
        }
    }

    // ─── Handle Resize ───────────────────────────────────────────────────────

    function handleResize() {
        const container = document.getElementById("graph-container");
        width = container.clientWidth;
        height = container.clientHeight;
        svg.attr("width", width).attr("height", height);
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
        simulation.force("x", d3.forceX(width / 2).strength(0.03));
        simulation.force("y", d3.forceY(height / 2).strength(0.03));
        simulation.alpha(0.3).restart();
    }

    // ─── Initialize ──────────────────────────────────────────────────────────

    function init() {
        initGraph();
        setupControls();
        initMinimap();
        window.addEventListener("resize", handleResize);

        // Keyboard shortcuts
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                stopTimeline();
                deselectAll();
                document.getElementById("search").value = "";
                searchTerm = "";
                updateVisibility();
            }
            if (e.key === "/" && e.target.tagName !== "INPUT") {
                e.preventDefault();
                document.getElementById("search").focus();
            }
            if (e.key === " " && e.target.tagName !== "INPUT") {
                e.preventDefault();
                if (timelinePlaying) stopTimeline();
                else startTimeline();
            }
        });

        // Initial minimap after layout settles
        setTimeout(updateMinimap, 4000);
    }

    // Start
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

})();
