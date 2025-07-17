import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';

// Audio Analyzer Hook
function useAudioAnalyzer(audioRef) {
  const [frequencyData, setFrequencyData] = useState([]);
  const analyzerRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const animationRef = useRef(null);
  
  useEffect(() => {
    if (!audioRef.current) return;
    
    const setupAudio = () => {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const audioContext = audioContextRef.current;
      
      analyzerRef.current = audioContext.createAnalyser();
      analyzerRef.current.fftSize = 2048;
      analyzerRef.current.smoothingTimeConstant = 0.8;
      
      if (!sourceRef.current) {
        sourceRef.current = audioContext.createMediaElementSource(audioRef.current);
        sourceRef.current.connect(analyzerRef.current);
        analyzerRef.current.connect(audioContext.destination);
      }
    };
    
    const analyze = () => {
      if (!analyzerRef.current) return;
      
      const bufferLength = analyzerRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyzerRef.current.getByteFrequencyData(dataArray);
      
      const normalizedData = Array.from(dataArray).map(value => value / 255);
      setFrequencyData(normalizedData);
      
      animationRef.current = requestAnimationFrame(analyze);
    };
    
    const handlePlay = () => {
      if (!audioContextRef.current) setupAudio();
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
      analyze();
    };
    
    const handlePause = () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
    
    audioRef.current.addEventListener('play', handlePlay);
    audioRef.current.addEventListener('pause', handlePause);
    
    return () => {
      if (audioRef.current) {
        audioRef.current.removeEventListener('play', handlePlay);
        audioRef.current.removeEventListener('pause', handlePause);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [audioRef]);
  
  return frequencyData;
}

// Visualization styles
const STYLES = {
  STANDARD: 'standard',
  CORNER: 'corner',
  CENTER: 'center',
  RANDOM: 'random'
};

// Visualization modes
const MODES = {
  POINTS: 'points',
  NURBS: 'nurbs'
};

// NURBS Surface Visualizer
function NURBSVisualizer({ frequencyData, samples, style, mountRef }) {
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const meshRef = useRef(null);
  const frameRef = useRef(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  
  useEffect(() => {
    if (!mountRef.current) return;
    
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0.5, 1);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);
    
    // Create NURBS surface geometry
    const gridSize = Math.floor(Math.sqrt(samples));
    const geometry = new THREE.PlaneGeometry(1, 1, gridSize - 1, gridSize - 1);
    
    // Create gradient material
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      wireframe: false
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    scene.add(mesh);
    meshRef.current = mesh;
    
    // Mouse controls
    let isMouseDown = false;
    const startMouse = { x: 0, y: 0 };
    const startRotation = { x: 0, y: 0 };
    
    const handleMouseDown = (event) => {
      isMouseDown = true;
      startMouse.x = event.clientX;
      startMouse.y = event.clientY;
      startRotation.x = rotationRef.current.x;
      startRotation.y = rotationRef.current.y;
      mountRef.current.style.cursor = 'grabbing';
    };
    
    const handleMouseMove = (event) => {
      if (!isMouseDown) return;
      
      const deltaX = (event.clientX - startMouse.x) / width;
      const deltaY = (event.clientY - startMouse.y) / height;
      
      rotationRef.current.y = startRotation.y + deltaX * 2;
      rotationRef.current.x = startRotation.x + deltaY * 2;
    };
    
    const handleMouseUp = () => {
      isMouseDown = false;
      mountRef.current.style.cursor = 'grab';
    };
    
    mountRef.current.style.cursor = 'grab';
    mountRef.current.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    
    // Handle resize
    const handleResize = () => {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      
      // Rotate camera
      camera.position.x = Math.sin(rotationRef.current.y) * 1.5;
      camera.position.y = 0.5 + rotationRef.current.x * 0.5;
      camera.position.z = Math.cos(rotationRef.current.y) * 1.5;
      camera.lookAt(0, 0, 0);
      
      // Update mesh vertices based on frequency data
      if (meshRef.current && frequencyData.length > 0) {
        const geometry = meshRef.current.geometry;
        const vertices = geometry.attributes.position.array;
        const colors = [];
        
        // Apply style-based indexing
        const indices = [];
        for (let i = 0; i < gridSize * gridSize; i++) {
          indices.push(i);
        }
        
        if (style === STYLES.CORNER) {
          indices.sort((a, b) => {
            const ax = (a % gridSize) / gridSize - 0.5;
            const az = Math.floor(a / gridSize) / gridSize - 0.5;
            const bx = (b % gridSize) / gridSize - 0.5;
            const bz = Math.floor(b / gridSize) / gridSize - 0.5;
            return (Math.abs(ax) + Math.abs(az)) - (Math.abs(bx) + Math.abs(bz));
          });
        } else if (style === STYLES.CENTER) {
          indices.sort((a, b) => {
            const ax = (a % gridSize) / gridSize - 0.5;
            const az = Math.floor(a / gridSize) / gridSize - 0.5;
            const bx = (b % gridSize) / gridSize - 0.5;
            const bz = Math.floor(b / gridSize) / gridSize - 0.5;
            return Math.sqrt(ax*ax + az*az) - Math.sqrt(bx*bx + bz*bz);
          });
        }
        
        // Update vertices
        for (let i = 0; i < vertices.length / 3; i++) {
          const dataIndex = Math.floor((indices[i] / indices.length) * frequencyData.length);
          const amplitude = frequencyData[dataIndex] || 0;
          
          // Smooth the height with neighbors for NURBS-like effect
          let smoothedAmplitude = amplitude;
          if (i > 0 && i < vertices.length / 3 - 1) {
            const prevIndex = Math.floor((indices[i-1] / indices.length) * frequencyData.length);
            const nextIndex = Math.floor((indices[i+1] / indices.length) * frequencyData.length);
            const prevAmp = frequencyData[prevIndex] || 0;
            const nextAmp = frequencyData[nextIndex] || 0;
            smoothedAmplitude = (prevAmp + amplitude * 2 + nextAmp) / 4;
          }
          
          // Update Y position (remember, plane is rotated so Y becomes Z)
          const currentY = vertices[i * 3 + 2];
          const targetY = smoothedAmplitude * 0.5;
          vertices[i * 3 + 2] += (targetY - currentY) * 0.1;
          
          // Calculate color based on height
          const hue = smoothedAmplitude * 0.7;
          const color = new THREE.Color();
          color.setHSL(hue, 1, 0.5 + smoothedAmplitude * 0.3);
          colors.push(color.r, color.g, color.b);
        }
        
        // Update geometry
        geometry.attributes.position.needsUpdate = true;
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals();
      }
      
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Cleanup
    return () => {
      mountRef.current?.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('resize', handleResize);
      
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      
      if (meshRef.current) {
        meshRef.current.geometry.dispose();
        meshRef.current.material.dispose();
      }
      
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [samples, style]);
  
  return null;
}

// Points Visualizer (original)
function PointsVisualizer({ frequencyData, samples, style, mountRef }) {
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const meshesRef = useRef([]);
  const frameRef = useRef(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  
  useEffect(() => {
    if (!mountRef.current) return;
    
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0.5, 1);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);
    
    // Create grid of cubes
    const gridSize = Math.floor(Math.sqrt(samples));
    const geometry = new THREE.BoxGeometry(0.02, 0.02, 0.02);
    const meshes = [];
    
    // Generate positions based on style
    const positions = [];
    const offset = -0.5;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = (i / (gridSize - 1)) + offset;
        const z = (j / (gridSize - 1)) + offset;
        positions.push({ x, y: 0, z, index: i * gridSize + j });
      }
    }
    
    // Apply style sorting
    if (style === STYLES.CORNER) {
      positions.sort((a, b) => {
        const distA = Math.abs(a.x) + Math.abs(a.z);
        const distB = Math.abs(b.x) + Math.abs(b.z);
        return distA - distB;
      });
    } else if (style === STYLES.CENTER) {
      positions.sort((a, b) => {
        const distA = Math.sqrt(a.x ** 2 + a.z ** 2);
        const distB = Math.sqrt(b.x ** 2 + b.z ** 2);
        return distA - distB;
      });
    } else if (style === STYLES.RANDOM) {
      for (let i = positions.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [positions[i], positions[j]] = [positions[j], positions[i]];
      }
    }
    
    // Create meshes
    positions.forEach((pos, i) => {
      const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(pos.x, pos.y, pos.z);
      mesh.userData = { baseY: pos.y, index: i };
      scene.add(mesh);
      meshes.push(mesh);
    });
    
    meshesRef.current = meshes;
    
    // Mouse controls
    let isMouseDown = false;
    const startMouse = { x: 0, y: 0 };
    const startRotation = { x: 0, y: 0 };
    
    const handleMouseDown = (event) => {
      isMouseDown = true;
      startMouse.x = event.clientX;
      startMouse.y = event.clientY;
      startRotation.x = rotationRef.current.x;
      startRotation.y = rotationRef.current.y;
      mountRef.current.style.cursor = 'grabbing';
    };
    
    const handleMouseMove = (event) => {
      if (!isMouseDown) return;
      
      const deltaX = (event.clientX - startMouse.x) / width;
      const deltaY = (event.clientY - startMouse.y) / height;
      
      rotationRef.current.y = startRotation.y + deltaX * 2;
      rotationRef.current.x = startRotation.x + deltaY * 2;
    };
    
    const handleMouseUp = () => {
      isMouseDown = false;
      mountRef.current.style.cursor = 'grab';
    };
    
    mountRef.current.style.cursor = 'grab';
    mountRef.current.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    
    // Handle resize
    const handleResize = () => {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      
      // Rotate camera
      camera.position.x = Math.sin(rotationRef.current.y) * 1.5;
      camera.position.y = 0.5 + rotationRef.current.x * 0.5;
      camera.position.z = Math.cos(rotationRef.current.y) * 1.5;
      camera.lookAt(0, 0, 0);
      
      // Update cube positions and colors based on frequency data
      meshesRef.current.forEach((mesh, i) => {
        const dataIndex = Math.floor((i / meshesRef.current.length) * frequencyData.length);
        const amplitude = frequencyData[dataIndex] || 0;
        
        // Smooth position update
        const targetY = mesh.userData.baseY + amplitude * 5;
        mesh.position.y += (targetY - mesh.position.y) * 0.1;
        
        // Update color based on amplitude
        const hue = amplitude * 0.7;
        mesh.material.color.setHSL(hue, 1, 0.5);
      });
      
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Cleanup
    return () => {
      mountRef.current?.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('resize', handleResize);
      
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      
      meshesRef.current.forEach(mesh => {
        mesh.geometry.dispose();
        mesh.material.dispose();
      });
      
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [samples, style]);
  
  return null;
}

// Main 3D Visualizer Component
function ThreeVisualizer({ frequencyData, samples, style, mode }) {
  const mountRef = useRef(null);
  
  return (
    <div ref={mountRef} className="w-full h-full">
      {mode === MODES.NURBS ? (
        <NURBSVisualizer 
          frequencyData={frequencyData} 
          samples={samples} 
          style={style}
          mountRef={mountRef}
        />
      ) : (
        <PointsVisualizer 
          frequencyData={frequencyData} 
          samples={samples} 
          style={style}
          mountRef={mountRef}
        />
      )}
    </div>
  );
}

// Main App Component
export default function AudioVisualizer3D() {
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  const [samples, setSamples] = useState(1024);
  const [style, setStyle] = useState(STYLES.STANDARD);
  const [mode, setMode] = useState(MODES.POINTS);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const frequencyData = useAudioAnalyzer(audioRef);
  
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && audioRef.current) {
      const url = URL.createObjectURL(file);
      audioRef.current.src = url;
      setIsPlaying(false);
    }
  };
  
  const togglePlayPause = () => {
    if (audioRef.current && audioRef.current.src) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(err => {
          console.error('Error playing audio:', err);
        });
      }
    }
  };
  
  // Demo audio for testing - using Web Audio API oscillator
  const loadDemoAudio = () => {
    if (!audioRef.current) return;
    
    // Create a synthetic audio source using Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const duration = 30; // 30 seconds of audio
    const sampleRate = audioContext.sampleRate;
    const buffer = audioContext.createBuffer(2, duration * sampleRate, sampleRate);
    
    // Generate some interesting audio patterns
    for (let channel = 0; channel < 2; channel++) {
      const channelData = buffer.getChannelData(channel);
      for (let i = 0; i < channelData.length; i++) {
        const t = i / sampleRate;
        // Mix of different frequencies to create interesting patterns
        channelData[i] = 
          Math.sin(2 * Math.PI * 440 * t) * 0.1 * Math.sin(0.5 * t) + // A4 with envelope
          Math.sin(2 * Math.PI * 220 * t) * 0.1 * Math.sin(0.3 * t) + // A3
          Math.sin(2 * Math.PI * 880 * t) * 0.05 * Math.sin(0.7 * t) + // A5
          Math.sin(2 * Math.PI * 110 * t) * 0.2 * Math.sin(0.2 * t) + // A2 bass
          (Math.random() - 0.5) * 0.02; // Some noise
      }
    }
    
    // Convert buffer to blob and create URL
    const wav = audioBufferToWav(buffer);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    
    audioRef.current.src = url;
    setIsPlaying(false);
  };
  
  // Helper function to convert AudioBuffer to WAV
  const audioBufferToWav = (buffer) => {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    const setUint16 = (data) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };
    const setUint32 = (data) => {
      view.setUint32(pos, data, true);
      pos += 4;
    };
    
    // RIFF identifier
    setUint32(0x46464952);
    // file length
    setUint32(length - 8);
    // WAVE identifier
    setUint32(0x45564157);
    // fmt chunk identifier
    setUint32(0x20746d66);
    // chunk length
    setUint32(16);
    // sample format (PCM)
    setUint16(1);
    // channel count
    setUint16(buffer.numberOfChannels);
    // sample rate
    setUint32(buffer.sampleRate);
    // byte rate
    setUint32(buffer.sampleRate * buffer.numberOfChannels * 2);
    // block align
    setUint16(buffer.numberOfChannels * 2);
    // bits per sample
    setUint16(16);
    // data chunk identifier
    setUint32(0x61746164);
    // data chunk length
    setUint32(length - pos - 4);
    
    // Write interleaved data
    for (let i = 0; i < buffer.numberOfChannels; i++) {
      channels.push(buffer.getChannelData(i));
    }
    
    while (pos < length) {
      for (let i = 0; i < buffer.numberOfChannels; i++) {
        let sample = Math.max(-1, Math.min(1, channels[i][offset]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(pos, sample, true);
        pos += 2;
      }
      offset++;
    }
    
    return arrayBuffer;
  };
  
  // Ensure samples is power of 2
  const validSamples = useMemo(() => {
    const powers = [64, 256, 1024, 4096];
    return powers.reduce((prev, curr) => 
      Math.abs(curr - samples) < Math.abs(prev - samples) ? curr : prev
    );
  }, [samples]);
  
  return (
    <div className="w-full h-screen bg-black flex flex-col">
      {/* Controls */}
      <div className="bg-gray-900 p-4 flex gap-4 items-center text-white">
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
        >
          Load Audio
        </button>
        
        <button
          onClick={loadDemoAudio}
          className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-700 transition-colors"
        >
          Demo Audio
        </button>
        
        <button
          onClick={togglePlayPause}
          className={`px-4 py-2 rounded transition-colors ${
            audioRef.current?.src 
              ? 'bg-green-600 hover:bg-green-700 cursor-pointer' 
              : 'bg-gray-600 cursor-not-allowed opacity-50'
          }`}
          disabled={!audioRef.current?.src}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="px-4 py-2 bg-gray-800 rounded text-white"
        >
          <option value={MODES.POINTS}>Points</option>
          <option value={MODES.NURBS}>NURBS Surface</option>
        </select>
        
        <select
          value={style}
          onChange={(e) => setStyle(e.target.value)}
          className="px-4 py-2 bg-gray-800 rounded text-white"
        >
          <option value={STYLES.STANDARD}>Standard</option>
          <option value={STYLES.CORNER}>Corner</option>
          <option value={STYLES.CENTER}>Center</option>
          <option value={STYLES.RANDOM}>Random</option>
        </select>
        
        <label className="flex items-center gap-2">
          Samples:
          <select
            value={validSamples}
            onChange={(e) => setSamples(Number(e.target.value))}
            className="px-2 py-1 bg-gray-800 rounded text-white"
          >
            <option value={64}>64</option>
            <option value={256}>256</option>
            <option value={1024}>1024</option>
            <option value={4096}>4096</option>
          </select>
        </label>
        
        <div className="ml-auto text-sm text-gray-400">
          Click and drag to rotate view
        </div>
      </div>
      
      {/* 3D Canvas */}
      <div className="flex-1">
        <ThreeVisualizer 
          frequencyData={frequencyData} 
          samples={validSamples}
          style={style}
          mode={mode}
        />
      </div>
      
      {/* Hidden audio element */}
      <audio 
        ref={audioRef}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />
    </div>
  );
}