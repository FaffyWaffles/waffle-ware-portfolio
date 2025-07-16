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

// Main 3D Visualizer Component
function ThreeVisualizer({ frequencyData, samples, style }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const meshesRef = useRef([]);
  const frameRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const rotationRef = useRef({ x: 0, y: 0 });
  
  useEffect(() => {
    if (!mountRef.current) return;
    
    // Setup Three.js
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
    const handleMouseMove = (event) => {
      mouseRef.current.x = (event.clientX / width) * 2 - 1;
      mouseRef.current.y = -(event.clientY / height) * 2 + 1;
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
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
      
      // Rotate camera based on mouse
      rotationRef.current.y += (mouseRef.current.x * 0.5 - rotationRef.current.y) * 0.05;
      rotationRef.current.x += (mouseRef.current.y * 0.5 - rotationRef.current.x) * 0.05;
      
      camera.position.x = Math.sin(rotationRef.current.y) * 1.5;
      camera.position.y = 0.5 + rotationRef.current.x;
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
      window.removeEventListener('mousemove', handleMouseMove);
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
  
  // Update only the frequency response
  useEffect(() => {
    // This effect runs when frequencyData changes
    // The actual update happens in the animation loop
  }, [frequencyData]);
  
  return <div ref={mountRef} className="w-full h-full" />;
}

// Main App Component
export default function AudioVisualizer3D() {
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  const [samples, setSamples] = useState(1024);
  const [style, setStyle] = useState(STYLES.STANDARD);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const frequencyData = useAudioAnalyzer(audioRef);
  
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && audioRef.current) {
      const url = URL.createObjectURL(file);
      audioRef.current.src = url;
    }
  };
  
  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
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
          onClick={togglePlayPause}
          className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors"
          disabled={!audioRef.current?.src}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        
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
          Move mouse to rotate view
        </div>
      </div>
      
      {/* 3D Canvas */}
      <div className="flex-1">
        <ThreeVisualizer 
          frequencyData={frequencyData} 
          samples={validSamples}
          style={style}
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