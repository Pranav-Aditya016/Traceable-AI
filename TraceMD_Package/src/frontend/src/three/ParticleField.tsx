import { useEffect, useRef } from "react";
import * as THREE from "three";

const PARTICLE_COUNT = 800;

export default function ParticleField() {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 12;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 12;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 12;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.03,
      transparent: true,
      opacity: 0.75
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    let rafId = 0;
    const animate = () => {
      points.rotation.y += 0.0008;
      points.rotation.x += 0.0004;
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(animate);
    };

    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };

    const handleScroll = () => {
      points.position.y = -window.scrollY * 0.003;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("scroll", handleScroll, { passive: true });
    animate();

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleScroll);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={mountRef} className="absolute inset-0" />;
}
