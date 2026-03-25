import { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import PointCloud from './PointCloud';
import ImagePlane from './ImagePlane';
import RelocationMarker from './RelocationMarker';
import { plyUrl } from '../api';

export default function Viewer3D({ dataset, cameras, activeImages, relocations }) {
  const activeCameras = useMemo(
    () => cameras.filter(c => activeImages.has(c.image_name)),
    [cameras, activeImages]
  );

  return (
    <Canvas
      camera={{ position: [0, 5, 10], fov: 60, near: 0.01, far: 1000 }}
      className="!absolute inset-0"
    >
      <ambientLight intensity={1} />
      <OrbitControls makeDefault />
      <axesHelper args={[2]} />

      <group rotation={[-Math.PI / 2, 0, 0]}>
        <Suspense fallback={null}>
          <PointCloud url={plyUrl(dataset)} key={dataset} />
        </Suspense>

        {activeCameras.map(cam => (
          <Suspense key={cam.image_name} fallback={null}>
            <ImagePlane camera={cam} dataset={dataset} />
          </Suspense>
        ))}

        {relocations.map(rel => (
          <RelocationMarker key={rel.name} relocation={rel} />
        ))}
      </group>
    </Canvas>
  );
}
