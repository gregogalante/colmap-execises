import { useLoader } from '@react-three/fiber';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

export default function PointCloud({ url }) {
  const geometry = useLoader(PLYLoader, url);

  return (
    <points>
      <primitive object={geometry} attach="geometry" />
      <pointsMaterial size={0.03} vertexColors sizeAttenuation />
    </points>
  );
}
