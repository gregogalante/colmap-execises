export async function fetchDatasets() {
  const res = await fetch('/api/datasets');
  const data = await res.json();
  return data.datasets;
}

export async function fetchCameras(dataset) {
  const res = await fetch(`/api/datasets/${dataset}/cameras`);
  const data = await res.json();
  return data.cameras;
}

export async function fetchRelocations() {
  const res = await fetch('/api/relocations');
  const data = await res.json();
  return data.relocations;
}

export function imageUrl(dataset, filename) {
  return `/api/datasets/${dataset}/images/${filename}`;
}

export function plyUrl(dataset) {
  return `/api/datasets/${dataset}/reconstruction.ply`;
}

export function relocationImageUrl(folder, name) {
  return `/api/relocations/${folder}/${name}/image`;
}
