export default function RelocationList({ relocations, show, onToggleShow }) {
  return (
    <div>
      <label className="flex items-center gap-2 mb-2 text-xs text-gray-400 cursor-pointer">
        <input
          type="checkbox"
          checked={show}
          onChange={onToggleShow}
          className="rounded"
        />
        Show in viewer
      </label>
      <div className="space-y-0.5">
        {relocations.map(rel => (
          <div
            key={rel.name}
            className="flex items-center gap-2 px-2 py-1 rounded text-xs text-gray-400"
          >
            <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
            <span className="truncate">{rel.name}</span>
            <span className="ml-auto text-gray-600">{rel.num_inliers} inliers</span>
          </div>
        ))}
      </div>
    </div>
  );
}
