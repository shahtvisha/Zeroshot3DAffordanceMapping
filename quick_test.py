from pathlib import Path
from web.app import _parse_ply

path = Path("data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply")
pts, col = _parse_ply(path)
print(pts.shape)