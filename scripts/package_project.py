from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
zip_path = root.with_suffix('.zip')

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob('*'):
        if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts:
            zf.write(path, path.relative_to(root.parent))

print(zip_path)
