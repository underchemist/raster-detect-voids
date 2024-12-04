from pathlib import Path

def list_files(directory_path, extensions=['*']):
    # https://stackoverflow.com/a/77259205
    return [path for i in extensions for path in Path(directory_path).rglob("*."+i)]