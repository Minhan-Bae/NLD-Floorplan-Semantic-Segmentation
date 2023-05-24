import os
import shutil
import sys

def remove_cache(path):
    dirs_to_remove = ["__pycache__"]
    for dirpath, dirnames, filenames in os.walk(path):
        if any(dir_to_remove in dirpath for dir_to_remove in dirs_to_remove):
            print(f"Removing {dirpath}")
            shutil.rmtree(dirpath)

def print_files(path, prefix=""):
    # Skip directories to ignore
    dirs_to_ignore = ["__pycache__", ".git"]
    if any(dir_to_ignore in os.path.basename(path) for dir_to_ignore in dirs_to_ignore):
        return

    print(f"{prefix}{os.path.basename(path)}")

def print_dir(path, prefix=""):
    # Skip directories to ignore
    dirs_to_ignore = ["__pycache__", ".git"]
    if any(dir_to_ignore in os.path.basename(path) for dir_to_ignore in dirs_to_ignore):
        return

    items = os.listdir(path)
    items = sorted(items, key=lambda x: (os.path.isfile(os.path.join(path, x)), x))
    for i, item in enumerate(items):
        new_path = os.path.join(path, item)
        if i == len(items) - 1:
            if os.path.isdir(new_path):
                print(f"{prefix}└── {os.path.basename(new_path)}/")
                print_dir(new_path, prefix + "    ")
            else:
                print_files(new_path, prefix[:-4] + "    └── ")
        else:
            if os.path.isdir(new_path):
                print(f"{prefix}├── {os.path.basename(new_path)}/")
                print_dir(new_path, prefix + "│   ")
            else:
                print_files(new_path, prefix[:-4] + "│   ├── ")

if __name__ == "__main__":
    remove_cache(sys.argv[1])
    print_dir(sys.argv[1], "")
