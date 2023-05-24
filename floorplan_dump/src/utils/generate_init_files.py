"""
This script creates __init__.py files for all directories and subdirectories
in the specified repository path. For each __init__.py file, it also adds
import statements for all the Python modules in the same directory.
Please note that this approach can lead to complications when managing
complex package structures. Modify the script as needed before using it
in a real project.
"""

import os

def create_init_py(path):
    """
    Create __init__.py files in all directories and subdirectories
    within the given path, and add import statements for all Python modules
    in the same directory.

    Args:
        path (str): The root directory to start creating __init__.py files from.
    """
    # Iterate through all directories and files in the given path.
    for root, _, files in os.walk(path):
        init_path = os.path.join(root, "__init__.py")

        # Create a new __init__.py file.
        with open(init_path, "w") as init_file:
            # Write the default comment.
            init_file.write("# This is an automatically generated __init__.py file\n")

            # Import all modules in the current directory.
            for file in sorted(files):
                if file.endswith(".py") and file != "__init__.py":
                    module_name = file[:-3]  # Remove '.py' extension
                    init_file.write(f"from . import {module_name}\n")

            print(f"Created: {init_path}")

# Use the current working directory as the repository path.
if __name__=="__main__":
    repository_path = os.getcwd()
    create_init_py(repository_path)