# This script generates a requirements.txt file for a given Python project directory.
# It automatically detects the imported modules in the project and writes the corresponding
# library names and versions to the requirements.txt file.

import os
import pkg_resources

def extract_modules_from_files(path):
    """
    Extract the imported modules from Python files in the given path.

    Args:
        path (str): The path of the Python project directory.

    Returns:
        set: The set of unique imported modules.
    """
    modules = set()
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("import") or line.startswith("from"):
                            parts = line.split()
                            module = parts[1].split(".")[0] if parts[0] == "import" else parts[1]
                            modules.add(module)

    return modules

def generate_requirements_txt(path, output_file="requirements.txt"):
    """
    Generate a requirements.txt file for a Python project.

    Args:
        path (str): The path of the Python project directory.
        output_file (str, optional): The name of the output file. Defaults to "requirements.txt".
    """
    modules = extract_modules_from_files(path)
    installed_packages = {pkg.key: pkg for pkg in pkg_resources.working_set}
    requirements = []
    
    for module in modules:
        if module in installed_packages:
            pkg = installed_packages[module]
            requirements.append(f"{pkg.project_name}=={pkg.version}")

    with open(output_file, "w") as f:
        for req in sorted(requirements):
            f.write(req + "\n")

    print(f"Generated {output_file}")

if __name__=="__main__":
    repository_path = os.getcwd()  # Use the current working directory
    generate_requirements_txt(repository_path)