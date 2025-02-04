import importlib.util
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def print_project_structure(root):
    logger.debug("Project directory structure:")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        indent = " " * 4 * depth
        logger.debug(f"{indent}{os.path.basename(dirpath)}/")
        for f in filenames:
            logger.debug(f"{indent}    {f}")

def check_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        logger.error(f"Module '{module_name}' not found.")
    else:
        logger.info(f"Module '{module_name}' found: {spec.origin}")
    return spec

if __name__ == "__main__":
    project_root = os.getcwd()
    print_project_structure(project_root)

    # Check for the tensor_graph module in the pyptx package
    check_module("pyptx.tensor_graph")

    logger.info("Debug scan complete.")