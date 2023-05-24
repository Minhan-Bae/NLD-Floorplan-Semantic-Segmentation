# This script defines a function to parse a string into a boolean value.
# It is intended for use with argparse, a Python module for parsing command-line options and arguments.

import argparse

def str2bool(v):
    """
    Convert a string input to a boolean value.
    
    Args:
        v (str): The string to convert.
        
    Returns:
        bool: True if the input string represents a truthy value, False if it represents a falsy value.
        
    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid representation of a boolean value.
    """
    # Check if the input string represents a truthy value.
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # Check if the input string represents a falsy value.
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # Raise an error if the input string is not a valid boolean representation.
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
