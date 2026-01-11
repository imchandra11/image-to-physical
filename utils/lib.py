import importlib
from typing import Any, Dict, Optional, Tuple


def getAttr(path: str) -> Any:
    """
    Split the . path and return type of object at the end.
    """
    module_path, name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def getCallableAndArgs(
    class_dict: Optional[Dict[str, Any]], 
    function_dict: Optional[Dict[str, Any]]
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Returns a callable object and its initialization arguments based on the 
    provided dictionaries.
    
    Args:
        class_dict: A dictionary containing the class path and initialization 
            arguments.
        function_dict: A dictionary containing the function path and arguments.
    
    Returns:
        A tuple containing the callable object and its initialization arguments.
    
    Raises:
        ValueError: If neither class_dict nor function_dict is provided.
    """
    if class_dict:
        callable_c = getAttr(class_dict['class_path'])
        init_args = class_dict.get('init_args')
    elif function_dict:
        callable_c = getAttr(function_dict['function_path'])
        init_args = function_dict.get('init_args')
    else:
        raise ValueError("Either class_dict or function_dict must be provided")
    
    return callable_c, init_args