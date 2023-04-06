from typing import Any


def instantiate_class(args: Any | tuple[Any, ...], init: dict[str, Any]) -> Any:
    kwargs = init.get('init_args', {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init['class_path'].rsplit('.', 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)
