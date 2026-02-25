import inspect

def assert_has_args(func, expected_args):
    sig = inspect.signature(func)
    params = sig.parameters

    missing = [arg for arg in expected_args if arg not in params]

    if missing:
        raise TypeError(
            f"{func.__name__} must accept arguments {expected_args}, "
            f"missing: {missing}"
        )