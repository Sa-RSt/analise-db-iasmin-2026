from contextlib import contextmanager

_stack = []


def _get_stack() -> list[dict[str, str]]:
    return _stack


def push_options(opts: dict[str, str]):
    stack = _get_stack()
    stack.append(opts)


def pop_options() -> dict[str, str] | None:
    stack = _get_stack()
    stack.pop()


@contextmanager
def set_options(opts: dict[str, str]):
    push_options(opts)
    yield
    pop_options()


def get_options() -> dict[str, str]:
    stack = _get_stack()
    if stack:
        return stack[-1]
    return {}
