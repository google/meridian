class immutabledict(dict):
    """Simple immutable dict implementation for tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _blocked(self, *args, **kwargs):
        raise TypeError("immutabledict is immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked

    def __hash__(self):
        return hash(tuple(sorted(self.items())))
