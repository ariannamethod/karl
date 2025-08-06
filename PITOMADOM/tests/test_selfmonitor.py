def _flatten(x):
    try:
        return x.reshape(-1).tolist()
    except AttributeError:
        return [float(x[0][0])]


class SimpleIndex:
    def __init__(self):
        self.vectors = []

    def add(self, x):
        self.vectors.extend(float(v) for v in _flatten(x))

    def search(self, x, k):
        q = float(_flatten(x)[0])
        scores = [-(abs(v - q)) for v in self.vectors]
        topk = sorted(range(len(self.vectors)), key=lambda i: scores[i], reverse=True)[:k]
        return [[scores[i] for i in topk]], [topk]


def test_selfmonitor_index_usage(tmp_path):
    import importlib
    import sys

    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "nn"):
        sys.modules.pop("torch")
        importlib.import_module("torch")
    from inference.model import SelfMonitor

    idx = SimpleIndex()
    sm = SelfMonitor(path=str(tmp_path / "sm.json"), index=idx)
    sm.note("hello world")
    sm.learn("feedback")
    assert len(idx.vectors) == 2
    res = sm.search_faiss("hello", limit=1)
    assert res[0] == "hello world"


def test_selfmonitor_fallback(tmp_path, monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "faiss", None)
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "nn"):
        sys.modules.pop("torch")
        importlib.import_module("torch")
    from inference.model import SelfMonitor

    sm = SelfMonitor(path=str(tmp_path / "sm.json"), index=None)
    sm.note("alpha beta")
    res = sm.search_faiss("alpha")
    assert res == ["alpha beta"]
