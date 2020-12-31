def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4

class TestClass(object):
    def test_one(self):
        x = "this"
        assert "h" in x
    
    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
