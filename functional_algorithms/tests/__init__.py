def run():
    import pytest
    import os

    pytest.main(["-svx", os.path.dirname(__file__)])
