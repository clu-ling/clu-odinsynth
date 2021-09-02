import odinson.ruleutils
from odinson.ruleutils import info
from .utils import EXAMPLE
import unittest


class TestTemplate(unittest.TestCase):
    def test_example(self):
        assert EXAMPLE == "Example"

    def test_package(self):
        assert info.authors == ["marcovzla", "myedibleenso", "BeckySharp"]
