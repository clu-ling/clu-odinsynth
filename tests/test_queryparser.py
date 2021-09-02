from odinson.ruleutils import QueryParser
from odinson.ruleutils.config import HOLE_GLYPH
from odinson.ruleutils.queryast import HoleSurface
import pyparsing
import unittest

class TestQueryParser(unittest.TestCase):

    def test_parse(self):
        qp = QueryParser()
        res = qp.parse(HOLE_GLYPH)
        self.assertTrue(isinstance(res, HoleSurface))
        # this should fail
        with self.assertRaises(pyparsing.ParseException):
          qp.parse("¯\(°_o)/¯")

