from odinson.ruleutils import parse_odinson_query
from odinson.ruleutils.config import HOLE_GLYPH
from odinson.ruleutils.queryast import HoleSurface
import pyparsing
import unittest


class TestQueryParser(unittest.TestCase):
    def test_parse(self):
        res = parse_odinson_query(HOLE_GLYPH)
        self.assertTrue(isinstance(res, HoleSurface))
        # this should fail
        with self.assertRaises(pyparsing.ParseException):
            parse_odinson_query("¯\(°_o)/¯")
