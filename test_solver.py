import unittest
from typing import List, Iterable

import solver

empty_chr_seq = solver.empty_chr_seq
empty_groups = solver.empty_group_bindings
empty_state = solver.empty_state
empty_match = solver.possible_match(empty_chr_seq)


def match(*res: solver.ReChr, state=empty_state) -> solver.PossibleMatch:
    return solver.PossibleMatch(solver.ChrSeq(res), state)


class ReChrTest(unittest.TestCase):
    re = solver.re_any

    def test_gen_possible_zero_max(self):
        self.assertListEqual(list(self.re.gen_possible(0, 0, empty_state)), [])

    def test_gen_possible_one_max(self):
        self.assertListEqual(list(self.re.gen_possible(1, 1, empty_state)),
                             [match(self.re)])

    def test_gen_possible_many_max(self):
        self.assertListEqual(list(self.re.gen_possible(1, 5, empty_state)),
                             [match(self.re)])


class ReSeqTest(unittest.TestCase):

    def test_gen_possible_empty_zero_max(self):
        re = solver.ReSeq(())
        self.assertListEqual(list(re.gen_possible(0, 0, empty_state)), [])

    def test_gen_possible_single_zero_max(self):
        re = solver.ReSeq((solver.re_any,))
        self.assertListEqual(list(re.gen_possible(0, 0, empty_state)), [])

    def test_gen_possible_single_one_max(self):
        re = solver.ReSeq((solver.re_any,))
        self.assertListEqual(list(re.gen_possible(1, 1, empty_state)),
                             [match(solver.re_any)])

    def test_gen_possible_single_many_max(self):
        re = solver.ReSeq((solver.re_any,))
        self.assertListEqual(list(re.gen_possible(1, 7, empty_state)),
                             [match(solver.re_any)])

    def test_gen_possible_two(self):
        re = solver.ReSeq((solver.re_any, solver.re_any))
        self.assertListEqual(list(re.gen_possible(1, 7, empty_state)),
                             [match(solver.re_any, solver.re_any)])

    def test_ref(self):
        re = solver.ReSeq.from_string(r'X(A*|BC)\1')
        matches = list(re.gen_possible(5, 5, empty_state))
        chr_seqs = [m.chr_seq for m in matches]

        def cs(*chrs: solver.ReChr):
            return solver.ChrSeq(chrs)

        def lit(*s: str):
            return solver.ReLit(s)

        def ref(i: int):
            return solver.ChrRef(i)

        x, a, b, c = map(lit, 'XABC')

        self.assertListEqual(chr_seqs,
                             [cs(x, a, a, ref(1), ref(2)),
                              cs(x, b, c, ref(1), ref(2))])


class ReRepeatTest(unittest.TestCase):
    lit = solver.ReLit(('A',))
    rep_lit_01 = solver.ReRepeat(0, 1, lit)
    rep_lit_02 = solver.ReRepeat(0, 2, lit)

    def run_lit_01(self, min_len: int, max_len: int) -> List[solver.PossibleMatch]:
        return list(self.rep_lit_01.gen_possible(min_len, max_len, empty_state))

    def run_lit_02(self, min_len: int, max_len: int) -> List[solver.PossibleMatch]:
        return list(self.rep_lit_02.gen_possible(min_len, max_len, empty_state))

    def create_lit_match(self, n: int) -> solver.PossibleMatch:
        return solver.possible_match(solver.chr_seq((self.lit,) * n), empty_state)

    def create_lit_matches(self, min_len: int, max_len: int) -> List[solver.PossibleMatch]:
        return [self.create_lit_match(n) for n in reversed(range(min_len, max_len + 1))]

    def test_gen_possible_lit01_zero_max(self):
        self.assertListEqual(self.run_lit_01(0, 0), [empty_match])

    def test_gen_possible_lit01_one_max(self):
        self.assertListEqual(self.run_lit_01(1, 1), [self.create_lit_match(1)])

    def test_gen_possible_lit01_range_1_2(self):
        self.assertListEqual(self.run_lit_01(1, 2), [self.create_lit_match(1)])

    def test_gen_possible_lit02_range_1_2(self):
        self.assertListEqual(self.run_lit_02(1, 2), self.create_lit_matches(1, 2))


class ReCombinationsTest(unittest.TestCase):

    @staticmethod
    def create_match(s: str) -> solver.PossibleMatch:
        return solver.possible_match(solver.chr_seq(
            tuple(solver.ReLit((c,)) for c in s)))

    def test_start_and_end_repeat(self):
        re = solver.ReSeq.from_string(r'A*BC*')
        cm = self.create_match
        self.assertListEqual(list(re.gen_possible(5, 5, empty_state)),
                             [cm('AAAAB'),
                              cm('AAABC'),
                              cm('AABCC'),
                              cm('ABCCC'),
                              cm('BCCCC')])

    def test_multiple_refs(self):
        re = solver.ReSeq.from_string(r'.*(.)(.)(.)(.)\4\3\2\1.*')
        matches = list(re.gen_possible(12, 12, empty_state))
        chr_seqs_str = [str(m.chr_seq) for m in matches]
        self.assertListEqual(chr_seqs_str, ['........<7><6><5><4>',
                                            '.......<6><5><4><3>.',
                                            '......<5><4><3><2>..',
                                            '.....<4><3><2><1>...',
                                            '....<3><2><1><0>....'])


class SolutionTest(unittest.TestCase):

    @staticmethod
    def make_string(s: str) -> solver.String:
        return solver.String(solver.Pattern(s, solver.ReSeq.from_string(s)), [])

    @staticmethod
    def pos(s: solver.String, cell: solver.Cell, index: int):
        assert index == len(s.positions)
        cn = solver.Position(index=index, string=s, cell=cell)
        s.positions.append(cn)
        cell.positions.append(cn)
        return cn

    @classmethod
    def generate_solutions(cls, chars: str, n: int) -> Iterable[solver.Solution]:
        string = cls.make_string(chars)
        for i in range(n):
            cls.pos(string, solver.Cell(0, i, []), i)
        return solver.Solution.generate_solutions(string)

    @classmethod
    def gen_single_solution(cls, chars: str, n: int) -> solver.Solution:
        itr = iter(cls.generate_solutions(chars, n))
        s = next(itr)
        o = object()
        assert next(itr, o) is o
        return s

    @staticmethod
    def lit_c(*chrs: str, negate=False):
        return solver.LiteralConstraint(frozenset(chrs), negate)

    @staticmethod
    def ref_c(*ixs: solver.cell_ix_type):
        return solver.RefConstraint(frozenset(ixs))

    @staticmethod
    def comp_c(l: solver.LiteralConstraint, r: solver.RefConstraint):
        return solver.CompoundConstraint(l, r)

    def test1(self):
        from pprint import pprint
        for sol in self.generate_solutions('[AB]CD(XY|D|F).*', 6):
            pprint(sol.cells)

    def test_intersection(self):
        a = self.gen_single_solution('[ABC]XYZ', 4)
        b = self.gen_single_solution('[BCD]XY.', 4)
        e = self.gen_single_solution('[BC]XYZ', 4)
        i = a.intersection(b)
        self.assertDictEqual(i.cells, e.cells)

    def test_no_intersection(self):
        a = self.gen_single_solution('[ABC]XYZ', 4)
        b = self.gen_single_solution('[BCD]XYY', 4)
        i = a.intersection(b)
        self.assertIsNone(i)

    def test_ref_intersection(self):
        a = self.gen_single_solution(r'([ABC])\1', 2)
        b = self.gen_single_solution('[AB]A', 2)
        i = a.intersection(b)
        e = {
            (0, 0): self.lit_c('A', 'B'),
            (0, 1): self.comp_c(self.lit_c('A'), self.ref_c((0, 0)))
        }
        self.assertDictEqual(i.cells, e)

    def test_ref_no_intersection(self):
        a = self.gen_single_solution(r'(A|B|C)\1', 2)
        b = self.gen_single_solution('AB', 2)
        i = a.intersection(b)
        self.assertIsNone(i)
