import unittest
from typing import List

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

    def test_example(self):
        re = solver.ReSeq.from_string(r'.*(.BC)\1(\1|D)')
        for m in re.gen_possible(13, 13, empty_state):
            print(m)


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
        return [self.create_lit_match(n) for n in reversed(range(min_len, max_len+1))]

    def test_gen_possible_lit01_zero_max(self):
        self.assertListEqual(self.run_lit_01(0, 0), [empty_match])

    def test_gen_possible_lit01_one_max(self):
        self.assertListEqual(self.run_lit_01(1, 1), [self.create_lit_match(1)])

    def test_gen_possible_lit01_range_1_2(self):
        self.assertListEqual(self.run_lit_01(1, 2), [self.create_lit_match(1)])

    def test_gen_possible_lit02_range_1_2(self):
        self.assertListEqual(self.run_lit_02(1, 2), self.create_lit_matches(1, 2))
