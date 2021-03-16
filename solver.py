import copy
import sre_constants
import sre_parse
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache, reduce, total_ordering
from random import Random
from typing import (Any, Callable, Dict, List, Tuple, Optional, Collection, Iterable, Sequence, Union, overload,
                    TypeVar, FrozenSet, Iterator, Mapping, Set)

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as np_random
from matplotlib.collections import PatchCollection
from matplotlib.markers import Path as MarkerPath
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D
from tqdm import tqdm

import constants

size = constants.size
assert size % 2 == 1
mid = size // 2


def row_size(row_index: int) -> int:
    assert 0 <= row_index < size
    return mid + min(row_index + 1, size - row_index)


T = TypeVar('T')
U = TypeVar('U')


def or_else(x: Optional[T], e: T) -> T:
    return x if x is not None else e


@dataclass(frozen=True)
class ChrSeq(Sequence['ReChr']):
    chrs: Tuple['ReChr', ...]

    def add(self, other: 'ChrSeq') -> 'ChrSeq':
        return chr_seq(self.chrs + other.chrs)

    @overload
    def __getitem__(self, i: int) -> 'ReChr':
        pass

    @overload
    def __getitem__(self, s: slice) -> 'ChrSeq':
        pass

    def __getitem__(self, i: Union[int, slice]) -> Union['ReChr', 'ChrSeq']:
        if isinstance(i, slice):
            return ChrSeq(self.chrs[i])
        else:
            return self.chrs[i]

    def __len__(self) -> int:
        return len(self.chrs)

    def __str__(self):
        return ''.join(map(str, self.chrs))

    def __repr__(self) -> str:
        return f'<CSeq {self!r}>'


@lru_cache(1000)
def chr_seq(chrs: Tuple['ReChr', ...]) -> ChrSeq:
    return ChrSeq(chrs)


empty_chr_seq = chr_seq(())


@dataclass(frozen=True)
class GroupBinding:
    start: int
    end: int

    def offset(self, n: int) -> 'GroupBinding':
        return GroupBinding(self.start + n, self.end + n)


@dataclass(frozen=True)
class GroupBindings:
    bindings: Tuple[GroupBinding, ...] = ()

    def bind(self, index: int, cs: GroupBinding) -> 'GroupBindings':
        assert index > 0
        if index - 1 == len(self.bindings):  # appending to end
            return GroupBindings(self.bindings + (cs,))
        else:  # resetting an existing group in a repeated group
            assert index <= len(self.bindings)
            bs = list(self.bindings)
            bs[index - 1] = cs
            return GroupBindings(tuple(bs))

    def get(self, index: int) -> 'GroupBinding':
        return self.bindings[index - 1]

    def offset(self, n: int) -> 'GroupBindings':
        assert n >= 0
        return GroupBindings(tuple(b.offset(n) for b in self.bindings))

    def __str__(self):
        return '(' + ', '.join(map(str, self.bindings)) + ')'

    def __repr__(self) -> str:
        return f'<GBs{self!s}>'

    def merge(self, other: 'GroupBindings') -> 'GroupBindings':
        return GroupBindings(self.bindings + other.bindings)


empty_group_bindings = GroupBindings(())


@dataclass(frozen=True)
class MatchState:
    start_index: int = 0
    groups: GroupBindings = empty_group_bindings

    def copy(self, start_index: Optional[int] = None,
             groups: Optional[GroupBindings] = None) -> 'MatchState':
        return MatchState(start_index=or_else(start_index, self.start_index),
                          groups=or_else(groups, self.groups))


empty_state = MatchState()


@dataclass(frozen=True)
class PossibleMatch:
    chr_seq: ChrSeq
    state: MatchState

    def __repr__(self) -> str:
        return f'<PMch {self.chr_seq!s} {self.state!r}>'

    def copy(self,
             chr_seq: Optional[ChrSeq] = None,
             state: Optional[MatchState] = None) -> 'PossibleMatch':
        return possible_match(chr_seq=or_else(chr_seq, self.chr_seq),
                              state=or_else(state, self.state))

    def extend(self, other: 'PossibleMatch'):
        if not self.chr_seq:
            return other
        return other.copy(chr_seq=self.chr_seq.add(other.chr_seq),
                          state=other.state.copy(start_index=self.state.start_index))


def possible_match(chr_seq: ChrSeq, state: MatchState = empty_state) -> PossibleMatch:
    assert isinstance(state, MatchState)
    return PossibleMatch(chr_seq, state)


class Re(ABC):
    op_converters: Dict[int, Callable[[int, Any, Dict[int, 'Re']], 'Re']] = {}

    @abstractmethod
    def span(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def gen_possible(self, min_len: int, max_len: int, groups: MatchState) -> Iterable[PossibleMatch]:
        pass

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Dict[int, 'Re']) -> 'Re':
        return cls.op_converters[op](op, arg, groups)


class ReChr(ABC):
    pass


class ReAny(Re, ReChr):
    _instance = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def span(self) -> Tuple[int, int]:
        return 1, 1

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        return [possible_match(chr_seq((self,)), state)] if max_len >= 1 else ()

    @classmethod
    def from_op_arg(cls, op, arg, groups) -> 'ReAny':
        assert op == sre_constants.ANY
        assert arg is None
        return re_any

    def __str__(self):
        return '.'

    def __repr__(self):
        return f'<{self.__class__.__name__}>'


re_any = ReAny._instance = ReAny()

Re.op_converters[sre_constants.ANY] = ReAny.from_op_arg


@dataclass(frozen=True)
class ReLit(Re, ReChr):
    chars: Tuple[str, ...]
    negate: bool = False

    def span(self) -> Tuple[int, int]:
        return 1, 1

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        if max_len >= 1:
            yield possible_match(chr_seq((self,)), state)

    def __str__(self):
        s = ''.join(self.chars)
        if self.negate:
            s = '^' + s
        if len(s) == 1:
            return s
        return f'[{s}]'

    def __repr__(self):
        return f'<Lit {self!s}>'

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups) -> 'ReLit':
        if op == sre_constants.LITERAL:
            return re_lit((chr(arg),))
        if op == sre_constants.NOT_LITERAL:
            return re_lit((chr(arg),), negate=True)
        elif op == sre_constants.IN:
            els: Collection[Tuple[int, Any]] = arg
            assert isinstance(els, (list, tuple))
            assert len(els) >= 1
            negate = False
            if els[0][0] == sre_constants.NEGATE:
                negate = True
                els = els[1::]
            assert all(op == sre_constants.LITERAL for op, _ in els)
            return re_lit(tuple(chr(sub_arg) for op, sub_arg in els), negate=negate)
        else:
            raise ValueError(f'unknown op {op}')


@lru_cache(1000)
def re_lit(chars: Tuple[str, ...], negate: bool = False) -> ReLit:
    return ReLit(chars, negate)


Re.op_converters[sre_constants.LITERAL] = ReLit.from_op_arg
Re.op_converters[sre_constants.NOT_LITERAL] = ReLit.from_op_arg
Re.op_converters[sre_constants.IN] = ReLit.from_op_arg


def extend_matches(re: Re, max_len: int, previous: Iterable[PossibleMatch]) -> Iterable[PossibleMatch]:
    if max_len < 0:
        return
    for prev in previous:
        n = len(prev.chr_seq)
        mx = max_len - n
        if mx >= 0:  # repeat can have 0-length matches
            for match in re.gen_possible(0, mx, prev.state.copy(start_index=n + prev.state.start_index)):
                ext = prev.extend(match)
                yield ext


@dataclass(frozen=True)
class ReSeq(Re):
    res: Tuple[Re, ...]
    index: Optional[int] = None

    def span(self) -> Tuple[int, int]:
        span = np.zeros((2,), dtype=int)
        for re in self.res:
            span += np.array(re.span())
        return tuple(span)

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        res = self.res
        n = len(res)
        if not n:
            return ()

        mns, mxs = np.array([r.span() for r in res]).T
        required_min = np.concatenate([np.cumsum(mns[::-1])[::-1], [0]])

        def rec(index: int, previous: Iterable[PossibleMatch]) -> Iterable[PossibleMatch]:
            if index == n:
                for p in previous:
                    s = len(p.chr_seq)
                    assert s <= max_len
                    if s >= min_len:
                        if self.index is not None:
                            start_ix = p.state.start_index
                            p = p.copy(state=p.state.copy(
                                groups=p.state.groups.bind(self.index,
                                                           GroupBinding(start=start_ix, end=start_ix + s))))
                        yield p
            else:
                yield from rec(index + 1, extend_matches(res[index],
                                                         max_len - required_min[index + 1], previous))

        return rec(0, [possible_match(empty_chr_seq, state)])

    @classmethod
    def from_string(cls, pattern: str) -> 'ReSeq':
        parsed: sre_parse.SubPattern = sre_parse.parse(pattern)
        return cls.from_op_args(parsed.data, {})

    @classmethod
    def from_op_args(cls,
                     ops_args: Iterable[Tuple[int, Any]],
                     groups: Dict[int, 'ReSeq'],
                     index: Optional[int] = None) -> 'ReSeq':
        return cls(res=tuple(Re.from_op_arg(op, arg, groups) for op, arg in ops_args), index=index)

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Dict[int, 'ReSeq']) -> 'ReSeq':
        assert op == sre_constants.SUBPATTERN
        index, add_flags, del_flags, p = arg
        assert index not in groups
        assert add_flags == 0
        assert del_flags == 0
        r = cls.from_op_args(ops_args=p, groups=groups, index=index)
        assert index not in groups
        groups[index] = r
        return r


Re.op_converters[sre_constants.SUBPATTERN] = ReSeq.from_op_arg


@dataclass(frozen=True)
class ChrRef(ReChr):
    index: int

    def __str__(self):
        return f'<{self.index}>'

    def __repr__(self):
        return f'<ChrRef {self.index}>'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ChrRef) and self.index == o.index

    def __hash__(self) -> int:
        return hash(self.index)


@dataclass(frozen=True)
class ReGroupRef(Re):
    group: ReSeq

    def span(self) -> Tuple[int, int]:
        return self.group.span()

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        binding = state.groups.get(self.group.index)
        n = binding.end - binding.start
        if min_len <= n <= max_len:
            chr_seq = ChrSeq(tuple(ChrRef(i) for i in range(binding.start, binding.end)))
            yield possible_match(chr_seq, state)

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Dict[int, ReSeq]) -> 'ReGroupRef':
        assert op == sre_constants.GROUPREF
        assert isinstance(arg, int)
        return cls(groups[arg])


Re.op_converters[sre_constants.GROUPREF] = ReGroupRef.from_op_arg


@dataclass(frozen=True)
class ReRepeat(Re):
    min_times: int
    max_times: int
    re: Re

    def span(self) -> Tuple[int, int]:
        mn, mx = self.re.span()
        return mn * self.min_times, mx * self.max_times

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        min_times = self.min_times
        max_times = self.max_times
        re = self.re
        re_min, re_max = re.span()
        if re_max * max_times < min_len:
            return
        if re_min * min_times > max_len:
            return

        def rec(depth: int, previous: Iterable[PossibleMatch]) -> Iterable[PossibleMatch]:
            if depth > max_times:
                return
            for p in previous:
                min_remaining_times = max(0, min_times - depth - 1)
                yield from rec(depth + 1,
                               extend_matches(re, max_len - min_remaining_times * re_min, [p]))
                if depth >= min_times and len(p.chr_seq) >= min_len:
                    assert len(p.chr_seq) <= max_len
                    yield p

        yield from rec(0, [possible_match(empty_chr_seq, state)])

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Dict[int, ReSeq]) -> 'ReRepeat':
        assert op == sre_constants.MAX_REPEAT
        mn, mx, p = arg
        if mx == sre_constants.MAXREPEAT:
            mx = constants.size
        assert len(p) == 1
        [(r_op, r_arg)] = p
        return cls(min_times=mn, max_times=mx, re=Re.from_op_arg(r_op, r_arg, groups))


Re.op_converters[sre_constants.MAX_REPEAT] = ReRepeat.from_op_arg


@dataclass(frozen=True)
class ReBranch(Re):
    branches: Tuple[ReSeq]

    def span(self) -> Tuple[int, int]:
        mns, mxs = zip(*(b.span() for b in self.branches))
        return min(mns), max(mxs)

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        for b in self.branches:
            yield from b.gen_possible(min_len, max_len, state)

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Dict[int, ReSeq]) -> 'ReBranch':
        assert op == sre_constants.BRANCH
        x, bs = arg
        assert x is None
        return cls(tuple(ReSeq.from_op_args(b, groups) for b in bs))


Re.op_converters[sre_constants.BRANCH] = ReBranch.from_op_arg


@dataclass(frozen=True)
class Pattern:
    raw: str
    re: ReSeq


@dataclass(frozen=True)
class Position:
    index: int
    string: 'String'
    cell: 'Cell'

    def __repr__(self) -> str:
        return f'<Cn ix={self.index} st={self.string.pattern.raw!r} cl={self.cell.index}>'


@dataclass()
class String:
    pattern: Pattern
    positions: List[Position]
    name: str = ''

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.cell.index}' for c in self.positions)
        return f'<St pt={self.pattern.raw!r} cns=[{cns}]>'

    @property
    def size(self):
        return len(self.positions)

    def gen_possible(self) -> Iterable[PossibleMatch]:
        return self.pattern.re.gen_possible(self.size, self.size, empty_state)


cell_ix_type = Tuple[int, int]


@dataclass()
class Cell:
    row: int
    col: int
    positions: List[Position]

    @property
    def index(self) -> cell_ix_type:
        return self.row, self.col

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.index}@{c.string.pattern.raw!r}' for c in self.positions)
        return f'<Ce ix={self.index} cns=[{cns}]>'


def build_strings() -> List[String]:
    cells = [[Cell(i, j, []) for j in range(row_size(i))]
             for i in range(constants.size)]

    strings: List[String] = []

    def make_string(s: str, name: str) -> String:
        s = String(Pattern(s, ReSeq.from_string(s)), [], name)
        strings.append(s)
        return s

    def pos(s: String, cell: Cell, index: int):
        assert 0 <= index < constants.size, f'{index} is invalid'
        cn = Position(index=index, string=s, cell=cell)
        s.positions.append(cn)
        cell.positions.append(cn)

    def add_y():
        for i, s in enumerate(constants.y[::-1]):
            s = make_string(s, f'y{i}')
            for j, c in enumerate(cells[i]):
                pos(s, c, j)

    def create_diag_range(i: int) -> Tuple[int, int]:
        start = max(0, i - mid)
        end = min(i + mid + 1, constants.size)
        return start, end

    def add_x():
        for ii, s in enumerate(constants.x):
            s = make_string(s, f'x{ii}')
            start, end = create_diag_range(ii)
            for jj in range(start, end):
                i = constants.size - jj - 1
                j = ii
                if jj > mid:
                    j -= jj - mid
                cell = cells[i][j]
                pos(s, cell, end - jj - 1)

    def add_z():
        for i, s in enumerate(constants.z):
            s = make_string(s, f'z{i}')
            start, end = create_diag_range(i)
            for j in range(start, end):
                cell = cells[j][i if j <= mid else i - (j - mid)]
                pos(s, cell, end - j - 1)

    add_y()
    add_x()
    add_z()

    for string in strings:
        string.positions.sort(key=lambda c: c.index)
        assert [c.index for c in string.positions] == list(range(len(string.positions)))

    cell_constraints_count = Counter(len(c.positions) for row in cells for c in row)
    assert cell_constraints_count == {3: sum(map(len, cells))}, f'{cell_constraints_count}'

    return strings


@total_ordering
class Constraint(ABC):

    @abstractmethod
    def intersection(self, other: 'Constraint') -> Optional['Constraint']:
        pass

    @abstractmethod
    def label(self) -> str:
        pass

    @property
    @abstractmethod
    def has_references(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def class_rank(cls) -> int:
        pass

    @abstractmethod
    def less_than_same_type(self, other: 'Constraint') -> bool:
        pass

    @property
    @abstractmethod
    def lit_component(self) -> 'LiteralConstraint':
        pass

    @property
    @abstractmethod
    def ref_component(self) -> 'RefConstraint':
        pass

    @property
    def compound_constraint(self) -> 'CompoundConstraint':
        return CompoundConstraint(self.lit_component, self.ref_component)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self}>'

    def __str__(self) -> str:
        return self.label()

    def __lt__(self, other):
        if self.__class__ != other.__class__:
            return self.class_rank() < other.class_rank()
        return self.less_than_same_type(other)


class AnyConstraint(Constraint):
    _instance: Optional['AnyConstraint'] = None

    def __new__(cls) -> 'AnyConstraint':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def intersection(self, other: Constraint) -> Constraint:
        return other

    def label(self) -> str:
        return '.'

    @property
    def has_references(self) -> bool:
        return False

    @classmethod
    def class_rank(cls) -> int:
        return 0

    def less_than_same_type(self, other: 'AnyConstraint') -> bool:
        return False

    @property
    def lit_component(self) -> 'LiteralConstraint':
        return all_literal_constraint

    @property
    def ref_component(self) -> 'RefConstraint':
        return empty_ref_constraint


any_constraint = AnyConstraint()


@dataclass(frozen=True, repr=False)
class LiteralConstraint(Constraint):
    chars: FrozenSet[str]
    negate: bool

    def intersection(self, other: Constraint) -> Optional[Constraint]:
        if other is any_constraint:
            return self
        if isinstance(other, RefConstraint):
            return CompoundConstraint(self, other)
        if isinstance(other, CompoundConstraint):
            lit = self.intersection(other.lit_component)
            if lit is None:
                return None
            assert isinstance(lit, LiteralConstraint)
            return CompoundConstraint(lit, other.ref_component)

        assert isinstance(other, LiteralConstraint)
        if self.negate == other.negate:
            if self.negate:
                return LiteralConstraint(self.chars | other.chars, negate=True)
            combined = self.chars & other.chars
            if not combined:
                return None
            return LiteralConstraint(combined, negate=False)

        pos, neg = self, other
        if pos.negate:
            pos, neg = neg, pos
        assert not pos.negate
        assert neg.negate
        combined = pos.chars - neg.chars
        if not combined:
            return None
        return LiteralConstraint(combined, negate=False)

    def label(self) -> str:
        if self.negate and not self.chars:
            return '[.]'
        s = ''.join(self.chars)
        if self.negate:
            s = '^' + s
        if len(s) > 1:
            s = f'[{s}]'
        return s

    @classmethod
    def for_re_lit(cls, chr: ReLit) -> 'LiteralConstraint':
        return cls(frozenset(chr.chars), chr.negate)

    @property
    def has_references(self) -> bool:
        return False

    @classmethod
    def class_rank(cls) -> int:
        return 1

    def less_than_same_type(self, other: 'LiteralConstraint') -> bool:
        if self.negate != other.negate:
            return self.negate < other.negate
        return len(self.chars) < len(other.chars)

    @property
    def lit_component(self) -> 'LiteralConstraint':
        return self

    @property
    def ref_component(self) -> 'RefConstraint':
        return empty_ref_constraint


all_literal_constraint = LiteralConstraint(frozenset([]), True)


@dataclass(frozen=True, repr=False)
class RefConstraint(Constraint):
    indices: FrozenSet[cell_ix_type]

    def intersection(self, other: Constraint) -> Constraint:
        if other is any_constraint:
            return self
        if isinstance(other, LiteralConstraint):
            return CompoundConstraint(other, self)
        if isinstance(other, CompoundConstraint):
            ref = other.ref_component.intersection(self)
            assert isinstance(ref, RefConstraint)
            return CompoundConstraint(other.lit_component, ref)

        assert isinstance(other, RefConstraint)
        return RefConstraint(self.indices | other.indices)

    def label(self) -> str:
        return ', '.join('%d,%d' % x for x in sorted(self.indices))

    @property
    def has_references(self) -> bool:
        return True

    @classmethod
    def class_rank(cls) -> int:
        return 2

    def less_than_same_type(self, other: 'RefConstraint') -> bool:
        return len(self.indices) < len(other.indices)

    @property
    def lit_component(self) -> 'LiteralConstraint':
        return all_literal_constraint

    @property
    def ref_component(self) -> 'RefConstraint':
        return self


empty_ref_constraint = RefConstraint(frozenset([]))


@dataclass(frozen=True, repr=False)
class CompoundConstraint(Constraint):
    lit_constraint: LiteralConstraint
    ref_constraint: RefConstraint

    def intersection(self, other: Constraint) -> Optional[Constraint]:
        if not isinstance(other, CompoundConstraint):
            return other.intersection(self)
        lit = self.lit_component.intersection(other.lit_component)
        if lit is None:
            return None
        assert isinstance(lit, LiteralConstraint)
        ref = self.ref_component.intersection(other.ref_component)
        assert isinstance(ref, RefConstraint)
        return CompoundConstraint(lit, ref)

    def label(self) -> str:
        return self.lit_component.label() + ' : ' + self.ref_component.label()

    @property
    def has_references(self) -> bool:
        return True

    @classmethod
    def class_rank(cls) -> int:
        return 3

    def less_than_same_type(self, other: 'CompoundConstraint') -> bool:
        return (self.lit_component.less_than_same_type(other.lit_component) and
                self.ref_component.less_than_same_type(other.ref_component))

    @property
    def lit_component(self) -> 'LiteralConstraint':
        return self.lit_constraint

    @property
    def ref_component(self) -> 'RefConstraint':
        return self.ref_constraint

    @property
    def compound_constraint(self) -> 'CompoundConstraint':
        return self


class Solution:
    cells: Dict[cell_ix_type, Constraint]

    def __init__(self, cells: Dict[cell_ix_type, Constraint]):
        self.cells = cells

    @classmethod
    def for_chr_seq(cls, string: String, chr_seq: ChrSeq) -> 'Solution':
        assert string.size == len(chr_seq)
        cells: Dict[cell_ix_type, Constraint] = {}
        for i, position in enumerate(string.positions):
            assert position.index == i
            ix = position.cell.index
            chr = chr_seq[i]
            if isinstance(chr, ReAny):
                constraint = any_constraint
            elif isinstance(chr, ReLit):
                constraint = LiteralConstraint.for_re_lit(chr)
            elif isinstance(chr, ChrRef):
                ref_index = string.positions[chr.index].cell.index
                constraint = RefConstraint(frozenset([ref_index]))
            else:
                raise TypeError(f'{chr!r}')
            cells[ix] = constraint
        return cls(cells)

    @classmethod
    def generate_solutions(cls, string: String) -> Iterable['Solution']:
        for match in string.gen_possible():
            yield cls.for_chr_seq(string, match.chr_seq)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} cells={len(self.cells)}>'

    def intersection(self, other: 'Solution') -> Optional['Solution']:
        ixs_both, ixs_o, ixs_s = self.compute_intersection_sets(other)
        return self._intersection(ixs_both, ixs_o, ixs_s, other)

    def can_intersect(self, other: 'Solution') -> bool:
        ixs_both, ixs_o, ixs_s = self.compute_intersection_sets(other)
        if not ixs_both:
            return True
        return self._intersection(ixs_both, ixs_o, ixs_s, other) is not None

    def compute_intersection_sets(self, other: 'Solution'):
        assert self is not other
        ixs_s = frozenset(self.cells)
        ixs_o = frozenset(other.cells)
        ixs_both = ixs_s & ixs_o
        return ixs_both, ixs_o, ixs_s

    def _intersection(self, ixs_both, ixs_o, ixs_s, other) -> Optional['Solution']:
        cells = {}
        for ix in ixs_both:
            c = self.cells[ix].intersection(other.cells[ix])
            if c is None:
                return None
            cells[ix] = c

        def add_unique(ixs: FrozenSet[cell_ix_type], source: Dict[cell_ix_type, Constraint]):
            for ix in ixs - ixs_both:
                cells[ix] = source[ix]

        add_unique(ixs_s, self.cells)
        add_unique(ixs_o, other.cells)

        @lru_cache(100)
        def resolve_compound(cell_ix: cell_ix_type) -> Optional[CompoundConstraint]:
            cc = cells[cell_ix]
            acc = cc
            for ref_ix in cc.ref_component.indices:
                ref_c = cells[ref_ix]
                if isinstance(ref_c, CompoundConstraint):
                    ref_c = resolve_compound(ref_ix)
                    if ref_c is None:
                        return None
                acc = acc.intersection(ref_c)
                if acc is None:
                    return None
                assert isinstance(acc, CompoundConstraint)
            return acc

        for k, c in cells.items():
            if isinstance(c, CompoundConstraint):
                r = resolve_compound(k)
                if r is None:
                    return None
                cells[k] = r
        return Solution(cells)


@dataclass(frozen=True)
class OptionallySizedIterable(Iterable[T]):
    iterable: Iterable[T]
    max_size: Optional[int]
    known_size: Optional[int]

    @classmethod
    def of_known_size(cls, iterable: Iterable[T], n: int) -> 'OptionallySizedIterable[T]':
        return cls(iterable, n, n)

    @classmethod
    def of_bound_size(cls, iterable: Iterable[T], max_size: int) -> 'OptionallySizedIterable[T]':
        return cls(iterable, max_size, None)

    @classmethod
    def of_unbound(cls, iterable: Iterable[T]) -> 'OptionallySizedIterable[T]':
        return cls(iterable, None, None)

    @classmethod
    def of(cls, iterable: Union['OptionallySizedIterable[T]', Iterable[T]]) -> 'OptionallySizedIterable[T]':
        if isinstance(iterable, OptionallySizedIterable):
            return iterable
        if isinstance(iterable, Collection):
            return cls.of_known_size(iterable, len(iterable))
        assert isinstance(iterable, Iterable)
        return cls.of_unbound(iterable)

    @classmethod
    def of_cartesian_product(cls, iterable: Iterable[T], x: 'OptionallySizedIterable', y: 'OptionallySizedIterable'
                             ) -> 'OptionallySizedIterable[T]':
        if x.has_known_size and y.has_known_size:
            return cls.of_known_size(iterable, x.known_size * y.known_size)
        if x.is_bound and y.is_bound:
            return cls.of_bound_size(iterable, x.max_size * y.max_size)
        return cls.of_unbound(iterable)

    def __iter__(self) -> Iterator[T]:
        return iter(self.iterable)

    @property
    def is_bound(self) -> bool:
        return self.max_size is not None

    @property
    def is_unbound(self) -> bool:
        return self.max_size is None

    @property
    def has_known_size(self) -> bool:
        return self.known_size is not None

    @property
    def len_str(self) -> str:
        if self.is_unbound:
            return 'unbound'
        if self.has_known_size:
            return f'={self._len_str(self.known_size)}'
        return f'<={self._len_str(self.max_size)}'

    @staticmethod
    def _len_str(n: int) -> str:
        return f'{n:,}' if n < 1e9 else f'{n:.1e}'

    def __str__(self):
        return f'seq {self.len_str}'

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.len_str}>'

    def as_sequence(self) -> Sequence[T]:
        assert self.is_bound
        return self.iterable if isinstance(self.iterable, Sequence) else tuple(self.iterable)

    def map(self, func: Callable[[T], U]) -> 'OptionallySizedIterable[U]':
        return OptionallySizedIterable(map(func, self.iterable), self.max_size, self.known_size)

    def filter(self, func: Callable[[T], bool]) -> 'OptionallySizedIterable[T]':
        return OptionallySizedIterable(filter(func, self.iterable), self.max_size, None)


class SolutionSet(Collection[Solution]):
    names: FrozenSet[str]
    cell_indices: FrozenSet[cell_ix_type]
    solutions: OptionallySizedIterable[Solution]

    def __init__(self,
                 names: FrozenSet[str],
                 cell_indices: FrozenSet[cell_ix_type],
                 solutions: Union[Iterable[Solution], OptionallySizedIterable[Solution]]):
        self.names = names
        self.cell_indices = cell_indices
        self.solutions = OptionallySizedIterable.of(solutions)

    @property
    def names_str(self) -> str:
        return ', '.join(sorted(self.names))

    def contains(self, other: 'SolutionSet') -> bool:
        return self.names >= other.names

    def __str__(self) -> str:
        return f'{self.names_str} cells={len(self.cell_indices)} (sols {self.solutions.len_str})'

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {str(self)}>'

    def __len__(self) -> int:
        assert self.solutions.has_known_size
        return self.solutions.known_size

    def __iter__(self) -> Iterator[Solution]:
        return iter(self.solutions)

    def __contains__(self, x: object) -> bool:
        raise NotImplementedError()

    def intersection(self, other: 'SolutionSet') -> 'SolutionSet':
        if self.contains(other):
            return self
        if other.contains(self):
            return other

        assert self.solutions.is_bound
        assert other.solutions.is_bound

        solutions = []
        for a in self:
            for b in other:
                ab = a.intersection(b)
                if ab is not None:
                    solutions.append(ab)

        return self.build_intersection(other, solutions)

    def build_intersection(self, other: 'SolutionSet',
                           solutions: Union[Iterable[Solution], OptionallySizedIterable[Solution]]):
        return SolutionSet(self.names | other.names, self.cell_indices | other.cell_indices, solutions)

    def cell_intersection(self, other: 'SolutionSet') -> FrozenSet[cell_ix_type]:
        return self.cell_indices & other.cell_indices

    def cell_intersection_frac(self, other: 'SolutionSet') -> float:
        return len(self.cell_intersection(other)) / max(len(self), len(other))

    def estimate_intersection_size(self, other: 'SolutionSet', random: Random, sample_size=50,
                                   rate_floor_clamp=0.005) -> float:
        if not self.cell_intersection(other):
            return len(self) * len(other)

        def sample(ss: SolutionSet) -> Collection[Solution]:
            if len(ss) < sample_size:
                return ss
            shuffled = list(ss.solutions)
            random.shuffle(shuffled)
            return shuffled[:sample_size:]

        sample_a = sample(self)
        sample_b = sample(other)
        intersection_count = sum(1 for a in sample_a for b in sample_b if a.can_intersect(b))
        intersection_rate = intersection_count / (len(sample_b) * len(sample_b))
        intersection_rate = max(intersection_rate, rate_floor_clamp)
        return intersection_rate * (len(self) * len(other))

    @classmethod
    def for_string(cls, s: String, random: Random):
        solutions = list(Solution.generate_solutions(s))
        random.shuffle(solutions)
        return cls(frozenset([s.name]), frozenset(p.cell.index for p in s.positions), solutions)


def select_smallest_solution_set(solution_sets: Iterable[SolutionSet]) -> SolutionSet:
    smallest: Optional[SolutionSet] = None
    for solution_set in solution_sets:
        if len(solution_set) <= 1:
            return solution_set
        if smallest is None or len(solution_set) < len(smallest):
            smallest = solution_set
    return smallest


def drop_redundant_solution_sets(solution_sets: Iterable[SolutionSet]) -> Collection[SolutionSet]:
    unique_solution_sets: Iterable[SolutionSet] = {s.names: s for s in solution_sets}.values()
    return [s for s in unique_solution_sets
            if not any(o.contains(s) for o in unique_solution_sets if o is not s)]


class SolutionSetIntersectionCache:
    cache: Dict[FrozenSet[str], SolutionSet]

    def __init__(self, existing: Iterable[SolutionSet] = ()):
        self.cache = {s.names: s for s in existing}

    def intersection(self, a: SolutionSet, b: SolutionSet) -> SolutionSet:
        key = a.names | b.names
        result = self.cache.get(key)
        if result is None:
            result = self.cache[key] = a.intersection(b)
        return result


def merge_cross_axes_solution_sets(axes_solution_sets: Dict[str, Collection[SolutionSet]]):
    axes = frozenset(axes_solution_sets)
    intersection = SolutionSetIntersectionCache().intersection
    intersections = []
    with tqdm(total=sum(map(len, axes_solution_sets.values()))) as progress:
        for axis, axis_solutions in axes_solution_sets.items():
            other_axes_solutions = [other_solution for other_axis in axes - {axis}
                                    for other_solution in axes_solution_sets[other_axis]]
            for axis_solution in axis_solutions:
                progress.set_description(f'merge cross axis {axis_solution}')
                progress.update()
                intersections.append(select_smallest_solution_set(intersection(axis_solution, p)
                                                                  for p in other_axes_solutions))
    return drop_redundant_solution_sets(intersections)


def reduce_solution_sets_using_best_pair_intersections(solution_sets: Collection[SolutionSet]
                                                       ) -> Collection[SolutionSet]:
    solution_sets = sorted(solution_sets, key=len)
    intersection = SolutionSetIntersectionCache(solution_sets).intersection
    intersections = []
    with tqdm(solution_sets, desc='reduce_best_pair_intersect') as progress:
        for solution_set in progress:
            solution_set: SolutionSet = solution_set

            def gen_intersections() -> Iterable[SolutionSet]:
                for other in solution_sets:
                    if other is not solution_set:
                        progress.set_description(f'reduce {solution_set} & {other}')
                        yield intersection(solution_set, other)

            intersections.append(select_smallest_solution_set(gen_intersections()))
    return drop_redundant_solution_sets(intersections)


def reduce_solution_sets_by_smallest_pair(solution_sets: Collection[SolutionSet]) -> Collection[SolutionSet]:
    if len(solution_sets) <= 1:
        return solution_sets
    solution_sets = sorted(solution_sets, key=len, reverse=True)
    a = solution_sets.pop(-1)
    b = solution_sets.pop(-1)
    solution_sets.append(a.intersection(b))
    return solution_sets


def reduce_solution_sets_by_highest_overlap(solution_sets: Collection[SolutionSet]) -> Collection[SolutionSet]:
    if len(solution_sets) <= 1:
        return solution_sets
    solution_sets = list(solution_sets)
    n = len(solution_sets)
    indexes = ((i, j) for i in range(n) for j in range(i + 1, n))
    i, j = max(indexes, key=lambda x: solution_sets[x[0]].cell_intersection_frac(solution_sets[x[1]]))
    assert j > i  # pop the largest index first so smaller is still valid
    a = solution_sets.pop(j)
    b = solution_sets.pop(i)
    solution_sets.append(a.intersection(b))
    return solution_sets


def reduce_solution_sets_by_lowest_estimated_count(solution_sets: Collection[SolutionSet], random: Random
                                                   ) -> Collection[SolutionSet]:
    if len(solution_sets) <= 1:
        return solution_sets
    solution_sets = list(solution_sets)
    n = len(solution_sets)

    def gen_indexes():
        with tqdm(total=n * (n - 1) // 2) as progress:
            for i in range(n):
                for j in range(i + 1, n):
                    progress.set_description(f'estimating intersection ({i},{j})', refresh=False)
                    progress.update()
                    yield i, j

    i, j = min(gen_indexes(), key=lambda x: solution_sets[x[0]].estimate_intersection_size(solution_sets[x[1]], random))
    assert j > i  # pop the largest index first so smaller is still valid
    a = solution_sets.pop(j)
    b = solution_sets.pop(i)
    solution_sets.append(a.intersection(b))
    return solution_sets


def multi_reduce_solution_sets_by_lowest_estimated_count(solution_sets: Collection[SolutionSet], random: Random,
                                                         max_reduce_size: int = 2000
                                                         ) -> Collection[SolutionSet]:
    too_large: List[SolutionSet] = []

    def filter_to_large(sss: Collection[SolutionSet]) -> List[SolutionSet]:
        too_large.extend(ss for ss in sss if len(ss) > max_reduce_size)
        return [ss for ss in sss if len(ss) <= max_reduce_size]

    solution_sets = filter_to_large(solution_sets)
    while len(solution_sets) > 1:
        solution_sets = reduce_solution_sets_by_lowest_estimated_count(solution_sets, random)
        solution_sets = filter_to_large(solution_sets)

    return solution_sets + too_large


def iter_sequence_pair_indices(n: int) -> OptionallySizedIterable[Tuple[int, int]]:
    def gen(n=n) -> Iterable[Tuple[int, int]]:
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j

    return OptionallySizedIterable.of_known_size(gen(), n * (n - 1) // 2)


def iter_sequence_pairs(xs: Sequence[T]) -> OptionallySizedIterable[Tuple[T, T]]:
    return iter_sequence_pair_indices(len(xs)).map(lambda p: (xs[p[0]], xs[p[1]]))


def iter_pairs_randomly_with_replacement(xs: Sequence[T], ys: Sequence[U],
                                         random_state: Optional[Union[int, Random]] = None
                                         ) -> OptionallySizedIterable[Tuple[T, U]]:
    random: Random = random_state if isinstance(random_state, Random) else Random(random_state)

    def gen(xs=xs, ys=ys) -> Iterable[Tuple[T, U]]:
        nx = len(xs)
        ny = len(ys)
        randint = random.randint
        while True:
            yield xs[randint(0, nx)], ys[randint(0, ny)]

    return OptionallySizedIterable.of_unbound(gen())


def iter_pairs_randomly_wo_replacement(xs: Sequence[T], ys: Sequence[U],
                                       random_state: Optional[Union[int, Random]] = None
                                       ) -> OptionallySizedIterable[Tuple[T, U]]:
    if isinstance(random_state, Random):
        random_state = random_state.randrange(0, 0xFFFFFFFF)
    random = np_random.RandomState(random_state)
    nx = len(xs)
    n = nx * len(ys)
    indices = np.arange(0, n, dtype=np.int32)
    random.shuffle(indices)

    def gen(xs=xs, ys=ys) -> Iterable[Tuple[T, U]]:
        for inx in indices:
            j, i = divmod(inx, nx)
            yield xs[i], ys[j]

    return OptionallySizedIterable.of_known_size(gen(), n)


def iter_pairs_randomly(xs: Sequence[T], ys: Sequence[U],
                        random_state: Optional[Union[int, Random]] = None) -> OptionallySizedIterable[Tuple[T, U]]:
    n = len(xs) * len(ys)
    if n > 10_000_000:
        return OptionallySizedIterable.of_unbound(
            iter_pairs_randomly_with_replacement(xs, ys, random_state))
    else:
        return iter_pairs_randomly_wo_replacement(xs, ys, random_state)


def stochastically_merge_solution_set_pair_bound(
        x: SolutionSet, y: SolutionSet, random: Random,
        callback: Callable[[str, str, Optional[Solution]], Any]) -> SolutionSet:
    pairs = iter_pairs_randomly_wo_replacement(x.solutions.as_sequence(), y.solutions.as_sequence(),
                                               random_state=random)
    assert pairs.is_bound, f'unbound pairs for {x} and {y}'

    nx = str(x)
    ny = str(y)

    def mapper(pair: Tuple[Solution, Solution]) -> Optional[Solution]:
        xi, yi = pair
        xy = xi.intersection(yi)
        callback(nx, ny, xy)
        return xy

    lazy_intersections = pairs.map(mapper).filter(lambda s: s is not None)
    return x.build_intersection(y, lazy_intersections)


class RepeatableLazy(Iterable[T]):
    source: Iterator[T]
    realized: List[T]
    is_realized: bool = False

    def __init__(self, source: Iterator[T]):
        self.source = source
        self.realized = []

    def __iter__(self) -> Iterator[T]:
        if self.is_realized:
            return iter(self.realized)
        return iter(self.realize())

    def realize(self) -> Iterable[T]:
        sentinel = object()
        while True:
            n = next(self.source, sentinel)
            if n is sentinel:
                self.is_realized = True
                break
            yield n
            self.realized.append(n)


def merge_solution_set_pair_unbound(x: SolutionSet, y: SolutionSet,
                                    callback: Callable[[str, str, Optional[Solution]], Any]) -> SolutionSet:
    xs = x.solutions
    ys = y.solutions
    if xs.is_bound and (not ys.is_bound or xs.max_size < ys.max_size):
        x, y = y, x
        xs, ys = ys, xs
    assert ys.is_bound, f'inner iterable is unbound for {y}'

    def gen() -> Iterable[Solution]:
        repeatable_ys = RepeatableLazy(iter(ys))
        nx = str(x)
        ny = str(y)
        for xi in xs:
            for yi in repeatable_ys:
                xy = xi.intersection(yi)
                callback(nx, ny, xy)
                if xy is not None:
                    yield xy

    lazy_intersections = (OptionallySizedIterable.of_bound_size(gen(), xs.max_size * ys.max_size)
                          if xs.is_bound and ys.is_bound else
                          OptionallySizedIterable.of_unbound(gen()))
    return x.build_intersection(y, lazy_intersections)


def stochastically_merge_solution_set_pair(
        x: SolutionSet, y: SolutionSet, random: Random,
        max_known_size: int, max_bound_size: int,
        callback: Callable[[str, str, Optional[Solution]], Any]) -> SolutionSet:
    if x.contains(y):
        return x
    if y.contains(x):
        return y
    xs = x.solutions
    ys = y.solutions
    if ((xs.has_known_size and ys.has_known_size and xs.known_size * ys.known_size < max_known_size) or
            (xs.is_bound and ys.is_bound and xs.max_size * ys.max_size < max_bound_size)):
        return stochastically_merge_solution_set_pair_bound(x, y, random, callback)
    return merge_solution_set_pair_unbound(x, y, callback)


def stochastically_merge_solution_sets(solution_sets: Collection[SolutionSet], random: Random,
                                       callback: Callable[[str, str, Optional[Solution]], Any],
                                       max_known_size: int = 1_000_000,
                                       max_bound_size: int = 200_000) -> SolutionSet:
    return reduce(lambda x, y: stochastically_merge_solution_set_pair(x, y, random,
                                                                      max_known_size, max_bound_size, callback),
                  sorted(solution_sets, key=len))


@dataclass(frozen=True, order=True)
class Dimension:
    axis: str
    num: int
    size: int

    def __str__(self):
        return f'{self.axis}{self.num}({self.size})'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self}>'


class DimensionSolutions(ABC):

    @abstractmethod
    def size(self, dimension: Dimension) -> int:
        pass

    @abstractmethod
    def points(self, dimension: Dimension) -> Iterable[int]:
        pass

    @abstractmethod
    def intersection(self, other: 'DimensionSolutions') -> 'DimensionSolutions':
        pass

    @abstractmethod
    def union(self, other: 'DimensionSolutions') -> 'DimensionSolutions':
        pass

    @abstractmethod
    def optimize(self, dimension: Dimension) -> 'DimensionSolutions':
        pass

    @staticmethod
    def for_dimension(dim: Dimension, indices: Collection[int]):
        if len(indices) == dim.size:
            return dimension_all
        if len(indices) > dim.size // 2:
            return DimensionSolutionIndexes(frozenset(range(dim.size)) - frozenset(indices), negate=True)
        else:
            return DimensionSolutionIndexes(frozenset(indices), negate=False)


class DimensionSolutionAll(DimensionSolutions):
    _instance: Optional['DimensionSolutionAll'] = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    def __str__(self):
        return 'all'

    def intersection(self, other: DimensionSolutions) -> DimensionSolutions:
        return other

    def union(self, other: DimensionSolutions) -> DimensionSolutions:
        return self

    def size(self, dimension: Dimension) -> int:
        return dimension.size

    def points(self, dimension: Dimension) -> Iterable[int]:
        return range(dimension.size)

    def optimize(self, dimension: Dimension) -> DimensionSolutions:
        return self


dimension_all: DimensionSolutions = DimensionSolutionAll()


class DimensionSolutionIndexes(DimensionSolutions):
    indexes: FrozenSet[int]
    negate: bool

    def __init__(self, solutions: FrozenSet[int], negate: bool = False):
        self.indexes = solutions
        self.negate = negate

    def __str__(self):
        return f'{"-" if self.negate else "+"}{len(self.indexes)}'

    def __repr__(self):
        return f'<{self.__class__.__name__} sz={self}>'

    def intersection(self, other: DimensionSolutions) -> DimensionSolutions:
        if other is dimension_all:
            return self
        assert isinstance(other, DimensionSolutionIndexes)
        if self.negate == other.negate:
            return DimensionSolutionIndexes(
                self.indexes | other.indexes if self.negate else self.indexes & other.indexes,
                self.negate)
        pos, neg = self, other
        if pos.negate:
            pos, neg = neg, pos
        return DimensionSolutionIndexes(pos.indexes - neg.indexes, False)

    def union(self, other: DimensionSolutions) -> DimensionSolutions:
        if other is dimension_all:
            return self
        assert isinstance(other, DimensionSolutionIndexes)
        if self.negate == other.negate:
            return DimensionSolutionIndexes(
                self.indexes & other.indexes if self.negate else self.indexes | other.indexes,
                self.negate)
        pos, neg = self, other
        if pos.negate:
            pos, neg = neg, pos
        return DimensionSolutionIndexes(neg.indexes - pos.indexes, True)

    def size(self, dimension: Dimension) -> int:
        return dimension.size - len(self.indexes) if self.negate else len(self.indexes)

    def points(self, dimension: Dimension) -> Iterable[int]:
        if self.negate:
            return frozenset(range(dimension.size)) - self.indexes
        return self.indexes

    def optimize(self, dim: Dimension) -> DimensionSolutions:
        if len(self.indexes) == dim.size:
            return dimension_all
        if len(self.indexes) > dim.size // 2:
            return DimensionSolutionIndexes(frozenset(range(dim.size)) - self.indexes, negate=not self.negate)
        return self


class DimensionCombination(Collection[Tuple[Dimension, DimensionSolutions]]):
    dimensions: Mapping[Dimension, DimensionSolutions]

    def __init__(self, dimensions: Mapping[Dimension, DimensionSolutions]):
        self.dimensions = dimensions

    def __len__(self) -> int:
        return len(self.dimensions)

    def __iter__(self) -> Iterator[Tuple[Dimension, DimensionSolutions]]:
        return iter(self.dimensions.items())

    def __contains__(self, x: object) -> bool:
        return x in self.dimensions.items()

    def size(self) -> int:
        if not self.dimensions:
            return 0
        sz = 1
        for dim, cell in self.dimensions.items():
            c_sz = cell.size(dim)
            if c_sz == 0:
                return 0
            sz *= c_sz
        return sz

    def intersection(self, other: 'DimensionCombination') -> 'DimensionCombination':
        return self._merge(other, lambda a, b: a.intersection(b))

    def union(self, other: 'DimensionCombination') -> 'DimensionCombination':
        return self._merge(other, lambda a, b: a.union(b))

    def _merge(self, other: 'DimensionCombination',
               merger: Callable[[DimensionSolutions, DimensionSolutions], DimensionSolutions]
               ) -> 'DimensionCombination':
        dimensions: Dict[Dimension, DimensionSolutions] = {}
        common = self.dimensions.keys() & other.dimensions.keys()
        for dim in common:
            dimensions[dim] = merger(self.dimensions[dim], other.dimensions[dim])
        for dim in self.dimensions.keys() - common:
            dimensions[dim] = self.dimensions[dim]
        for dim in other.dimensions.keys() - common:
            dimensions[dim] = other.dimensions[dim]
        return DimensionCombination(dimensions)

    @classmethod
    def for_dimension(cls, dim: Dimension, indices: Collection[int]):
        return cls({dim: DimensionSolutions.for_dimension(dim, indices)})

    def optimize(self) -> 'DimensionCombination':
        return DimensionCombination({dim: sol.optimize(dim) for dim, sol in self.dimensions.items()})

    def add_dimensions_all(self, dims: FrozenSet[Dimension]) -> 'DimensionCombination':
        cp = dict(self.dimensions)
        for dim in dims:
            assert dim not in cp, f'{dim} in {cp}'
            cp[dim] = dimension_all
        return DimensionCombination(cp)

    def iter_points(self) -> Iterable['DimensionCombination']:

        def rec(remaining: Iterator[Tuple[Dimension, DimensionSolutions]],
                previous: Iterable[Mapping[Dimension, DimensionSolutions]]) -> Iterable[DimensionCombination]:
            n = next(remaining, None)
            if n is None:
                for m in previous:
                    yield DimensionCombination(m)
                return

            def gen() -> Iterable[Mapping[Dimension, DimensionSolutions]]:
                dim, sols = n
                for p in previous:
                    for i in sols.points(dim):
                        c = dict(p)
                        c[dim] = DimensionSolutionIndexes(frozenset([i]), negate=False)
                        yield c

            yield from rec(remaining, gen())

        return rec(iter(self.dimensions.items()), [{}])


class ConstraintDimensionCombination(Collection[Tuple[Constraint, DimensionCombination]]):
    constraint_dimensions = Dict[Constraint, DimensionCombination]

    def __init__(self):
        self.constraint_dimensions = {}

    def __len__(self) -> int:
        return len(self.constraint_dimensions)

    def __iter__(self) -> Iterator[Tuple[Constraint, DimensionCombination]]:
        return iter(self.constraint_dimensions.items())

    def __contains__(self, x: object) -> bool:
        return x in self.constraint_dimensions.items()

    def add(self, constraint: Constraint, combination: DimensionCombination):
        existing = self.constraint_dimensions.get(constraint)
        if existing is not None:
            combination = combination.union(existing)
        self.constraint_dimensions[constraint] = combination

    def add_all(self, other: 'ConstraintDimensionCombination'):
        assert self is not other
        for con, comb in other.items():
            self.add(con, comb)

    def items(self) -> Sequence[Tuple[Constraint, DimensionCombination]]:
        return tuple((con, comb) for con, comb in self.constraint_dimensions.items() if comb.size())

    def __bool__(self):
        return any(c.size() for c in self.constraint_dimensions.values())


class CellConstraints(Collection[Tuple[Constraint, DimensionCombination]]):
    constraint_dimensions: Sequence[Tuple[Constraint, DimensionCombination]]

    def __init__(self, constraints: Sequence[Tuple[Constraint, DimensionCombination]]):
        self.constraint_dimensions = constraints

    def __len__(self) -> int:
        return len(self.constraint_dimensions)

    def __iter__(self) -> Iterator[Tuple[Constraint, DimensionCombination]]:
        return iter(self.constraint_dimensions)

    def __contains__(self, x: object) -> bool:
        return x in self.constraint_dimensions

    def constraints(self) -> Collection[Constraint]:
        return [con for con, comb in self.constraint_dimensions]

    def intersection(self, other: 'CellConstraints') -> Optional['CellConstraints']:
        int_combs = ConstraintDimensionCombination()
        for con_a, comb_a in self.constraint_dimensions:
            for con_b, comb_b in other.constraint_dimensions:
                con_ab = con_a.intersection(con_b)
                if con_ab is None:
                    continue
                comb_ab = comb_a.intersection(comb_b)
                if not comb_ab.size():
                    continue
                int_combs.add(con_ab, comb_ab)
        return None if not int_combs else CellConstraints(int_combs.items())

    def add_dimensions_all(self, dims: FrozenSet[Dimension]) -> 'CellConstraints':
        if not dims:
            return self
        return CellConstraints(tuple((con, comb.add_dimensions_all(dims))
                                     for con, comb in self.constraint_dimensions))

    @property
    def has_references(self) -> bool:
        return any(c.has_references for c, _ in self.constraint_dimensions)

    def references(self) -> Set[cell_ix_type]:
        refs = set()
        for c, _ in self.constraint_dimensions:
            if c.has_references:
                refs.update(c.ref_component.indices)
        return refs

    def apply_references(self, cells: Mapping[cell_ix_type, 'CellConstraints']) -> Optional['CellConstraints']:
        applied = ConstraintDimensionCombination()
        for con, comb in self.constraint_dimensions:
            if not con.has_references:
                applied.add(con, comb)
                continue

            ref_con: RefConstraint
            lit_con: Optional[LiteralConstraint]
            if isinstance(con, RefConstraint):
                ref_con = con
                lit_con = None
            else:
                assert isinstance(con, CompoundConstraint)
                ref_con = con.ref_component
                lit_con = con.lit_component
            applied.add_all(self.resolve_references(cells, comb, ref_con, lit_con))

        return None if not applied else CellConstraints(applied.items())

    @staticmethod
    def resolve_references(cells: Mapping[cell_ix_type, 'CellConstraints'],
                           comb: DimensionCombination, ref_con: RefConstraint,
                           lit_con: Optional[LiteralConstraint]) -> ConstraintDimensionCombination:
        int_combs = ConstraintDimensionCombination()
        for ix in ref_con.indices:
            for refed_con, refed_comb in cells[ix].constraint_dimensions:
                if lit_con is not None:
                    refed_con = refed_con.intersection(lit_con)
                    if refed_con is None:
                        continue
                ref_int = ref_con.intersection(refed_con)
                assert ref_int is not None
                int_combs.add(ref_int, refed_comb.intersection(comb))
        return int_combs

    def optimize(self) -> 'CellConstraints':
        return CellConstraints(tuple((con, comb.optimize()) for con, comb in self.constraint_dimensions))

    def union_dimensions(self) -> DimensionCombination:
        return reduce(lambda a, b: a.union(b), (comb for _, comb in self.constraint_dimensions))

    def intersection_dim_combs(self, comb: DimensionCombination) -> 'CellConstraints':
        constraint_dimensions = []
        for con_i, comb_i in self.constraint_dimensions:
            comb_i = comb_i.intersection(comb)
            if comb_i.size():
                constraint_dimensions.append((con_i, comb_i))
        assert constraint_dimensions, f'no matching dimension combinations'
        return CellConstraints(constraint_dimensions)

    def intersection_constraint(self, con: Constraint) -> 'CellConstraints':
        constraint_dimensions = []
        for con_i, comb_i in self.constraint_dimensions:
            con_i = con_i.intersection(con)
            if con_i is not None:
                constraint_dimensions.append((con_i, comb_i))
        assert constraint_dimensions, f'no matching dimension combinations'
        return CellConstraints(constraint_dimensions)

    def point_constraint(self, point: DimensionCombination) -> Optional[Constraint]:
        point_constraint: Optional[Constraint] = None
        for con_i, comb_i in self.constraint_dimensions:
            comb_i = comb_i.intersection(point)
            sz = comb_i.size()
            assert sz <= 1, f'{sz}'
            if sz:
                if point_constraint is not None:
                    point_constraint = point_constraint.intersection(con_i)
                    if point_constraint is None:
                        return None
                else:
                    point_constraint = con_i
        return point_constraint

    @classmethod
    def for_dimension(cls, dim: Dimension, constraints: Dict[Constraint, List[int]]):
        return cls(tuple((con, DimensionCombination.for_dimension(dim, indices))
                         for con, indices in constraints.items()))


class SparseSolutionSet:
    dimensions: FrozenSet[Dimension]
    cells: Mapping[cell_ix_type, CellConstraints]

    def __init__(self, dimensions: FrozenSet[Dimension],
                 cells: Mapping[cell_ix_type, CellConstraints]) -> None:
        self.dimensions = dimensions
        self.cells = cells

    @property
    def names_str(self) -> str:
        return ', '.join(sorted(self.dimensions))

    def contains(self, other: 'SparseSolutionSet') -> bool:
        return self.dimensions >= other.dimensions

    def intersection(self, other: 'SparseSolutionSet') -> Optional['SparseSolutionSet']:
        if self.contains(other):
            return self
        if other.contains(self):
            return other

        cells: Dict[cell_ix_type, CellConstraints] = {}
        common = self.cells.keys() & other.cells.keys()
        for ix in common:
            c = self.cells[ix].intersection(other.cells[ix])
            if c is None:
                return None
            cells[ix] = c
        dims_s = self.dimensions - other.dimensions
        dims_o = other.dimensions - self.dimensions
        for ix in self.cells.keys() - common:
            cells[ix] = self.cells[ix].add_dimensions_all(dims_o)
        for ix in other.cells.keys() - common:
            cells[ix] = other.cells[ix].add_dimensions_all(dims_s)

        @lru_cache(len(cells))
        def apply_constraints(cell_ix: cell_ix_type) -> Optional[CellConstraints]:
            cc = cells[cell_ix]
            assert cc.has_references
            for ref in cc.references():
                refed = cells[ref]
                if refed.has_references:
                    if apply_constraints(ref) is None:
                        return None

            return cc.apply_references(cells)

        for ix, c in cells.items():
            if c.has_references:
                ac = apply_constraints(ix)
                if ac is None:
                    return None
                cells[ix] = c

        return SparseSolutionSet(self.dimensions | other.dimensions, cells)

    def filter_cells_using_other_unions(self) -> Optional['SparseSolutionSet']:
        common_intersection = self.common_intersection()
        assert common_intersection.size() > 0
        return SparseSolutionSet(self.dimensions,
                                 {ix: c.intersection_dim_combs(common_intersection)
                                  for ix, c in self.cells.items()})

    def common_intersection(self) -> DimensionCombination:
        return reduce(lambda a, b: a.intersection(b),
                      (c.union_dimensions() for c in self.cells.values()))

    def push_reference_constraints(self) -> Optional['SparseSolutionSet']:
        cells = dict(self.cells)
        for ix, cell in self.cells.items():
            if not cell.has_references or len(cell) > 1:
                continue
            constraint, = cell.constraints()
            if isinstance(constraint, RefConstraint):
                continue
            assert isinstance(constraint, CompoundConstraint)
            for ref in constraint.ref_component.indices:
                cells[ref] = cells[ref].intersection_constraint(constraint.lit_component)
        return SparseSolutionSet(self.dimensions, cells)

    def iter_solutions(self) -> Iterable[Solution]:

        def build_solution(point: DimensionCombination) -> Optional[Solution]:
            cells: Dict[cell_ix_type, Constraint] = {}
            for ix, cell in self.cells.items():
                cell = cell.point_constraint(point)
                if cell is None:
                    return None
                cells[ix] = cell
            return Solution(cells)

        ci = self.common_intersection()
        with tqdm(self.common_intersection().iter_points(),
                  total=ci.size(), desc='iter_solutions') as progress:
            for p in progress:
                solution = build_solution(p)
                if solution is not None:
                    yield solution

    @classmethod
    def for_string(cls, s: String) -> 'SparseSolutionSet':
        solutions = list(Solution.generate_solutions(s))
        cells: Dict[cell_ix_type, Dict[Constraint, List[int]]] = defaultdict(lambda: defaultdict(list))
        for sol_i, solution in enumerate(solutions):
            for ix, con in solution.cells.items():
                cells[ix][con].append(sol_i)

        dim = Dimension(s.name[0], int(s.name[1::]), len(solutions))
        return cls(frozenset([dim]), {ix: CellConstraints.for_dimension(dim, cons)
                                      for ix, cons in cells.items()})


class PuzzleDrawer:
    ax: plt.Axes
    fontsize: int = 13
    font: str = 'DejaVu Sans Mono'

    v_scale: float = 1 / 1.154

    unit_hex: plt.Polygon = plt.Polygon(MarkerPath.unit_regular_polygon(6).vertices,
                                        edgecolor='k', linewidth=1, fill=True, facecolor='#ffefa8')
    unit_hex.set_transform(Affine2D().scale(0.577))

    def __init__(self,
                 ax: plt.Axes,
                 fontsize: Optional[int] = None,
                 font: Optional[str] = None):
        self.ax = ax
        if fontsize is not None:
            self.fontsize = fontsize
        if font is not None:
            self.font = font

    def position(self, i, j) -> Tuple[float, float]:
        x = j + (constants.size - row_size(i)) / 2.0
        y = i * self.v_scale
        return x, y

    def translate_hexagon(self, x, y) -> plt.Polygon:
        patch = copy.copy(self.unit_hex)
        patch.set_transform(patch.get_transform() + Affine2D().translate(x, y))
        return patch

    def create_hexagons(self) -> PatchCollection:
        patches = [self.translate_hexagon(*self.position(i, j))
                   for i in range(constants.size)
                   for j in range(row_size(i))]
        return PatchCollection(patches, match_original=True)

    def draw_hexagons(self):
        self.ax.add_collection(self.create_hexagons())

    def text(self, x, y, s, background_color='#dffcd7', fontsize=None, **kwds):
        self.ax.text(x, y, s,
                     fontsize=or_else(fontsize, self.fontsize),
                     fontproperties=dict(family=self.font),
                     bbox=dict(facecolor=background_color, edgecolor='k'),
                     **kwds)

    def texts(self, vs, pos, off=(0, 0), inv=False, align='left', rot=0, anchor=False, **kwds):
        for i, s in enumerate(vs):
            if inv:
                i = constants.size - i - 1
            x, y = pos(i)
            x += off[0]
            y += off[1]
            if anchor:
                kwds['rotation_mode'] = 'anchor'
            self.text(x, y, s, horizontalalignment=align, rotation=rot, **kwds)

    def draw_texts(self):
        self.texts(constants.x[:mid + 1:], pos=lambda i: self.position(12, i),
                   off=(0.1, 0.7), rot=60)
        self.texts(constants.x[mid + 1::], inv=True, pos=lambda i: self.position(i, row_size(i)),
                   off=(-0.2, -0.3), rot=60)

        self.texts(constants.y[:mid + 1:], inv=True, pos=lambda i: self.position(i, 0),
                   off=(-0.8, 0), align='right')
        self.texts(constants.y[mid + 1::], pos=lambda i: self.position(mid - i - 1, 0),
                   off=(-0.8, -0.3), align='right')

        self.texts(constants.z[:mid + 1:], pos=lambda i: self.position(0, i),
                   off=(0.15, -0.8), rot=-60, anchor=True)
        self.texts(constants.z[mid + 1::], pos=lambda i: self.position(i, row_size(i)),
                   off=(0, 0.3), rot=-60, anchor=True)

    def draw_arrow(self, start, end, color='r'):
        arrow = FancyArrowPatch(start, end,
                                connectionstyle="arc3,rad=.5",
                                shrinkA=0, shrinkB=0,
                                arrowstyle="Simple, tail_width=0.5, head_width=10, head_length=20",
                                color=color)
        self.ax.add_patch(arrow)

    def draw_solution(self, solution: Solution):
        for (i, j), con in solution.cells.items():
            x, y = self.position(i, j)
            self.text(x, y, con.lit_component.label(), background_color='#b5f1ff')

            for ref in con.ref_component.indices:
                self.draw_arrow((x, y), self.position(*ref))


def draw_puzzle(ax: Optional[plt.Axes] = None, fig_size=11, fontsize=13, font='DejaVu Sans Mono',
                solution: Optional[Solution] = None) -> PuzzleDrawer:
    if ax is None:
        f = plt.figure(figsize=(fig_size, fig_size))
        ax = f.add_subplot(111)

    drawer = PuzzleDrawer(ax, fontsize=fontsize, font=font)
    drawer.draw_hexagons()
    drawer.draw_texts()
    if solution is not None:
        drawer.draw_solution(solution)

    ax.set_xlim(-6, 17)
    ax.set_ylim(-6, 17)

    return drawer


def old_main():
    print('building strings')
    strings = build_strings()

    random = Random(0xCAFE)
    random.shuffle(strings)
    # strings = strings[:16]

    with tqdm(strings, desc='init solutions') as progress:
        solution_sets = [SolutionSet.for_string(s, random) for s in progress]

    solution_sets = multi_reduce_solution_sets_by_lowest_estimated_count(solution_sets, random)
    print(f'post estimation {len(solution_sets)} solutions')

    i = 0
    with tqdm() as progress:
        def callback(xi, yi, xy):
            nonlocal i
            i += 1
            if i % 10 == 0:
                progress.set_description(('no', '  ')[xy is not None] + f'match {xi} & {yi}', refresh=False)
                progress.update()

        solutions = stochastically_merge_solution_sets(solution_sets, random=random, callback=callback)
        print(solutions)
        solution = next(iter(solutions))

    print(solution)


def main():
    from pprint import pprint

    print('building strings')
    strings = build_strings()

    def show(sol: SparseSolutionSet):
        pprint({ix: [[con, comb.dimensions]
                     for con, comb in cc.constraint_dimensions]
                for ix, cc in sol.cells.items()})

    # a = SparseSolutionSet.for_string(strings[0])
    # b = SparseSolutionSet.for_string(strings[13])
    #
    # show(a)
    # show(b)
    # i = a.intersection(b)
    # show(i)

    random = Random(0xCAFE)
    random.shuffle(strings)
    # strings = strings[:20]

    with tqdm(strings, desc='init solutions') as progress:
        solution_sets = [SparseSolutionSet.for_string(s) for s in progress]

    s = reduce(lambda a, b: a.intersection(b), solution_sets)
    # show(s)

    with tqdm(range(10), desc='filter') as progress:
        for _ in progress:
            s = s.filter_cells_using_other_unions()
            s = s.push_reference_constraints()

    c = s.common_intersection()
    print(c.size())

    solutions = list(s.iter_solutions())
    print(len(solutions))


__name__ == '__main__' and main()
