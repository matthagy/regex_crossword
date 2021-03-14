import copy
import sre_constants
import sre_parse
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import (Any, Callable, Dict, List, Tuple, Optional, Collection, Iterable, Sequence, Union, overload,
                    TypeVar, FrozenSet, Set)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.markers import Path as MarkerPath
from matplotlib.transforms import Affine2D

import constants

size = constants.size
assert size % 2 == 1
mid = size // 2


def row_size(row_index: int) -> int:
    assert 0 <= row_index < size
    return mid + min(row_index + 1, size - row_index)


def draw_puzzel(ax=None, fig_size=11, fontsize=13, font='DejaVu Sans Mono'):
    if ax is None:
        f = plt.figure(figsize=(fig_size, fig_size))
        ax = f.add_subplot(111)

    unit_hex = plt.Polygon(MarkerPath.unit_regular_polygon(6).vertices,
                           edgecolor='k', linewidth=1, fill=True, facecolor='#ffefa8')
    unit_hex.set_transform(Affine2D().scale(0.577))

    vscale = 1 / 1.154

    def position(i, j):
        x = j + (constants.size - row_size(i)) / 2.0
        y = i * vscale
        return x, y

    def translate_hexagon(x, y):
        patch = copy.copy(unit_hex)
        patch.set_transform(patch.get_transform() + Affine2D().translate(x, y))
        return patch

    patches = [translate_hexagon(*position(i, j))
               for i in range(constants.size)
               for j in range(row_size(i))]

    p = PatchCollection(patches, match_original=True)

    ax.add_collection(p)

    def text(x, y, s, **kwds):
        ax.text(x, y, s,
                fontsize=fontsize,
                fontproperties=dict(family=font),
                bbox=dict(facecolor='#dffcd7', edgecolor='k'),
                **kwds)

    def texts(vs, pos, off=(0, 0), inv=False, align='left', rot=0, anchor=False, **kwds):
        for i, s in enumerate(vs):
            if inv:
                i = constants.size - i - 1
            x, y = pos(i)
            x += off[0]
            y += off[1]
            if anchor:
                kwds['rotation_mode'] = 'anchor'
            text(x, y, s, horizontalalignment=align, rotation=rot, **kwds)

    texts(constants.x[:mid + 1:], pos=lambda i: position(12, i), off=(0.1, 0.7), rot=60)
    texts(constants.x[mid + 1::], inv=True, pos=lambda i: position(i, row_size(i)), off=(-0.2, -0.3), rot=60)

    texts(constants.y[:mid + 1:], inv=True, pos=lambda i: position(i, 0), off=(-0.8, 0), align='right')
    texts(constants.y[mid + 1::], pos=lambda i: position(mid - i - 1, 0), off=(-0.8, -0.3), align='right')

    texts(constants.z[:mid + 1:], pos=lambda i: position(0, i), off=(0.15, -0.8), rot=-60, anchor=True)
    texts(constants.z[mid + 1::], pos=lambda i: position(i, row_size(i)), off=(0, 0.3), rot=-60, anchor=True)

    ax.set_xlim(-6, 17)
    ax.set_ylim(-6, 17)


T = TypeVar('T')


def or_else(x: Optional[T], e: T) -> T:
    return x if x is not None else e


@dataclass(frozen=True)
class ChrSeq(Sequence['ReChr']):
    chrs: Tuple['ReChr', ...]

    def add(self, other: 'ChrSeq') -> 'ChrSeq':
        offset = len(self)

        def offset_char(chr: ReChr) -> ReChr:
            if isinstance(chr, ChrRef):
                return chr.offset(offset)
            return chr

        return chr_seq(self.chrs + tuple(map(offset_char, other.chrs)))

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
        return repr(''.join(map(str, self.chrs)))

    def __repr__(self) -> str:
        return f'<CSeq {self!s}>'


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
    groups: GroupBindings = empty_group_bindings

    def copy(self,
             groups: Optional[GroupBindings] = None) -> 'MatchState':
        return MatchState(groups=or_else(groups, self.groups))

    def extend(self, other, offset: int) -> 'MatchState':
        return self.copy(groups=self.groups.merge(other.groups.offset(offset)))


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
        n = len(self.chr_seq)
        if not n:
            return other
        return self.copy(chr_seq=self.chr_seq.add(other.chr_seq),
                         state=self.state.extend(other.state, n))


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
    for p in previous:
        mx = max_len - len(p.chr_seq)
        if mx >= 0:  # repeat can have 0-length matches
            for match in re.gen_possible(0, mx, p.state):
                yield p.extend(match)


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
                            p = p.copy(state=p.state.copy(
                                p.state.groups.bind(self.index, GroupBinding(start=0, end=s))))
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
    needs_offset: bool = False

    def __str__(self):
        return f'<{self.index}>'

    def __repr__(self):
        return f'<ChrRef {self.index}>'

    def offset(self, n: int) -> 'ChrRef':
        # don't offset first time
        return ChrRef(self.index if not self.needs_offset else self.index + n, True)

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
        bindings = state.groups.get(self.group.index)
        n = bindings.end - bindings.start
        if min_len <= n <= max_len:
            chr_seq = ChrSeq(tuple(ChrRef(i) for i in range(bindings.start, bindings.end)))
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

    def make_string(s: str) -> String:
        s = String(Pattern(s, ReSeq.from_string(s)), [])
        strings.append(s)
        return s

    def pos(s: String, cell: Cell, index: int):
        assert 0 <= index < constants.size, f'{index} is invalid'
        cn = Position(index=index, string=s, cell=cell)
        s.positions.append(cn)
        cell.positions.append(cn)

    def add_horizontal():
        for i, s in enumerate(constants.y):
            s = make_string(s)
            for j, c in enumerate(cells[i]):
                pos(s, c, j)

    def add_diag(ss: List[str], reverse=False):
        for i, s in enumerate(ss):
            s = make_string(s)
            start = max(0, i - mid)
            end = min(i + mid + 1, constants.size)
            for j in range(start, end):
                cell = cells[j][i if j <= mid else i - (j - mid)]
                pos(s, cell, end - j - 1 if reverse else j - start)

    add_horizontal()
    add_diag(constants.x, reverse=True)
    add_diag(constants.z, reverse=False)

    for string in strings:
        string.positions.sort(key=lambda c: c.index)
        assert [c.index for c in string.positions] == list(range(len(string.positions)))

    cell_constraints_count = Counter(len(c.positions) for row in cells for c in row)
    assert cell_constraints_count == {3: sum(map(len, cells))}, f'{cell_constraints_count}'

    return strings


class Constraint(ABC):

    @classmethod
    def for_chr(cls, chr: ReChr):
        pass

    @abstractmethod
    def intersection(self, other: 'Constraint') -> Optional['Constraint']:
        pass


class AnyConstraint(Constraint):

    def __new__(cls) -> Any:
        global any_constraint
        try:
            return any_constraint
        except NameError:
            return super().__new__(cls)

    def intersection(self, other: Constraint) -> Constraint:
        return other


any_constraint = AnyConstraint()


@dataclass(frozen=True)
class LiteralConstraint(Constraint):
    chars: FrozenSet[str]
    negate: bool

    def intersection(self, other: Constraint) -> Optional[Constraint]:
        if other is any_constraint:
            return self
        if isinstance(other, RefConstraint):
            return CompoundConstraint(self, other)
        if isinstance(other, CompoundConstraint):
            lit = self.intersection(other.lit_constraint)
            if lit is None:
                return None
            assert isinstance(lit, LiteralConstraint)
            return CompoundConstraint(lit, other.ref_constraint)

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

    @classmethod
    def for_re_lit(cls, chr: ReLit) -> 'LiteralConstraint':
        return cls(frozenset(chr.chars), chr.negate)


@dataclass(frozen=True)
class RefConstraint(Constraint):
    indices: FrozenSet[cell_ix_type]

    def intersection(self, other: Constraint) -> Constraint:
        if other is any_constraint:
            return self
        if isinstance(other, LiteralConstraint):
            return CompoundConstraint(other, self)
        if isinstance(other, CompoundConstraint):
            ref = other.ref_constraint.intersection(self)
            assert isinstance(ref, RefConstraint)
            return CompoundConstraint(other.lit_constraint, ref)

        assert isinstance(other, RefConstraint)
        return RefConstraint(self.indices | other.indices)


@dataclass(frozen=True)
class CompoundConstraint(Constraint):
    lit_constraint: LiteralConstraint
    ref_constraint: RefConstraint

    def intersection(self, other: Constraint) -> Optional[Constraint]:
        if not isinstance(other, CompoundConstraint):
            return other.intersection(self)
        lit = self.lit_constraint.intersection(other.lit_constraint)
        if lit is None:
            return None
        assert isinstance(lit, LiteralConstraint)
        ref = self.ref_constraint.intersection(other.ref_constraint)
        assert isinstance(ref, RefConstraint)
        return CompoundConstraint(lit, ref)


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

    def intersection(self, other: 'Solution') -> Optional['Solution']:
        ixs_s = set(self.cells)
        ixs_o = set(other.cells)
        intersection = ixs_s & ixs_o

        cells = {}
        for ix in intersection:
            c = self.cells[ix].intersection(other.cells[ix])
            if c is None:
                return None
            cells[ix] = c

        def add_unique(ixs: Set[cell_ix_type], source: Dict[cell_ix_type, Constraint]):
            for ix in ixs - intersection:
                cells[ix] = source[ix]

        add_unique(ixs_s, self.cells)
        add_unique(ixs_o, other.cells)

        @lru_cache(100)
        def resolve_compound(cell_ix: cell_ix_type) -> Optional[CompoundConstraint]:
            cc = cells[cell_ix]
            acc = cc
            for ref_ix in cc.ref_constraint.indices:
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
class SolutionSource:
    name: str
    solutions: Iterable[Solution]


def merge_two_solutions_seq(xs: SolutionSource, ys: SolutionSource, callback: Callable[[str, bool], Any]
                            ) -> SolutionSource:
    name = f'{xs.name} & {ys.name}'

    def gen() -> Iterable[Solution]:
        xl = list(xs.solutions)
        for y in ys.solutions:
            for x in xl:
                i = y.intersection(x)
                if i is None:
                    callback(name, False)
                else:
                    callback(name, True)
                    yield i

    return SolutionSource(name, gen())


def merge_many_solutions(sol_seqs: Iterable[SolutionSource],
                         callback: Callable[[str, bool], Any]) -> SolutionSource:
    acc: Optional[SolutionSource] = None
    for sol in sol_seqs:
        if acc is None:
            acc = sol
        else:
            acc = merge_two_solutions_seq(acc, sol, callback)
    return SolutionSource('', []) if acc is None else acc


def main():
    from pprint import pprint
    strings = build_strings()
    for s in strings:
        print(s.pattern.raw, s.size)
        for m in s.gen_possible():
            print(' ', m.chr_seq)
        print('-' * 60)

    strings.sort(key=lambda st: st.size)

    i = 0

    def callback(name, res):
        nonlocal i
        i += 1
        if i % 50 == 0:
            print(('no', '  ')[res], 'match', name)

    lazy_solutions = merge_many_solutions((SolutionSource(s.pattern.raw, Solution.generate_solutions(s))
                                           for s in strings), callback)
    for solution in lazy_solutions.solutions:
        pprint(solution.cells)


__name__ == '__main__' and main()
