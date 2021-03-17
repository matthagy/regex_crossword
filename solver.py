import copy
import sre_constants
import sre_parse
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache, reduce, total_ordering
from typing import (Any, Callable, Dict, List, Tuple, Optional, Collection, Iterable, Sequence, Union, overload,
                    TypeVar, FrozenSet, Iterator, Mapping, Set, ValuesView, AbstractSet)

import matplotlib.pyplot as plt
import numpy as np
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
K = TypeVar('K')
V = TypeVar('V')


def or_else(x: Optional[T], e: T) -> T:
    return x if x is not None else e


class FrozenDict(Mapping[K, V]):
    _d: Mapping[K, V]
    _h: Optional[int] = None

    def __init__(self, d: Mapping[K, V]) -> None:
        self._d = d

    def __new__(cls, d: Mapping[K, V]) -> 'FrozenDict[K, V]':
        if isinstance(d, FrozenDict):
            return d
        if not isinstance(d, Mapping):
            d = dict(d)
        fd = super().__new__(cls)
        fd.__init__(d)
        return fd

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if isinstance(o, FrozenDict):
            return o._d == self._d
        if isinstance(0, Mapping):
            return o == self._d
        return False

    def __hash__(self) -> int:
        if self._h is None:
            self._h = hash(tuple(sorted(self._d.items())))
        return self._h

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self}>'

    def __str__(self) -> str:
        return str(self._d)

    def __getitem__(self, k: K) -> V:
        return self._d[k]

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self) -> Iterator[K]:
        return iter(self._d)

    def get(self, key: K, value: Optional[T] = None) -> Union[V, Optional[T]]:
        return self._d.get(key, value)

    def items(self) -> AbstractSet[Tuple[K, V]]:
        return self._d.items()

    def keys(self) -> AbstractSet[K]:
        return self._d.keys()

    def values(self) -> ValuesView[V]:
        return self._d.values()

    def __contains__(self, o: object) -> bool:
        return o in self._d


class ReChr(ABC):
    pass

@dataclass(frozen=True)
class ChrSeq(Sequence[ReChr]):
    chrs: Tuple[ReChr, ...]

    def add(self, other: 'ChrSeq') -> 'ChrSeq':
        return chr_seq(self.chrs + other.chrs)

    @overload
    def __getitem__(self, i: int) -> ReChr:
        pass

    @overload
    def __getitem__(self, s: slice) -> 'ChrSeq':
        pass

    def __getitem__(self, i: Union[int, slice]) -> Union[ReChr, 'ChrSeq']:
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
    def from_op_arg(cls, op: int, arg: Any, groups: Mapping[int, 'Re']) -> 'Re':
        return cls.op_converters[op](op, arg, groups)





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
                     groups: Mapping[int, 'ReSeq'],
                     index: Optional[int] = None) -> 'ReSeq':
        return cls(res=tuple(Re.from_op_arg(op, arg, groups) for op, arg in ops_args), index=index)

    @classmethod
    def from_op_arg(cls, op: int, arg: Any, groups: Mapping[int, 'ReSeq']) -> 'ReSeq':
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
    def from_op_arg(cls, op: int, arg: Any, groups: Mapping[int, ReSeq]) -> 'ReGroupRef':
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
    def from_op_arg(cls, op: int, arg: Any, groups: Mapping[int, ReSeq]) -> 'ReRepeat':
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
    def from_op_arg(cls, op: int, arg: Any, groups: Mapping[int, ReSeq]) -> 'ReBranch':
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

    def union(self, other: 'LiteralConstraint') -> 'LiteralConstraint':
        if self.negate == other.negate:
            if self.negate:
                combined = self.chars & other.chars
                if not combined:
                    return all_literal_constraint
                return LiteralConstraint(combined, negate=True)
            return LiteralConstraint(self.chars | other.chars, negate=False)

        pos, neg = self, other
        if pos.negate:
            pos, neg = neg, pos
        assert not pos.negate
        assert neg.negate
        combined = neg.chars - pos.chars
        if not combined:
            return all_literal_constraint
        return LiteralConstraint(combined, negate=True)

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


@dataclass(frozen=True, repr=False)
class Solution(Mapping[cell_ix_type, Constraint]):
    cells: FrozenDict[cell_ix_type, Constraint]

    def __init__(self, cells: Mapping[cell_ix_type, Constraint]):
        object.__setattr__(self, 'cells', FrozenDict(cells))

    def __getitem__(self, k: cell_ix_type) -> Constraint:
        return self.cells[k]

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[cell_ix_type]:
        return iter(self.cells)

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
        return f'<{self.__class__.__name__} n={len(self.cells)}>'

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
    def get_indexes(self, dimension: Dimension) -> Iterable[int]:
        pass

    @abstractmethod
    def is_point(self, dimension: Dimension) -> bool:
        pass

    @abstractmethod
    def select(self, dimension: Dimension, values: Mapping[int, T]) -> Iterable[T]:
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

    def __new__(cls) -> 'DimensionSolutionAll':
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

    def get_indexes(self, dimension: Dimension) -> Iterable[int]:
        return range(dimension.size)

    def is_point(self, dimension: Dimension) -> bool:
        return dimension.size == 1

    def select(self, dimension: Dimension, values: Mapping[int, T]) -> Iterable[T]:
        return values.values()

    def optimize(self, dimension: Dimension) -> DimensionSolutions:
        return self


dimension_all: DimensionSolutions = DimensionSolutionAll()


@dataclass(frozen=True)
class DimensionSolutionIndexes(DimensionSolutions):
    indexes: FrozenSet[int]
    negate: bool = False

    def __repr__(self):
        return f'<{self.__class__.__name__} sz={self}>'

    def __str__(self):
        return f'{"-" if self.negate else "+"}{len(self.indexes)}'

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

    def get_indexes(self, dimension: Dimension) -> Iterable[int]:
        if self.negate:
            return frozenset(range(dimension.size)) - self.indexes
        return self.indexes

    def is_point(self, dimension: Dimension) -> bool:
        if self.negate:
            return len(self.indexes) == dimension.size - 1
        return len(self.indexes) == 1

    def select(self, dimension: Dimension, values: Mapping[int, T]) -> Iterable[T]:
        return (values[i] for i in self.get_indexes(dimension))

    def optimize(self, dim: Dimension) -> DimensionSolutions:
        if len(self.indexes) == dim.size:
            return dimension_all
        if len(self.indexes) > dim.size // 2:
            return DimensionSolutionIndexes(frozenset(range(dim.size)) - self.indexes, negate=not self.negate)
        return self


@dataclass(frozen=True)
class DimensionCombination(Collection[Tuple[Dimension, DimensionSolutions]]):
    dimensions: FrozenDict[Dimension, DimensionSolutions]

    def __init__(self, dimensions: Mapping[Dimension, DimensionSolutions]):
        object.__setattr__(self, 'dimensions', FrozenDict(dimensions))

    def __len__(self) -> int:
        return len(self.dimensions)

    def __iter__(self) -> Iterator[Tuple[Dimension, DimensionSolutions]]:
        return iter(self.dimensions.items())

    def __contains__(self, x: object) -> bool:
        return x in self.dimensions.items()

    def __repr__(self):
        return f'<{self.__class__.__name__} {self}>'

    def __str__(self):
        sz = self.size()
        if sz > 1e9:
            sz_str = f'{sz:.1e}'
        else:
            sz_str = f'{sz:,}'

        point_dimensions = self.point_dimensions()
        index_dimensions = {dim: sol for dim, sol in self.index_dimensions().items()
                            if dim not in point_dimensions}

        def dim_str(dims: Collection[T], detail_func: Callable[[Collection[T]], str]) -> str:
            if not dims:
                return '[]'
            if len(dims) > 5:
                return f'[{len(dims)} dims]'
            return detail_func(dims)

        all_str = dim_str(self.all_dimensions(), lambda x: '[{}]'.format(', '.join(f'{dim}' for dim in x)))
        pts_str = dim_str(point_dimensions, lambda x: '[{}]'.format(
            ', '.join(f'{dim}: {list(sol.get_indexes(dim))[0]}' for dim, sol in x.items())))
        ixs_str = dim_str(index_dimensions, lambda x: '[{}]'.format(
            ', '.join(f'{dim}: {sol.size(dim)}' for dim, sol in x.items())))

        return f'sz={sz_str} all={all_str} pts={pts_str} ixs={ixs_str}'

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

    def all_dimensions(self) -> Set[Dimension]:
        return {dim for dim, sol in self.dimensions.items()
                if isinstance(sol, DimensionSolutionAll)}

    def index_dimensions(self) -> Mapping[Dimension, DimensionSolutionIndexes]:
        return {dim: sol for dim, sol in self.dimensions.items()
                if isinstance(sol, DimensionSolutionIndexes)}

    def point_dimensions(self) -> Mapping[Dimension, DimensionSolutions]:
        return {dim: sol for dim, sol in self.dimensions.items() if sol.is_point(dim)}

    @property
    def is_point(self) -> bool:
        return bool(self.dimensions) and all(sol.is_point(dim) for dim, sol in self.dimensions.items())

    def select(self, values: Mapping[Dimension, Mapping[int, T]]) -> Mapping[Dimension, Iterable[T]]:
        return {dim: sol.select(dim, values[dim]) for dim, sol in self.dimensions.items()}

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
                    for i in sols.get_indexes(dim):
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


@dataclass(frozen=True)
class CellConstraints(Collection[Tuple[Constraint, DimensionCombination]]):
    constraint_dimensions: Sequence[Tuple[Constraint, DimensionCombination]]

    def __len__(self) -> int:
        return len(self.constraint_dimensions)

    def __iter__(self) -> Iterator[Tuple[Constraint, DimensionCombination]]:
        return iter(self.constraint_dimensions)

    def __contains__(self, x: object) -> bool:
        return x in self.constraint_dimensions

    def __repr__(self):
        return f'<{self.__class__.__name__} {self}>'

    def __str__(self):
        cds = self.constraint_dimensions
        if not cds:
            return '[]'
        if len(cds) > 10:
            return f'[n={len(cds)} cds]'
        if len(cds) > 5:
            return ', '.join(str(con) for con, comb in cds)
        return '[{}]'.format(', '.join(f'{con}: {comb}' for con, comb in cds))

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
                           comb: DimensionCombination,
                           ref_con: RefConstraint,
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
                int_combs.add(ref_int, comb)
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

    def intersection_constraint(self, con: Constraint) -> Optional['CellConstraints']:
        constraint_dimensions = []
        for con_i, comb_i in self.constraint_dimensions:
            con_i = con_i.intersection(con)
            if con_i is not None:
                constraint_dimensions.append((con_i, comb_i))
        if not constraint_dimensions:
            raise RuntimeError(f'no matching dimension combinations')
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
    def for_dimension(cls, dim: Dimension, constraints: Mapping[Constraint, List[int]]):
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

        return SparseSolutionSet(self.dimensions | other.dimensions, cells)

    def apply_references(self) -> Optional['SparseSolutionSet']:
        cells = dict(self.cells)

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
                cells[ix] = ac

        return SparseSolutionSet(self.dimensions, cells)

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

    @classmethod
    def for_string(cls, s: String) -> Tuple['SparseSolutionSet', Sequence[Solution]]:
        solutions = list(Solution.generate_solutions(s))
        cells: Dict[cell_ix_type, Dict[Constraint, List[int]]] = defaultdict(lambda: defaultdict(list))
        for sol_i, solution in enumerate(solutions):
            for ix, con in solution.cells.items():
                cells[ix][con].append(sol_i)

        dim = Dimension(s.name[0], int(s.name[1::]), len(solutions))
        ss = cls(frozenset([dim]), {ix: CellConstraints.for_dimension(dim, cons) for ix, cons in cells.items()})
        return ss, solutions


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


@dataclass(frozen=True)
class Solver:
    dims_to_strings: FrozenDict[Dimension, String]
    indexed_solutions: FrozenDict[Dimension, Mapping[int, Solution]]

    @classmethod
    def create_for_strings(cls, strings: Iterable[String]) -> Tuple['Solver', Sequence[SparseSolutionSet]]:
        dims_to_strings = {}
        solution_sets = []
        indexed_solutions = {}
        with tqdm(strings, desc='init solutions') as progress:
            for string in progress:
                ss, solutions = SparseSolutionSet.for_string(string)
                dim, = ss.dimensions
                # skip dimensions that only have a single solution
                if dim.size == 1:
                    continue
                dims_to_strings[dim] = string
                solution_sets.append(ss)
                indexed_solutions[dim] = {i: sol for i, sol in enumerate(solutions)}
        solver = cls(dims_to_strings=FrozenDict(dims_to_strings), indexed_solutions=FrozenDict(indexed_solutions))
        return solver, tuple(solution_sets)

    @staticmethod
    def intersect(xs: Iterable[T]) -> Optional[T]:
        return reduce(lambda a, b: None if a is None or b is None else a.intersection(b), xs)

    @staticmethod
    def iteratively_simplify(solution_set: SparseSolutionSet, max_iterations=20) -> SparseSolutionSet:
        ci = solution_set.common_intersection()
        with tqdm(range(max_iterations), desc='process') as progress:
            for i in progress:
                start_ci = ci
                progress.set_description(f'process {i} {start_ci}')
                solution_set = solution_set.filter_cells_using_other_unions()
                solution_set = solution_set.apply_references()
                solution_set = solution_set.push_reference_constraints()
                ci = solution_set.common_intersection()
                if start_ci == ci:
                    break
        return solution_set

    def point_solution(self, point: DimensionCombination) -> Optional[Solution]:
        assert point.is_point
        return self.intersect([sol for sol, in point.select(self.indexed_solutions).values()])

    def expand_solution_set(self, solution_set: SparseSolutionSet) -> FrozenSet[Solution]:
        ci = solution_set.common_intersection()
        point_dim = DimensionCombination(ci.point_dimensions())
        print('pt', point_dim)
        point_solution = self.point_solution(point_dim)
        print('pt sol', point_solution)
        if point_solution is None:
            return frozenset()

        ix_dims = DimensionCombination({dim: sol for dim, sol in ci if not sol.is_point(dim)})
        print('ix dims', ix_dims)
        if len(ix_dims) == 0:
            return frozenset([point_solution])

        final_solution = set()
        with tqdm(ix_dims.iter_points(), total=ix_dims.size()) as progress:
            for p in progress:
                progress.set_description(f'expand {p}')
                ix_solution = self.point_solution(p)
                if ix_solution is not None:
                    ix_solution = ix_solution.intersection(point_solution)
                    if ix_solution is not None:
                        final_solution.add(ix_solution)

        return frozenset(final_solution)

    @classmethod
    def solve(cls) -> FrozenSet[Solution]:
        strings = build_strings()
        solver, solution_sets = cls.create_for_strings(strings)
        solution_set = solver.intersect(solution_sets)
        if solution_set is None:
            return frozenset()

        ci = solution_set.common_intersection()
        print('init ci', ci)

        solution_set = solver.iteratively_simplify(solution_set)
        ci = solution_set.common_intersection()
        print('iter ci', ci)

        final_solutions = solver.expand_solution_set(solution_set)
        print('final_solutions', final_solutions)
        return final_solutions


__name__ == '__main__' and Solver.solve()
