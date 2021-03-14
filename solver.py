import copy
import sre_constants
import sre_parse
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import (Any, Callable, Dict, List, Tuple, Optional, Collection, Iterable, Sequence, Union, overload,
                    TypeVar)

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
        if not self.chrs:
            return other
        if not other.chrs:
            return self
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
        return f'<CSeq {self!s}>'


@lru_cache(1000)
def chr_seq(chrs: Tuple['ReChr', ...]) -> ChrSeq:
    return ChrSeq(chrs)


empty_chr_seq = chr_seq(())


@dataclass(frozen=True)
class GroupBindings:
    bindings: Tuple[ChrSeq, ...] = ()

    def bind(self, index: int, cs: ChrSeq) -> 'GroupBindings':
        assert index > 0
        if index - 1 == len(self.bindings):  # appending to end
            return group_bindings(self.bindings + (cs,))
        else:  # resetting an existing group in a repeated group
            assert index <= len(self.bindings)
            bs = list(self.bindings)
            bs[index - 1] = cs
            return group_bindings(tuple(bs))

    def get(self, index: int) -> ChrSeq:
        return self.bindings[index - 1]

    def __str__(self):
        return '(' + ', '.join(map(str, self.bindings)) + ')'

    def __repr__(self) -> str:
        return f'<GBs{self!s}>'


@lru_cache(1000)
def group_bindings(bindings: Tuple[ChrSeq, ...]) -> GroupBindings:
    return GroupBindings(bindings)


empty_group_bindings = group_bindings(())


@dataclass(frozen=True)
class MatchState:
    groups: GroupBindings = empty_group_bindings
    bound_anys: Tuple['BoundAny', ...] = ()

    def copy(self,
             groups: Optional[GroupBindings] = None,
             bound_anys: Optional[Tuple['BoundAny', ...]] = None):
        return match_state(groups=or_else(groups, self.groups),
                           bound_anys=or_else(bound_anys, self.bound_anys))


@lru_cache(1000)
def match_state(groups: GroupBindings = empty_group_bindings,
                bound_anys: Tuple['BoundAny', ...] = ()) -> MatchState:
    return MatchState(groups, bound_anys)


empty_state = match_state()


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


@lru_cache(1000)
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


@dataclass(frozen=True)
class BoundAny(ReChr):
    index: int

    def __str__(self):
        return f'<{self.index}>'

    def __repr__(self):
        return f'<BndAny {self.index}>'


class ReAny(Re):
    _instance = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def span(self) -> Tuple[int, int]:
        return 1, 1

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        if max_len >= 1:
            bound = BoundAny(len(state.bound_anys))
            yield possible_match(chr_seq((bound,)), state=state.copy(bound_anys=state.bound_anys + (bound,)))

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
    if max_len <= 0:
        return
    for p in previous:
        mx = max_len - len(p.chr_seq)
        if mx > 0:
            for match in re.gen_possible(0, mx, p.state):
                yield match.copy(p.chr_seq.add(match.chr_seq))


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
                            p = p.copy(state=p.state.copy(p.state.groups.bind(self.index, p.chr_seq)))
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
class ReGroupRef(Re):
    group: ReSeq

    def span(self) -> Tuple[int, int]:
        return self.group.span()

    def gen_possible(self, min_len: int, max_len: int, state: MatchState) -> Iterable[PossibleMatch]:
        match = state.groups.get(self.group.index)
        if min_len <= len(match) <= max_len:
            yield possible_match(match, state)

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
        return f'<Cn ix={self.index} st={self.string.pattern.raw!r} cl={self.cell.position}>'


@dataclass()
class String:
    pattern: Pattern
    positions: List[Position]

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.cell.position}' for c in self.positions)
        return f'<St pt={self.pattern.raw!r} cns=[{cns}]>'

    @property
    def size(self):
        return len(self.positions)

    def gen_possible(self) -> Iterable[PossibleMatch]:
        return self.pattern.re.gen_possible(self.size, self.size, empty_state)


@dataclass()
class Cell:
    row: int
    col: int
    positions: List[Position]

    @property
    def position(self):
        return self.row, self.col

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.index}@{c.string.pattern.raw!r}' for c in self.positions)
        return f'<Ce pt={self.position} cns=[{cns}]>'


def build_constraints() -> List[String]:
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


def main():
    strings = build_constraints()
    for s in strings:
        print(s.pattern.raw, s.size)
        for m in s.gen_possible():
            print(' ', m.chr_seq)
        print('-' * 60)


__name__ == '__main__' and main()
