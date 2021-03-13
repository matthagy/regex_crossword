import copy
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.markers import Path as MarkerPath
from matplotlib.transforms import Affine2D

import constants

size = constants.size
assert size % 2 == 1
mid = size // 2


def row_size(row_index: int) -> int:
    assert 0 <= row_index < size
    return mid + 1 + min(row_index, size - 1 - row_index)


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


@dataclass()
class Pattern:
    pattern: str


@dataclass()
class Constrains:
    index: int
    string: 'String'
    cell: 'Cell'

    def __repr__(self) -> str:
        return f'<Cn ix={self.index} st={self.string.pattern.pattern!r} cl={self.cell.position}>'


@dataclass()
class String:
    pattern: Pattern
    constraints: List[Constrains]

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.cell.position}' for c in self.constraints)
        return f'<St pt={self.pattern.pattern!r} cns=[{cns}]>'


@dataclass()
class Cell:
    row: int
    col: int
    constraints: List[Constrains]

    @property
    def position(self):
        return self.row, self.col

    def __repr__(self) -> str:
        cns = ', '.join(f'{c.index}@{c.string.pattern.pattern!r}' for c in self.constraints)
        return f'<Ce pt={self.position} cns=[{cns}]>'


def build_constraints() -> List[String]:
    cells = [[Cell(i, j, []) for j in range(row_size(i))]
             for i in range(constants.size)]

    strings: List[String] = []

    def con(string: String, cell: Cell, index: int):
        assert 0 <= index < constants.size, f'{index} is invalid'
        cn = Constrains(index=index, string=string, cell=cell)
        string.constraints.append(cn)
        cell.constraints.append(cn)

    def add_horizontal():
        for i, s in enumerate(constants.y):
            s = String(Pattern(s), [])
            strings.append(s)
            for j, c in enumerate(cells[i]):
                con(s, c, j)

    def add_diag(ss: List[str], reverse=False):
        for i, s in enumerate(ss):
            s = String(Pattern(s), [])
            strings.append(s)
            start = max(0, i - mid)
            end = min(i + mid + 1, constants.size)
            for j in range(start, end):
                cell = cells[j][i if j <= mid else i - (j - mid)]
                con(s, cell, end - j - 1 if reverse else j - start)

    add_horizontal()
    add_diag(constants.x, reverse=True)
    add_diag(constants.z, reverse=False)

    for s in strings:
        s.constraints.sort(key=lambda c: c.index)
        assert [c.index for c in s.constraints] == list(range(len(s.constraints)))

    return strings
