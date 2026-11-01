from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Grid = List[List[str]]

METRIC_KEYS = ("table_teds",)

_WS = re.compile(r"\s+")


def _norm(s: Any) -> str:
    return _WS.sub(" ", str(s)).strip().casefold()


def _cell_sim(a: Any, b: Any) -> float:
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _rectangularize(grid: Grid) -> Grid:
    width = max((len(r) for r in grid), default=0)
    return [list(r) + [""] * (width - len(r)) for r in grid]


def _grid_from_html(html: str) -> Optional[Grid]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return None
    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return None
    rows: Grid = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            rows.append([c.get_text(" ", strip=True) for c in cells])
    return _rectangularize(rows) if rows else None


def _grid_from_markdown(md: str) -> Optional[Grid]:
    rows: Grid = []
    for line in (md or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        rows.append(cells)
    return _rectangularize(rows) if rows else None


def to_grid(entry: Any) -> Optional[Grid]:
    if not isinstance(entry, dict):
        return None
    headers, rows = entry.get("headers"), entry.get("rows")
    if headers is not None or rows is not None:
        grid: Grid = []
        if headers:
            grid.append([str(h) for h in headers])
        for r in (rows or []):
            grid.append([str(c) for c in r])
        return _rectangularize(grid) if grid else None
    if entry.get("html"):
        g = _grid_from_html(str(entry["html"]))
        if g:
            return g
    if entry.get("markdown"):
        return _grid_from_markdown(str(entry["markdown"]))
    return None


_BIG = 1e6


class _Node:
    __slots__ = ("tag", "text", "children")

    def __init__(self, tag: str, text: str = "", children: Optional[List["_Node"]] = None):
        self.tag, self.text, self.children = tag, text, children or []


def _grid_to_tree(grid: Grid) -> _Node:
    return _Node("table", "", [_Node("tr", "", [_Node("td", c) for c in row]) for row in grid])


def _prepare(root: _Node) -> Tuple[List[Tuple[str, str]], List[int], List[int]]:
    labels: List[Tuple[str, str]] = []
    lld: List[int] = []

    def rec(n: _Node) -> int:
        child_ids = [rec(c) for c in n.children]
        labels.append((n.tag, n.text))
        my_id = len(labels) - 1
        lld.append(lld[child_ids[0]] if child_ids else my_id)
        return my_id

    rec(root)
    last_by_lld: Dict[int, int] = {}
    for i, l in enumerate(lld):
        last_by_lld[l] = i
    return labels, lld, sorted(last_by_lld.values())


def _tree_edit_distance(a: _Node, b: _Node) -> float:
    la, llda, kra = _prepare(a)
    lb, lldb, krb = _prepare(b)
    na, nb = len(la), len(lb)

    def rename(x: int, y: int) -> float:
        (ta, xa), (tb, xb) = la[x], lb[y]
        if ta != tb:
            return _BIG
        if ta == "td":
            return 1.0 - _cell_sim(xa, xb)
        return 0.0

    td = [[0.0] * nb for _ in range(na)]
    for i in kra:
        for j in krb:
            ioff, joff = llda[i], lldb[j]
            m, n = i - ioff + 2, j - joff + 2
            fd = [[0.0] * n for _ in range(m)]
            for x in range(1, m):
                fd[x][0] = fd[x - 1][0] + 1.0
            for y in range(1, n):
                fd[0][y] = fd[0][y - 1] + 1.0
            for x in range(1, m):
                for y in range(1, n):
                    ax, by = x + ioff - 1, y + joff - 1
                    if llda[ax] == ioff and lldb[by] == joff:
                        fd[x][y] = min(fd[x - 1][y] + 1.0, fd[x][y - 1] + 1.0,
                                       fd[x - 1][y - 1] + rename(ax, by))
                        td[ax][by] = fd[x][y]
                    else:
                        p, q = llda[ax] - ioff, lldb[by] - joff
                        fd[x][y] = min(fd[x - 1][y] + 1.0, fd[x][y - 1] + 1.0,
                                       fd[p][q] + td[ax][by])
    return td[na - 1][nb - 1]


def teds(grid_a: Grid, grid_b: Grid) -> float:
    MAX_EVAL_ROWS = 30
    if len(grid_a) > MAX_EVAL_ROWS:
        grid_a = grid_a[:MAX_EVAL_ROWS]
    if len(grid_b) > MAX_EVAL_ROWS:
        grid_b = grid_b[:MAX_EVAL_ROWS]

    ta, tb = _grid_to_tree(grid_a), _grid_to_tree(grid_b)
    na = 1 + len(grid_a) + sum(len(r) for r in grid_a)
    nb = 1 + len(grid_b) + sum(len(r) for r in grid_b)
    denom = max(na, nb)
    if denom == 0:
        return 1.0
    dist = _tree_edit_distance(ta, tb)
    return max(0.0, 1.0 - dist / denom)


def _cells(grid: Grid) -> set:
    return {_norm(c) for row in grid for c in row if _norm(c)}


def _match_score(gold: Grid, pred: Grid) -> float:
    g, p = _cells(gold), _cells(pred)
    if not g and not p:
        return 1.0
    inter = len(g & p)
    union = len(g | p) or 1
    return inter / union


def _pair_scores(gold: Grid, pred: Grid) -> Dict[str, float]:
    return {"table_teds": teds(gold, pred)}


def score_tables(gold_tables: Sequence[Any], pred_tables: Sequence[Any]) -> Optional[Dict[str, float]]:
    gold_grids = [g for g in (to_grid(t) for t in (gold_tables or [])) if g]
    if not gold_grids:
        return None
    pred_grids = [g for g in (to_grid(t) for t in (pred_tables or [])) if g]

    sums = {k: 0.0 for k in METRIC_KEYS}
    used: set = set()
    for gg in gold_grids:
        best_j, best_s = None, -1.0
        for j, pg in enumerate(pred_grids):
            if j in used:
                continue
            s = _match_score(gg, pg)
            if s > best_s:
                best_s, best_j = s, j
        if best_j is None:
            continue
        used.add(best_j)
        for k, v in _pair_scores(gg, pred_grids[best_j]).items():
            sums[k] += v
    n = len(gold_grids)
    return {k: round(sums[k] / n, 4) for k in METRIC_KEYS}
