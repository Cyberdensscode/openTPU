"""The card session scripts' parameter defaults and messages."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?[-?=+]([^}\n]*)\}")


def test_no_apostrophe_in_a_parameter_word():
    """An apostrophe in an unquoted ${NAME:?word} / ${NAME:-word} opens a single-quoted span that
    bash closes only at the next apostrophe, lines later: emprol.sh's EXPECT message ("the
    card's build") swallowed its MAP= and REFCFG= lines (session emprol, 2026-10-14: "MAP:
    unbound variable"), and mtps.sh / mtpt.sh printed "the cards build (selftests build ...)".
    A word with a command substitution in it ($(...)) keeps its own quotes."""
    bad = [f"{p.relative_to(ROOT)}:{i}: {l.strip()}"
           for p in sorted((ROOT / "tools").rglob("*.sh"))
           for i, l in enumerate(p.read_text(errors="replace").splitlines(), 1)
           for w in WORD.findall(l) if "'" in w and "$(" not in w]
    assert not bad, "\n".join(bad)
