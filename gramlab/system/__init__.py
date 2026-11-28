"""gramlab.system — the components as one managed board.

A text goes through components that each read some layers of a Document and write one. Every component states its
contract (reads / writes), its failure value (never a guess), and how to benchmark itself. `System.run(text)` returns the
layered analysis; `System.status()` prints the board: component, version, benchmark, number, date.

    from gramlab.system import System
    sysm = System.default(models_dir="results")      # frontend.pkl, linker.pkl, about.pkl
    doc = sysm.run(open("chapter.txt").read())
    doc.layers["about"], doc.layers["records"], doc.layers["skeleton"], doc.layers["direction"]
    print(sysm.status())
"""
from .board import Component, Document, System  # noqa: F401
