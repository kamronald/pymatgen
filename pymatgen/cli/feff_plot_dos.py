#!/usr/bin/env python

"""Script to plot density of states (DOS) generated by an FEFF run
either by site, element, or orbital.
"""

from __future__ import annotations

import argparse

from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.feff.outputs import LDos

__author__ = "Alan Dozier"
__credits__ = "Anubhav Jain, Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Alan Dozier"
__email__ = "adozier@uky.edu"
__date__ = "April 7, 2012"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="""Convenient DOS Plotter for Feff runs.
    Author: Alan Dozier
    Version: 1.0
    Last updated: April, 2013"""
    )

    parser.add_argument(
        "filename",
        metavar="filename",
        type=str,
        nargs=1,
        help="ldos%% file set to plot",
    )
    parser.add_argument("filename1", metavar="filename1", type=str, nargs=1, help="feff.inp input file ")
    parser.add_argument(
        "-s",
        "--site",
        dest="site",
        action="store_const",
        const=True,
        help="plot site projected DOS",
    )
    parser.add_argument(
        "-e",
        "--element",
        dest="element",
        action="store_const",
        const=True,
        help="plot element projected DOS",
    )
    parser.add_argument(
        "-o",
        "--orbital",
        dest="orbital",
        action="store_const",
        const=True,
        help="plot orbital projected DOS",
    )

    args = parser.parse_args()
    f = LDos.from_file(args.filename1[0], args.filename[0])
    dos = f.complete_dos

    all_dos = {}
    all_dos["Total"] = dos

    structure = f.complete_dos.structure

    if args.site:
        for i, site in enumerate(structure):
            all_dos[f"Site {i} {site.specie.symbol}"] = dos.get_site_dos(site)
    if args.element:
        all_dos.update(dos.get_element_dos())
    if args.orbital:
        all_dos.update(dos.get_spd_dos())

    plotter = DosPlotter()
    plotter.add_dos_dict(all_dos)
    plotter.show()


if __name__ == "__main__":
    raise SystemExit(main())
