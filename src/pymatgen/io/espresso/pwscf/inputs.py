"""This module implements input processing from PWSCF in Quantum Espresso."""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
from monty.io import Path
from monty.io import zopen, reverse_readfile
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn

from pymatgen.core import Element, Lattice, Structure, Species, PeriodicSite
from pymatgen.util.io_utils import clean_lines
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import str_delimited

MODULE_DIR = Path(__file__).resolve().parent

class PWInput:
    """
    Base input file class. Right now, only supports no symmetry and is
    very basic.
    """

    def __init__(
        self,
        structure,
        struc_sorting_type="hubbards_first",
        pseudo=None,
        control=None,
        system=None,
        #symmetrize=False,
        #symprec=1e-5,
        electrons=None,
        ions=None,
        cell=None,
        kpoints_mode="automatic",
        kpoints_grid=(1, 1, 1),
        kpoints_shift=(0, 0, 0),
        high_sym_kpts=None,
        kpt_labels=None,
        kpoints_line_dens=20,
        hubbard_model=None
    ):
        """
        Initializes a PWSCF input file.

        Args:
            structure (Structure): Input structure. For spin-polarized calculation,
                properties (e.g. {"starting_magnetization": -0.5,
                "pseudo": "Mn.pbe-sp-van.UPF"}) on each site is needed instead of
                pseudo (dict).
            pseudo (dict): A dict of the pseudopotentials to use. Default to None.
            control (dict): Control parameters. Refer to official PWSCF doc
                on supported parameters. Default to {"calculation": "scf"}
            system (dict): System parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            electrons (dict): Electron parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            ions (dict): Ions parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            cell (dict): Cell parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            kpoints_mode (str): Kpoints generation mode. Default to automatic.
            kpoints_grid (sequence): The kpoint grid. Default to (1, 1, 1).
            kpoints_shift (sequence): The shift for the kpoints. Defaults to
                (0, 0, 0).
        """
        self.structure = structure
        if struc_sorting_type == 'electronegativity':
            self.structure = structure.get_sorted_structure()
        elif struc_sorting_type == 'hubbards_first':
            self.sort_structure_hubbards()

        self.structure = self.add_species_labels(self.structure)

        #self.symmetrize = symmetrize
        #self.symprec = symprec

        # if symmetrize:
        #     sga = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        #     sg_num = sga.get_space_group_number()
        #     sym_struc = sga.get_symmetrized_structure()

        sections = {}
        sections["control"] = control or {"calculation": "scf"}
        sections["system"] = system or {}
        sections["electrons"] = electrons or {}
        sections["ions"] = ions or {}
        sections["cell"] = cell or {}
        if pseudo is None:
            for site in self.structure:
                try:
                    site.properties["pseudo"]
                except KeyError:
                    raise PWInputError(f"Missing {site} in pseudo specification!")
        else:
            for species in self.structure.composition:
                if species.symbol not in pseudo:
                    raise PWInputError(f"Missing {species} in pseudo specification!")
        self.pseudo = pseudo
        self.sections = sections
        self.kpoints_mode = kpoints_mode
        self.kpoints_grid = kpoints_grid
        self.kpoints_shift = kpoints_shift
        self.hubbard_model = hubbard_model

    def __str__(self):
        out = []
        site_descriptions = {}

        if self.pseudo is not None:
            site_descriptions = self.pseudo
        else:
            c = 1
            for site in self.structure:
                name = None
                for k, v in site_descriptions.items():
                    if site.properties == v:
                        name = k

                if name is None:
                    name = f"{site.specie.symbol}{c}"
                    site_descriptions[name] = site.properties
                    c += 1

        def to_str(v):
            if isinstance(v, str):
                return f"{v!r}"
            if isinstance(v, float):
                return f"{str(v).replace('e', 'd')}"
            if isinstance(v, bool):
                if v:
                    return ".TRUE."
                return ".FALSE."
            return v

        for k1 in ["control", "system", "electrons", "ions", "cell"]:
            v1 = self.sections[k1]
            out.append(f"&{k1.upper()}")
            sub = []
            updated_site_descriptions = {}
            for i, (site, spec) in enumerate(zip(self.structure, self.structure.species)):
                updated_site_descriptions[spec.symbol] = site_descriptions[spec.symbol]

            for k2 in sorted(v1):
                if isinstance(v1[k2], list):
                    n = 1
                    for _ in v1[k2][: len(site_descriptions)]:
                        sub.append(f"  {k2}({n}) = {to_str(v1[k2][n - 1])}")
                        n += 1
                else:
                    sub.append(f"  {k2} = {to_str(v1[k2])}")
            if k1 == "system":
                if "ibrav" not in self.sections[k1]:
                    sub.append("  ibrav = 0")
                if "nat" not in self.sections[k1]:
                    sub.append(f"  nat = {len(self.structure)}")
                if "ntyp" not in self.sections[k1]:
                    sub.append(f"  ntyp = {len(updated_site_descriptions)}")

                if self.structure.site_properties.get('magmom') and self.sections['system'].get('nspin') == 2:
                    labels_ind_map = {}
                    spec_ind = 1
                    for site in self.structure:
                        spec_label = site.properties['species_label']
                        magmom = site.properties['magmom']
                        if spec_label not in labels_ind_map:
                            labels_ind_map[spec_label] = spec_ind
                            if abs(magmom) > 0.2:
                                sub.append(f"  starting_magnetization({spec_ind}) = {magmom}")

                            spec_ind += 1

                        else:
                            continue
            sub.append("/")
            out.append(",\n".join(sub))

        out.append("ATOMIC_SPECIES")


        for k, v in updated_site_descriptions.items():
            e = re.match(r"[A-Z][a-z]?", k).group(0)
            p = v if self.pseudo is not None else v["pseudo"]
            out.append(f"  {k}  {Element(e).atomic_mass:.4f} {p}")

        out.append("ATOMIC_POSITIONS crystal")
        if self.pseudo is not None:
            for i, site in enumerate(self.structure):
                if 'species_label' in site.properties:
                    this_spec_sym = site.properties['species_label']
                else:
                    this_spec_sym = site.specie.symbol

                out.append(f"  {this_spec_sym}   {site.a:.6f}   {site.b:.6f}   {site.c:.6f}")
        else:
            for i, site in enumerate(self.structure):
                name = None
                for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
                    if v == site.properties:
                        name = k
                out.append(f"  {name} {site.a:.6f} {site.b:.6f} {site.c:.6f}")

        out.append(f"K_POINTS {self.kpoints_mode}")
        if self.kpoints_mode == "automatic":
            kpt_str = [f"{i}" for i in self.kpoints_grid]
            kpt_str.extend([f"{i}" for i in self.kpoints_shift])
            out.append(f"  {' '.join(kpt_str)}")
        elif self.kpoints_mode == "crystal_b":
            out.append(f" {len(self.kpoints_grid)}")
            for i in range(len(self.kpoints_grid)):
                kpt_str = [f"{entry:.4f}" for entry in self.kpoints_grid[i]]
                out.append(f" {' '.join(kpt_str)}")
        elif self.kpoints_mode == "gamma":
            pass
        else:
            raise RuntimeError(
                f"This k-point setting {self.kpoints_mode} is not supported!"
            )

        out.append("CELL_PARAMETERS angstrom")
        for vec in self.structure.lattice.matrix:
            out.append(f"  {vec[0]:f} {vec[1]:f} {vec[2]:f}")

        if self.hubbard_model is not None:
            for line in str(self.hubbard_model).split('\n'):
                out.append(line)

        return "\n".join(out)

    def as_dict(self):
        """
        Create a dictionary representation of a PWInput object.

        Returns:
            dict
        """
        return {
            "structure": self.structure.as_dict(),
            "pseudo": self.pseudo,
            "sections": self.sections,
            "kpoints_mode": self.kpoints_mode,
            "kpoints_grid": self.kpoints_grid,
            "kpoints_shift": self.kpoints_shift,
        }

    @classmethod
    def from_dict(cls, pwinput_dict):
        """
        Load a PWInput object from a dictionary.

        Args:
            pwinput_dict (dict): dictionary with PWInput data

        Returns:
            PWInput object
        """
        return cls(
            structure=Structure.from_dict(pwinput_dict["structure"]),
            pseudo=pwinput_dict["pseudo"],
            control=pwinput_dict["sections"]["control"],
            system=pwinput_dict["sections"]["system"],
            electrons=pwinput_dict["sections"]["electrons"],
            ions=pwinput_dict["sections"]["ions"],
            cell=pwinput_dict["sections"]["cell"],
            kpoints_mode=pwinput_dict["kpoints_mode"],
            kpoints_grid=pwinput_dict["kpoints_grid"],
            kpoints_shift=pwinput_dict["kpoints_shift"],
        )

    def sort_structure_hubbards(self):
        """
        Sort structure such that species with strongly correlated electronic states
        are listed first. This is a requirement for DFPT calculations of Hubbard terms
        in the HP module. Order of species listed will be f-d-p-s block elements.

        Returns:
            sorted Structure

        """
        sorting_d = {}
        for i, site in enumerate(self.structure):
            block = site.specie.block
            el = site.specie.symbol
            if block not in sorting_d:
                sorting_d[block] = {el: [i]}
            else:
                if el not in sorting_d[block]:
                    sorting_d[block][el] = [i]
                else:
                    sorting_d[block][el].append(i)

        reordered_sites = []
        for block in ['f', 'd', 'p', 's']:
            if block not in sorting_d:
                continue
            for spec, site_inds in sorting_d[block].items():
                reordered_sites.extend([self.structure[i] for i in site_inds])

        self.structure = Structure.from_sites(reordered_sites)

    def add_species_labels(self, structure):
        """
        Add appropriate labeling of species on each site. Species with up and down spins will need to be separately labeled
        (e.g. Mn(up) - Mn1, Mn(down) - Mn2

        Args:
            structure (Structure)

        Returns:
            labeled_structure (Structure) - with new site property "species_label"

        """
        spec_map = {}
        if 'magmom' not in structure.site_properties:
            spec_labels = [s.element.__str__() if type(s) != Element else s.__str__() for s in structure.species]

        else:
            significant_mags = np.array([m for m in structure.site_properties['magmom'] if abs(m) > 0.2])
            if np.all(significant_mags > 0):  # ferromagnetic - no need to number the species
                spec_labels = [s.element.__str__() if type(s) != Element else s.__str__() for s in structure.species]

            else:  # Contains up and down spins - need to distinguish them
                spec_ind = 1
                spec_labels = []

                for site, spec in zip(structure, structure.species):
                    mag = site.properties['magmom']
                    el = spec.element.__str__() if type(spec) != Element else spec.__str__()
                    if abs(mag) > 0.2:  # need to label the magnetic species
                        if mag > 0:
                            this_label = f"{el}_up"
                        else:
                            this_label = f"{el}_down"

                        if this_label not in spec_map:
                            this_spec_label = f"{el}{spec_ind}"
                            spec_ind += 1
                            spec_map[this_label] = this_spec_label
                        else:
                            this_spec_label = spec_map[this_label]

                    else:
                        this_spec_label = el
                        if el not in spec_map:
                            spec_map[el] = el
                            spec_ind += 1

                    spec_labels.append(this_spec_label)

        labeled_structure = structure.copy()
        labeled_structure.add_site_property('species_label', spec_labels)
        return labeled_structure

    def make_default_hubbard(self):
        self.hubbard_model = HubbardModel(self.structure)

    def write_file(self, filename):
        """
        Write the PWSCF input file.

        Args:
            filename (str): The string filename to output to.
        """
        with open(filename, "w") as f:
            f.write(str(self))

    @classmethod
    def from_file(cls, filename):
        """
        Reads an PWInput object from a file.

        Args:
            filename (str): Filename for file

        Returns:
            PWInput object
        """
        with zopen(filename, "rt") as f:
            return cls.from_str(f.read())

    @classmethod
    @np.deprecate(message="Use from_str instead")
    def from_string(cls, *args, **kwargs):
        return cls.from_str(*args, **kwargs)

    @classmethod
    def from_str(cls, string):
        """
        Reads an PWInput object from a string.

        Args:
            string (str): PWInput string

        Returns:
            PWInput object
        """
        lines = list(clean_lines(string.splitlines()))

        def input_mode(line):
            if line[0] == "&":
                return ("sections", line[1:].lower())
            if "ATOMIC_SPECIES" in line:
                return ("pseudo",)
            if "K_POINTS" in line:
                return "kpoints", line.split()[1]
            if "OCCUPATIONS" in line:
                return "occupations"
            if "CELL_PARAMETERS" in line or "ATOMIC_POSITIONS" in line:
                return "structure", line.split()[1]
            if line == "/":
                return None
            return mode

        sections = {
            "control": {},
            "system": {},
            "electrons": {},
            "ions": {},
            "cell": {},
        }
        pseudo = {}
        lattice = []
        species = []
        coords = []
        structure = None
        site_properties = {"pseudo": []}
        mode = None
        for line in lines:
            mode = input_mode(line)
            if mode is None:
                pass
            elif mode[0] == "sections":
                section = mode[1]
                m = re.match(r"(\w+)\(?(\d*?)\)?\s*=\s*(.*)", line)
                if m:
                    key = m.group(1).strip()
                    key_ = m.group(2).strip()
                    val = m.group(3).strip()
                    if key_ != "":
                        if sections[section].get(key) is None:
                            val_ = [0.0] * 20  # MAX NTYP DEFINITION
                            val_[int(key_) - 1] = PWInput.proc_val(key, val)
                            sections[section][key] = val_

                            site_properties[key] = []
                        else:
                            sections[section][key][int(key_) - 1] = PWInput.proc_val(key, val)
                    else:
                        sections[section][key] = PWInput.proc_val(key, val)

            elif mode[0] == "pseudo":
                m = re.match(r"(\w+)\s+(\d*.\d*)\s+(.*)", line)
                if m:
                    pseudo[m.group(1).strip()] = m.group(3).strip()
            elif mode[0] == "kpoints":
                m = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
                if m:
                    kpoints_grid = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    kpoints_shift = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
                else:
                    kpoints_mode = mode[1]
                    kpoints_grid = (1, 1, 1)
                    kpoints_shift = (0, 0, 0)

            elif mode[0] == "structure":
                m_l = re.match(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", line)
                m_p = re.match(r"(\w+)\s+(-?\d+\.\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", line)
                if m_l:
                    lattice += [
                        float(m_l.group(1)),
                        float(m_l.group(2)),
                        float(m_l.group(3)),
                    ]
                elif m_p:
                    site_properties["pseudo"].append(pseudo[m_p.group(1)])
                    species.append(m_p.group(1))
                    coords += [[float(m_p.group(2)), float(m_p.group(3)), float(m_p.group(4))]]

                    if mode[1] == "angstrom":
                        coords_are_cartesian = True
                    elif mode[1] == "crystal":
                        coords_are_cartesian = False
        structure = Structure(
            Lattice(lattice),
            species,
            coords,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties,
        )
        return cls(
            structure=structure,
            control=sections["control"],
            pseudo=pseudo,
            system=sections["system"],
            electrons=sections["electrons"],
            ions=sections["ions"],
            cell=sections["cell"],
            kpoints_mode=kpoints_mode,
            kpoints_grid=kpoints_grid,
            kpoints_shift=kpoints_shift,
        )

    @staticmethod
    def proc_val(key, val):
        """
        Static helper method to convert PWINPUT parameters to proper type, e.g.,
        integers, floats, etc.

        Args:
            key: PWINPUT parameter key
            val: Actual value of PWINPUT parameter.
        """
        float_keys = (
            "etot_conv_thr",
            "forc_conv_thr",
            "conv_thr",
            "Hubbard_U",
            "Hubbard_J0",
            "degauss",
            "starting_magnetization",
        )

        int_keys = (
            "nstep",
            "iprint",
            "nberrycyc",
            "gdir",
            "nppstr",
            "ibrav",
            "nat",
            "ntyp",
            "nbnd",
            "nr1",
            "nr2",
            "nr3",
            "nr1s",
            "nr2s",
            "nr3s",
            "nspin",
            "nqx1",
            "nqx2",
            "nqx3",
            "lda_plus_u_kind",
            "edir",
            "report",
            "esm_nfit",
            "space_group",
            "origin_choice",
            "electron_maxstep",
            "mixing_ndim",
            "mixing_fixed_ns",
            "ortho_para",
            "diago_cg_maxiter",
            "diago_david_ndim",
            "nraise",
            "bfgs_ndim",
            "if_pos",
            "nks",
            "nk1",
            "nk2",
            "nk3",
            "sk1",
            "sk2",
            "sk3",
            "nconstr",
        )

        bool_keys = (
            "wf_collect",
            "tstress",
            "tprnfor",
            "lkpoint_dir",
            "tefield",
            "dipfield",
            "lelfield",
            "lorbm",
            "lberry",
            "lfcpopt",
            "monopole",
            "nosym",
            "nosym_evc",
            "noinv",
            "no_t_rev",
            "force_symmorphic",
            "use_all_frac",
            "one_atom_occupations",
            "starting_spin_angle",
            "noncolin",
            "x_gamma_extrapolation",
            "lda_plus_u",
            "lspinorb",
            "london",
            "ts_vdw_isolated",
            "xdm",
            "uniqueb",
            "rhombohedral",
            "realxz",
            "block",
            "scf_must_converge",
            "adaptive_thr",
            "diago_full_acc",
            "tqr",
            "remove_rigid_rot",
            "refold_pos",
        )

        def smart_int_or_float(numstr):
            if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
                return float(numstr)
            return int(numstr)

        try:
            if key in bool_keys:
                if val.lower() == ".true.":
                    return True
                if val.lower() == ".false.":
                    return False
                raise ValueError(key + " should be a boolean type!")

            if key in float_keys:
                return float(re.search(r"^-?\d*\.?\d*d?-?\d*", val.lower()).group(0).replace("d", "e"))

            if key in int_keys:
                return int(re.match(r"^-?[0-9]+", val).group(0))

        except ValueError:
            pass

        try:
            return smart_int_or_float(val.replace("d", "e"))
        except ValueError:
            pass

        if "true" in val.lower():
            return True
        if "false" in val.lower():
            return False

        m = re.match(r"^[\"|'](.+)[\"|']$", val)
        if m:
            return m.group(1)
        return None


class PWInputError(BaseException):
    """Error for PWInput."""


class HubbardModel(MSONable):
    """
    A HubbardModel defines the species, orbital manifold, and Hubbard value for each species to
    be included in the Hubbard model of a structure. Allows ease of generating the Hubbard
    model from default values, extracting from a DFPT run, manually assigning, and writing
    parameters to a PW input file. Hubbard on-site U and inter-site V terms are supported.

    """
    def __init__(
        self,
        structure,
        hub_projectors='ortho-atomic',
        u_vals=None,
        u_atom_manifold=None,
        v_vals=None,
        v_atoms_manifolds=None,
        v_atoms_inds=None,
        extract_from_file=False,
        directory=None,
        file_name=None
    ):
        """
        Args:
            structure (Structure):
                Structure that Hubbard model is for.
            hub_projectors (str):
                Type of Hubbard projector, 'ortho-atomic' is preferred. The PW
                input docs have more info.
            u_vals (list):
                List of Hubbard U values (eV)
            u_atom_manifold (list):
                List of strings describing species and orbital manifold
                ie ['Mn1-3d', 'Mn2-3d'...]
            v_vals (list):
                List of Hubbard V values (eV)
        """
        self.structure = structure
        if hub_projectors not in ['ortho-atomic', 'atomic', 'norm-atomic', 'wf',
                                  'pseudo']:
            raise RuntimeError(
                f"This Hubbard projector is not supported: {hub_projectors}"
            )
        self.hub_projectors = hub_projectors

        if extract_from_file:
            self.set_hubbards_from_file(directory, file_name)
        elif u_vals is None and v_vals is None:
            self.use_defaults = True
            self.hubbard_us = self.gen_hubbard_u_from_default()
            self.hubbard_vs = []  # currently no way to generate V values
        else:
            self.use_defaults = False
            self.hubbard_us = [
                (atom_man, float(u)) for atom_man, u in zip(u_atom_manifold, u_vals)
            ]
            self.hubbard_vs = [
                (atom_man_i, atom_man_j, int(i), int(j), float(v)) for
                (atom_man_i, atom_man_j), (i, j), v in
                zip(v_atoms_manifolds, v_atoms_inds, v_vals)
                               ]

    def gen_hubbard_u_from_default(self):
        # Find the species in a structure and attempt to find it in the default hubbards dict
        default_hubbards = loadfn(os.path.join(MODULE_DIR, 'default_hubbard_u.yaml'))
        structure_ns = self.structure.copy()
        structure_ns.remove_spin()
        hubbard_us = []
        hubbards_to_use = {}
        for spec in set(structure_ns.species):
            if spec.__str__() not in default_hubbards:
                if spec.symbol not in default_hubbards:
                    warnings.warn(f"There is no default Hubbard U for {spec}!")
                    continue

                hubbards_to_use[spec.__str__()] = [(man, u) for man, u in default_hubbards[spec.symbol].items()]
            hubbards_to_use[spec.__str__()] = [(man, u) for man, u in default_hubbards[spec.__str__()].items()]

        for i, spec in enumerate(structure_ns.species):
            if spec.__str__() not in hubbards_to_use:
                continue
            hubbard_us.append(
                (f"{spec.symbol}{i+1}-{hubbards_to_use[spec.__str__()][0][0]}",
                 float(hubbards_to_use[spec.__str__()][0][1])
                )
            )

        return hubbard_us

    def __str__(self):
        out = []
        out.append("HUBBARD {}{}{}".format(r"{", self.hub_projectors, r"}"))
        for hub_tup in self.hubbard_us:
            u_line = f"U    {hub_tup[0]}    {hub_tup[1]}"
            out.append(u_line)

        for hub_tup in self.hubbard_vs:
            u_line = f"V    {hub_tup[0]}    {hub_tup[1]}    {hub_tup[2]}    {hub_tup[3]}"
            out.append(u_line)

        return "\n".join(out)

    def set_hubbards_from_file(self, directory='.', fname='HUBBARD.dat'):
        self.hubbard_us = []
        self.hubbard_vs = []
        with open(os.path.join(directory, fname), 'r') as f:
            for line in f.readlines():
                line_items = line.split()
                if line_items[0] == "U":
                    self.hubbard_us.append((line_items[1], float(line_items[2])))
                elif line_items[0] == "V":
                    self.hubbard_vs.append(
                        (line_items[1], line_items[2], int(line_items[3]), int(line_items[4]), float(line_items[5]))
                    )



