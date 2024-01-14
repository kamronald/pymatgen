"""This module implements output processing from PWSCF in Quantum Espresso."""

import re
from collections import defaultdict
import warnings

import numpy as np
from monty.io import zopen, reverse_readfile
from monty.re import regrep

from pymatgen.core import Element, Lattice, Structure, Species, PeriodicSite
from pymatgen.core.units import Energy, Length, EnergyArray, LengthArray
from pymatgen.util.io_utils import clean_lines
from pymatgen.electronic_structure.bandstructure import Kpoint, BandStructure

class PWOutput:
    """
    Parser for PWSCF output files.

    Authors: Ronald L. Kam
    """

    patterns = dict(
        energies=r"total energy\s+=\s+([\d\.\-]+)\sRy",
        final_energy="Final energy\s+=\s+([\d\.\-]+)\sRy",
        ecut=r"kinetic\-energy cutoff\s+=\s+([\d\.\-]+)\s+Ry",
        ecut_rho=r"charge density cutoff\s+=\s+([\d\.\-]+)\s+Ry",
        lattice_type=r"bravais\-lattice index\s+=\s+(\d+)",
        celldm1=r"celldm\(1\)=\s+([\d\.]+)\s",
        celldm2=r"celldm\(2\)=\s+([\d\.]+)\s",
        celldm3=r"celldm\(3\)=\s+([\d\.]+)\s",
        celldm4=r"celldm\(4\)=\s+([\d\.]+)\s",
        celldm5=r"celldm\(5\)=\s+([\d\.]+)\s",
        celldm6=r"celldm\(6\)=\s+([\d\.]+)\s",
        nkpts=r"number of k points=\s+([\d]+)",
        efermi=r"the Fermi energy is\s+([\d\.\-]+)\sev"
    )

    def __init__(self, filename):
        """
        Args:
            filename (str): Filename.
        """
        units_d = {'a.u.': 'bohr', 'angstrom': 'ang'}
        self.filename = filename
        self.data = defaultdict(list)
        self.read_pattern(PWOutput.patterns)
        for k, v in self.data.items():
            if k == "energies":
                self.data[k] = [Energy(i[0][0], "Ry") for i in v]
            elif k == "efermi":
                self.data[k] = [Energy(i[0][0], "eV") for i in v]
            elif k in ["lattice_type", "nkpts"]:
                self.data[k] = int(v[0][0][0])
            else:
                self.data[k] = float(v[0][0][0])

        # data to read from top: alat units, original lattice, original sites
        initial_crystal_axes = []
        initial_ion_positions = []
        site_species = []
        with open(self.filename, 'r') as f:
            all_lines = f.readlines()
            for i, line in enumerate(all_lines):
                if 'lattice parameter (alat)' in line:
                    self._alat = float(line.split()[-2])
                    print(self._alat)
                    alat_units = line.split()[-1]

                elif 'crystal axes:' in line:
                    for j in range(i+1, i+4):
                        cart_coords = [self._alat*float(v) for v in all_lines[j].split()[-4:-1]]
                        cart_coords_array = LengthArray(cart_coords, units_d[alat_units])
                        #cart_coords_ang = np.array(cart_coords_array.to('ang'))
                        initial_crystal_axes.append(cart_coords_array)

                elif 'site n.' in line and 'atom' in line:
                    for line_2 in all_lines[i+1:]:
                        line_items = line_2.split()
                        if len(line_items) == 0:
                            break
                        site_species.append(Species(line_items[1]))
                        initial_ion_positions.append(
                            LengthArray([float(v) * self._alat for v in line_items[-4:-1]],
                                        units_d[alat_units])
                        )
                elif 'number of k points' in line:
                    break

        print(initial_crystal_axes)

        # If symmetry is defined, then can immediately create the original lattice
        if self.data["lattice_type"] > 0:
            init_lattice = Lattice.from_parameters(
                self.data['celldm1'], self.data['celldm2'], self.data['celldm3'],
                self.data['celldm4'], self.data['celldm5'], self.data['celldm6']
            )

        else:
            init_latt_matrix = np.array(
                [np.array(axes.to('ang')) for axes in initial_crystal_axes]
            )
            init_lattice = Lattice(init_latt_matrix)

        init_sites = [
            PeriodicSite(
                species=spec, coords=coords.to('ang'), lattice=init_lattice, coords_are_cartesian=True
            )
            for spec, coords in zip(site_species, initial_ion_positions)
        ]

        self._initial_structure = Structure.from_sites(init_sites)

        # data to read from bottom:
        additional_data = {
            'finished': False, 'converged_ion': False, 'converged_el': False,
            'magmom': None, 'f_ionic_positions': None, 'forces': None,
            'band_structure': None
        }

        for k, v in additional_data.items():
            self.data[k] = v

        all_lines = []
        relax = False
        for i, line in enumerate(reverse_readfile(self.filename)):
            clean_line = line.strip()
            all_lines.append(clean_line)
            if 'JOB DONE' in clean_line:
                self.data['finished'] = True
            elif 'End of BFGS Geometry Optimization' in clean_line:
                self.data['converged_ion'] = True
                break
            elif 'convergence has been achieved' in clean_line:
                self.data['converged_el'] = True
            elif 'Begin final coordinates' in clean_line:
                struc_d = self.gather_struc_data(line_from_end=i)
                relax = True
            elif 'Magnetic moment per site' in clean_line:
                self.gather_mag_moments(line_from_end=i)
            #elif 'iteration #' in clean_line:
            #    print(clean_line)


        #print('struc dict', struc_d)
        if not self.data['finished']:
            warnings.warn('This calculation has not finished!')

        # set self._final_structure - convert to Angstroms!!
        if relax:
            print('Relaxation occured')
            if not len(struc_d['cell_params']):
                lattice = init_lattice
            else:
                lattice = Lattice(struc_d['cell_params'])

            sites = [
                PeriodicSite(species=spec, lattice=lattice, coords=coords) for spec, coords
                in zip(struc_d['species'], struc_d['coords'])
            ]
            structure = Structure.from_sites(sites)
            if self.data['magmom'] is not None:
                structure.add_site_property('magmom', self.data['magmom'])

            self._final_structure = structure

        else:
            self._final_structure = self._initial_structure

    def read_pattern(self, patterns, reverse=False, terminate_on_match=False,
                     postprocess=str):
        r"""
        General pattern reading. Uses monty's regrep method. Takes the same
        arguments.

        Args:
            patterns (dict): A dict of patterns, e.g.,
                {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"}.
            reverse (bool): Read files in reverse. Defaults to false. Useful for
                large files, esp OUTCARs, especially when used with
                terminate_on_match.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.

        Renders accessible:
            Any attribute in patterns. For example,
            {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"} will set the
            value of self.data["energy"] = [[-1234], [-3453], ...], to the
            results from regex and postprocess. Note that the returned
            values are lists of lists, because you can grep multiple
            items on one line.
        """
        matches = regrep(
            self.filename,
            patterns,
            reverse=reverse,
            terminate_on_match=terminate_on_match,
            postprocess=postprocess,
        )
        self.data.update(matches)

    def gather_struc_data(self, line_from_end=0):
        """
        Extract the species and coords of the final structure from output file.

        Args:
            line_from_end (int): Line to start parsing for the final structure

        Returns:
            Dict of species strings and coords
        """
        all_species = []
        all_coords = []
        new_cell_params = []
        param_units = None
        record_positions = False
        record_params = False
        with open(self.filename, 'r') as f:
            for line_f in f.readlines()[-line_from_end:]:
                if 'End final coordinates' in line_f:
                    break
                elif 'ATOMIC_POSITIONS' in line_f:
                    coord_units = line_f.split()[-1][1:-1]
                    record_positions = True
                elif 'CELL_PARAMETERS' in line_f:
                    print('cell params', line_f)
                    record_params = True
                    param_units = line_f.split()[-1][1:-1]
                elif record_positions:
                    line_f_vals = line_f.split()
                    if not len(line_f_vals):
                        record_positions = False
                        continue
                    all_species.append(line_f_vals[0])
                    coords = np.array([float(v) for v in line_f_vals[1:]])
                    all_coords.append(coords)

                elif record_params:
                    line_f_vals = line_f.split()
                    if not len(line_f_vals):
                        record_params = False
                        continue
                    params = np.array([float(v) for v in line_f_vals])
                    new_cell_params.append(params)

        new_cell_params = np.array(new_cell_params)
        all_coords = np.array(all_coords)

        return {
            'species': all_species, 'coords': all_coords, 'coord_units': coord_units,
            'cell_params': new_cell_params, 'param_units': param_units
        }

    def gather_mag_moments(self, line_from_end=0):
        """
        Extract magnetic moments (in Bohr magnetons) of the final structure.

        Args:
            line_from_end (int): Line to start parsing for the final structure

        """
        magmoms = []
        with open(self.filename, 'r') as f:
            for line_f in f.readlines()[-line_from_end:]:
                # for the collinear spin polarized case
                if 'magn=' in line_f:
                    magmoms.append(float(line_f.split('magn=')[-1]))

        if len(magmoms) > 0:
            self.data['magmom'] = magmoms


    def get_celldm(self, idx: int):
        """
        Args:
            idx (int): index.

        Returns:
            Cell dimension along index
        """
        return self.data[f"celldm{idx}"]

    @property
    def final_energy(self):
        """Returns: Final energy in eV."""
        return float(self.data["energies"][-1])

    @property
    def final_structure(self):
        """Returns: Final structure - cartesian units converted to Angstroms."""
        return self._final_structure

    @property
    def initial_structure(self):
        """Returns: Final structure - cartesian units converted to Angstroms."""
        return self._initial_structure

    @property
    def e_fermi(self):
        """Returns: Fermi energy."""
        return float(self.data["efermi"][-1].to('eV'))

    @property
    def lattice_type(self):
        """Returns: Lattice type."""
        return self.data["lattice_type"]

    @property
    def converged_ionic(self):
        return self.data['converged_ion']

    @property
    def converged_electronic(self):
        return self.data['converged_el']

    def get_band_structure(self):
        # Get K-points, band energies, and lattice
        pass
