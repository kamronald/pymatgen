# PWScfSet, PWNscfSet, PWRelaxSet, PHDfptPhononSet, PHDfptHubbardSet
from abc import ABCMeta, abstractmethod
import os
from typing import Optional
import warnings

from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
from pathlib import Path
from pymatgen.core import Structure, PeriodicSite, Species
from pymatgen.core.units import Energy, EnergyArray, Length, LengthArray
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.espresso.pwscf.inputs import PWInput, HubbardModel

MODULE_DIR = Path(__file__).resolve().parent

class PWInputSet(MSONable, metaclass=ABCMeta):
    """
    Class representing a base PW input.
    Must be able to write_input, establish default settings, over-ride settings, generate k-points.
    Important params:
    1) symmetrize or not - determines ibrav and the cell dimensions input
    2) K-point treatment - can just copy from VASP I think
    3) calculation type: will be determined by the class
    4) verbosity
    5) pseudopotentials dict {species: pseudo}
    6) pseudos directory
    7) Energy cutoffs : wfcut = 100, rhocut = 1100 (Ry)
    8) Structure!!
    9) General functionality to build up the dicts that go into PWInput. PWInput then can write

    """
    def __init__(
        self,
        structure: Structure,
        sections_config: dict,
        name: str,
        pseudo: dict,
        pseudo_dir: str,
        specify_brav: Optional[bool] = 0,
        kpt_path: Optional[dict] = None,
        symprec: Optional[float] = 1e-5,
        len_units: Optional[str] = 'angstrom',
        user_control_set: Optional[dict] = None,
        user_system_set: Optional[dict] = None,
        user_electrons_set: Optional[dict] = None,
        user_ions_set: Optional[dict] = None,
        user_cell_set: Optional[dict] = None,
        user_kpoints_set: Optional[dict] = None,
        add_hubbard: Optional[bool] = True,
        user_hubbards: Optional[list] = None,
        hubbards_dir: Optional[str] = None,
        hubbards_file: Optional[str] = None,
    ):
        self.structure = structure
        self.sections = {}
        for sec_name in ["control", "system", "electrons", "ions", "cell"]:
            self.sections[sec_name] = sections_config.get(sec_name)

        self.kpt_settings = sections_config.get('kpoint_settings')
        if user_kpoints_set is not None:
            self.kpt_settings.update(user_kpoints_set)

        self.name = name
        self.len_units = len_units
        self.pseudo = pseudo
        self.specify_brav = specify_brav
        self.sections['control'].update({'prefix': self.name, 'pseudo_dir': pseudo_dir})

        if not specify_brav:
            self.sections['system'].update({'ibrav': 0})
        # to-do: determine the ibrav and lattice dimensions from symmetry

        # kpoints: user specifies to generate them or not. If no, they need to specify the points
        sections_updates = {
            'control': user_control_set, 'system': user_system_set, 'electrons': user_electrons_set,
            'ions': user_ions_set, 'cell': user_cell_set
        }
        print(sections_updates)

        for section_name, settings in sections_updates.items():
            if settings is not None:
                self.sections[section_name].update(settings)

        # Deal with k-points
        if kpt_path is not None:
            k_grid = None

        elif self.kpt_settings is not None:
            k_grid = self.generate_kpoints(self.kpt_settings)

        # Hubbard model
        if add_hubbard:
            if hubbards_dir is not None:
                print('Reading hubbards from file')
                self.hubbard_model = HubbardModel(
                    self.structure, extract_from_file=True, directory=hubbards_dir,
                    file_name=hubbards_file
                )
            elif user_hubbards is not None:
                self.hubbard_model = None
                warnings.warn("There is no option to pass in Hubbard values yet!")
            else:
                self.hubbard_model = HubbardModel(self.structure)

        if k_grid is None:
            print(self.pseudo)
            self.pw_input = PWInput(
                structure=self.structure,
                struc_sorting_type='hubbards_first',
                pseudo=self.pseudo,
                control=self.sections['control'],
                system=self.sections['system'],
                electrons=self.sections['electrons'],
                ions=self.sections['ions'],
                cell=self.sections['cell'],
                kpoints_mode='crystal',
                hubbard_model=self.hubbard_model
            )

        else:
            print(k_grid)
            print(self.pseudo)
            self.pw_input = PWInput(
                structure=self.structure,
                struc_sorting_type='hubbards_first',
                pseudo=self.pseudo,
                control=self.sections['control'],
                system=self.sections['system'],
                electrons=self.sections['electrons'],
                ions=self.sections['ions'],
                cell=self.sections['cell'],
                kpoints_grid=k_grid,
                hubbard_model=self.hubbard_model
            )

    def update_section(self, section_name, settings):
        self.sections[section_name].update(settings)

    def generate_kpoints(self, kpoints_setting):
        """
        Generate k-points by reciprocal density or k-spacing.
        Args:
            kpoints_setting (dict):
                {param_type: param}, (ie {'reciprocal_density': 64})

        Returns:
            k_pts (tuple): (nkx, nky, nkz)
                tuple of number k-points per direction of the reciprocal lattice

        """
        vasp_kpoints = None
        kpt_set_t = [(setting, param) for setting, param in kpoints_setting.items()][0]
        setting = kpt_set_t[0].lower()
        param = kpt_set_t[1]
        if setting == "reciprocal_density":
            vasp_kpoints = VaspKpoints.automatic_density_by_vol(self.structure, int(param))
        elif setting == "kspacing":
            density = int(1 / (param**3))
            vasp_kpoints = VaspKpoints.automatic_density_by_vol(self.structure, density)
        elif setting == "grid_density":
            vasp_kpoints = VaspKpoints.automatic_density(self.structure, int(param))
        elif setting == "length":
            vasp_kpoints = VaspKpoints.automatic(param)

        if vasp_kpoints is None:
            raise ValueError(f"Invalid k-point generation scheme specified: {param}")

        return vasp_kpoints.kpts[0]

    def write_input(self, directory='.', fname=f'pwscf.in'):

        if not os.path.exists(directory):
            os.mkdir(directory)

        self.pw_input.write_file(os.path.join(directory, fname))


def _load_yaml_config(fname):
    config = loadfn(MODULE_DIR / (f"{fname}.yaml"))
    if "PARENT" in config:
        parent_config = _load_yaml_config(config["PARENT"])
        for k, v in parent_config.items():
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                v_new = config.get(k, {})
                v_new.update(v)
                config[k] = v_new
    return config

class PWScfSet(PWInputSet):
    """
    Input set for static scf calculations.
    """
    config_sections = _load_yaml_config('PWScfSet')

    def __init__(self,
                 structure: Structure,
                 pseudo: dict,
                 pseudo_dir: str,
                 name='pwscf',
                 **kwargs
                 ):
        super().__init__(structure, PWScfSet.config_sections, name, pseudo, pseudo_dir, **kwargs)

        self.kwargs = kwargs


class PWRelaxSet(PWInputSet):
    """
    Input set for ionic relaxation calculations.
    """
    config_sections = _load_yaml_config('PWRelaxSet')

    def __init__(self,
                 structure: Structure,
                 pseudo: dict,
                 pseudo_dir: str,
                 name='pwrelax',
                 **kwargs
                 ):
        super().__init__(structure, PWRelaxSet.config_sections, name, pseudo, pseudo_dir, **kwargs)

        self.kwargs = kwargs
