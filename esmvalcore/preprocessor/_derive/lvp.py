"""Derivation of variable `lvp`.

authors:
    - weig_ka

"""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lvp`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'var_name': 'hfls'
            },
            {
                'var_name': 'pr'
            },
            {
                'var_name': 'evspsbl'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute Latent Heat Release from Precipitation."""
        hfls_cube = cubes.extract_strict(
            Constraint(name='surface_upward_latent_heat_flux'))
        pr_cube = cubes.extract_strict(Constraint(name='precipitation_flux'))
        evspsbl_cube = cubes.extract_strict(
            Constraint(name='water_evaporation_flux'))

        lvp_cube = hfls_cube * (pr_cube / evspsbl_cube)

        return lvp_cube
