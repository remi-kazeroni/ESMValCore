"""Fixes for IPSL-CM6A-LR model."""
from iris.cube import CubeList
from iris.coords import AuxCoord
from iris.exceptions import ConstraintMismatchError

from ..fix import Fix


class AllVars(Fix):
    """Fixes for thetao."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        try:
            cell_area = cubes.extract_strict('cell_area')
        except ConstraintMismatchError:
            return cubes

        cell_area = AuxCoord(
            cell_area.data,
            standard_name=cell_area.standard_name,
            long_name=cell_area.long_name,
            var_name=cell_area.var_name,
            units=cell_area.units,
        )
        new_list = CubeList()
        for cube in cubes:
            if cube.name() == 'cell_area':
                continue

            cube.add_aux_coord(cell_area, cube.coord_dims('latitude'))
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
            new_list.append(cube)
        return CubeList(new_list)


class zostoga(Fix):
    """Fixes for zostoga."""

    def fix_metadata(self, cube):
        """
        Fix zostoga by removing unity length coordinates.

        Parameters
        ----------
        cube: iris cube
            cube to fix

        Returns
        -------
        iris.cube

        """
#        assert 0

        return cube
