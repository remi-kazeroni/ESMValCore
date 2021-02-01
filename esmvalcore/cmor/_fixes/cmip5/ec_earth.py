"""Fixes for EC-Earth model."""
from dask import array as da
import iris
import numpy as np

from ..fix import Fix
from ..shared import add_scalar_height_coord, cube_to_aux_coord


class allvars(Fix):
    """Common fixes to all vars"""

    def fix_metadata(self, cubes):
        """
        Fix metadata.
        Fixes error in time coordinate, sometimes contains trailing zeros
        Parameters
        ----------
        cube: iris.cube.CubeList
        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            try:
                old_time = cube.coord('time')
                if old_time.is_monotonic():
                    pass

                time_units = old_time.units
                time_data = old_time.points

                d = np.diff(time_data)
                idx_neg = np.where(d < 0.)[0]
                while len(idx_neg) > 0:
                    time_data = np.delete(time_data, idx_neg[0] + 1)
                    d = np.diff(time_data)
                    idx_neg = np.where(d < 0.)[0]

                for idx in idx_zeros:
                    if idx == 0:
                        continue
                    correct_time = time_units.num2date(time_data[idx - 1])
                    if days <= 31 and days >=28:  # assume monthly time steps
                        new_time = \
                            correct_time.replace(month=correct_time.month + 1)
                    else:  # use "time[1] - time[0]" as step
                        new_time = correct_time + time_diff
                    old_time.points[idx] = time_units.date2num(new_time)

                # create new time bounds
                old_time.bounds = None
                old_time.guess_bounds()

                # replace time coordinate with "repaired" values
                new_time = iris.coords.DimCoord.from_coord(old_time)
                time_idx = cube.coord_dims(old_time)
                cube.remove_coord('time')
                cube.add_dim_coord(new_time, time_idx)

            except iris.exceptions.CoordinateNotFoundError:
                pass

        return cubes


class Sic(Fix):
    """Fixes for sic."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """
        Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 273.15)
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """

        for cube in cubes:
            if not cube.coords(var_name='height'):
                add_scalar_height_coord(cube)

            if cube.coord('time').long_name is None:
                cube.coord('time').long_name = 'time'

        return cubes


class Areacello(Fix):
    """Fixes for areacello."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        areacello = cubes.extract('Areas of grid cell')[0]
        lat = cubes.extract('latitude')[0]
        lon = cubes.extract('longitude')[0]

        areacello.add_aux_coord(cube_to_aux_coord(lat), (0, 1))
        areacello.add_aux_coord(cube_to_aux_coord(lon), (0, 1))

        return iris.cube.CubeList([areacello, ])
