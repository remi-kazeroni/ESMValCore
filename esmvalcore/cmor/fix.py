"""
Apply automatic fixes for known errors in cmorized data

All functions in this module will work even if no fixes are available
for the given dataset. Therefore is recommended to apply them to all
variables to be sure that all known errors are
fixed.

"""
import logging
from collections import defaultdict

from iris.cube import CubeList

from ._fixes.fix import Fix
from .check import _get_cmor_checker

logger = logging.getLogger(__name__)


def fix_file(file, cmor_name, project, dataset, output_dir):
    """
    Fix files before ESMValTool can load them.

    This fixes are only for issues that prevent iris from loading the cube or
    that cannot be fixed after the cube is loaded.

    Original files are not overwritten.

    Parameters
    ----------
    file: str
        Path to the original file
    cmor_name: str
        Variable's cmor name
    project: str
    dataset:str
    output_dir: str
        Output directory for fixed files

    Returns
    -------
    str:
        Path to the fixed file

    """
    for fix in Fix.get_fixes(
            project=project, dataset=dataset, variable=cmor_name):
        file = fix.fix_file(file, output_dir)
    return file


def fix_metadata(cubes,
                 cmor_name,
                 project,
                 dataset,
                 cmor_table=None,
                 mip=None,
                 frequency=None):
    """
    Fix cube metadata if fixes are required and check it anyway.

    This method collects all the relevant fixes for a given variable, applies
    them and checks the resulting cube (or the original if no fixes were
    needed) metadata to ensure that it complies with the standards of its
    project CMOR tables.

    Parameters
    ----------
    cubes: iris.cube.CubeList
        Cubes to fix
    cmor_name: str
        Variable's name
    project: str

    dataset: str

    cmor_table: str, optional
        CMOR tables to use for the check, if available

    mip: str, optional
        Variable's MIP, if available

    frequency: str, optional
        Variable's data frequency, if available

    Returns
    -------
    iris.cube.Cube:
        Fixed and checked cube

    Raises
    ------
    CMORCheckError
        If the checker detects errors in the metadata that it can not fix.

    """
    fixes = Fix.get_fixes(
        project=project, dataset=dataset, variable=cmor_name)
    fixed_cubes = []
    by_file = defaultdict(list)
    for cube in cubes:
        by_file[cube.attributes.get('source_file', '')].append(cube)

    for cube_list in by_file.values():
        cube_list = CubeList(cube_list)
        for fix in fixes:
            cube_list = fix.fix_metadata(cube_list)

        if len(cube_list) != 1:
            cube = None
            for raw_cube in cube_list:
                if raw_cube.var_name == cmor_name:
                    cube = raw_cube
                    break
            if not cube:
                raise ValueError(
                    'More than one cube found for variable %s in %s:%s but '
                    'none of their var_names match the expected. \n'
                    'Full list of cubes encountered: %s' %
                    (cmor_name, project, dataset, cube_list)
                )
            logger.warning(
                'Found variable %s in %s:%s, but there were other present in '
                'the file. Those extra variables are usually metadata '
                '(cell area, latitude descriptions) that was not saved '
                'properly. It is possible that errors appear further on '
                'because of this. \nFull list of cubes encountered: %s',
                cmor_name,
                project,
                dataset,
                cube_list
            )
        else:
            cube = cube_list[0]

        if cmor_table and mip:
            checker = _get_cmor_checker(
                frequency=frequency,
                table=cmor_table,
                mip=mip,
                cmor_name=cmor_name,
                fail_on_error=False,
                automatic_fixes=True)
            cube = checker(cube).check_metadata()
        cube.attributes.pop('source_file', None)
        fixed_cubes.append(cube)
    return fixed_cubes


def fix_data(cube,
             cmor_name,
             project,
             dataset,
             cmor_table=None,
             mip=None,
             frequency=None):
    """
    Fix cube data if fixes add present and check it anyway.

    This method assumes that metadata is already fixed and checked.

    This method collects all the relevant fixes for a given variable, applies
    them and checks resulting cube (or the original if no fixes were
    needed) metadata to ensure that it complies with the standards of its
    project CMOR tables.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube to fix
    cmor_name: str
        Variable's name
    project: str

    dataset: str

    cmor_table: str, optional
        CMOR tables to use for the check, if available

    mip: str, optional
        Variable's MIP, if available

    frequency: str, optional
        Variable's data frequency, if available

    Returns
    -------
    iris.cube.Cube:
        Fixed and checked cube

    Raises
    ------
    CMORCheckError
        If the checker detects errors in the data that it can not fix.

    """
    for fix in Fix.get_fixes(
            project=project, dataset=dataset, variable=cmor_name):
        cube = fix.fix_data(cube)
    if cmor_table and mip:
        checker = _get_cmor_checker(
            frequency=frequency,
            table=cmor_table,
            mip=mip,
            cmor_name=cmor_name,
            fail_on_error=False,
            automatic_fixes=True)
        cube = checker(cube).check_data()
    return cube
