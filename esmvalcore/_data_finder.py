"""Data finder module for the ESMValTool."""
# Authors:
# Bouwe Andela (eScience, NL - b.andela@esciencecenter.nl)
# Valeriu Predoi (URead, UK - valeriu.predoi@ncas.ac.uk)
# Mattia Righi (DLR, Germany - mattia.righi@dlr.de)

import fnmatch
import logging
import os
import re
import glob

import iris

from ._config import get_project_config

logger = logging.getLogger(__name__)


def find_files(dirnames, filenames):
    """Find files matching filenames in dirnames."""
    logger.debug("Looking for files matching %s in %s", filenames, dirnames)

    result = []
    for dirname in dirnames:
        for path, _, files in os.walk(dirname, followlinks=True):
            for filename in filenames:
                matches = fnmatch.filter(files, filename)
                result.extend(os.path.join(path, f) for f in matches)

    return result


def get_start_end_year(filename):
    """Get the start and end year from a file name.

    This works for filenames matching

    *[-,_]YYYY*[-,_]YYYY*.*
      or
    *[-,_]YYYY*.*
      or
    YYYY*[-,_]*.*
      or
    YYYY*[-,_]YYYY*[-,_]*.*
      or
    YYYY*[-,_]*[-,_]YYYY*.* (Does this make sense? Is this worth catching?)
    """
    name = os.path.splitext(filename)[0]

    name = name.split(os.sep)[-1]
    filename_list = [elem.split('-') for elem in name.split('_')]
    filename_list = [elem for sublist in filename_list for elem in sublist]

    pos_ydates = [elem.isdigit() and len(elem) >= 4 for elem in filename_list]
    pos_ydates_l = list(pos_ydates)
    pos_ydates_r = list(pos_ydates)

    for ind, _ in enumerate(pos_ydates_l):
        if ind != 0:
            pos_ydates_l[ind] = (pos_ydates_l[ind - 1] and pos_ydates_l[ind])

    for ind, _ in enumerate(pos_ydates_r):
        if ind != 0:
            pos_ydates_r[-ind - 1] = (pos_ydates_r[-ind]
                                      and pos_ydates_r[-ind - 1])

    dates = [
        filename_list[ind] for ind, _ in enumerate(pos_ydates)
        if pos_ydates_r[ind] or pos_ydates_l[ind]
    ]
    start_year = None
    end_year = None
    if len(dates) == 1:
        start_year = int(dates[0][:4])
        end_year = start_year
    elif len(dates) == 2:
        start_year, end_year = int(dates[0][:4]), int(dates[1][:4])
    else:
        # Slower than just parsing the name
        try:
            cubes = iris.load(filename)
        except OSError:
            raise ValueError('File {0} can not be read'.format(filename))

        for cube in cubes:
            logger.debug(cube)
            try:
                time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            start_year = time.cell(0).point.year
            end_year = time.cell(-1).point.year
            break

    if not start_year or not end_year:
        raise ValueError(
            'File {0} dates do not match a recognized pattern and time can '
            'not be readed from the file'.format(filename)
        )
    return start_year, end_year


def select_files(filenames, start_year, end_year):
    """Select files containing data between start_year and end_year.

    This works for filenames matching *_YYYY*-YYYY*.* or *_YYYY*.*
    """
    selection = []
    for filename in filenames:
        start, end = get_start_end_year(filename)
        if start <= end_year and end >= start_year:
            selection.append(filename)
    return selection


def _replace_tags(path, variable):
    """Replace tags in the config-developer's file with actual values."""
    path = path.strip('/')
    tlist = re.findall(r'{([^}]*)}', path)
    paths = [path]
    for tag in tlist:
        original_tag = tag
        tag, _, _ = _get_caps_options(tag)

        if tag == 'latestversion':  # handled separately later
            continue
        elif tag in variable:
            replacewith = variable[tag]
        else:
            raise KeyError("Dataset key {} must be specified for {}, check "
                           "your recipe entry".format(tag, variable))

        paths = _replace_tag(paths, original_tag, replacewith)
    return paths


def _replace_tag(paths, tag, replacewith):
    """Replace tag by replacewith in paths."""
    _, lower, upper = _get_caps_options(tag)
    result = []
    if isinstance(replacewith, (list, tuple)):
        for item in replacewith:
            result.extend(_replace_tag(paths, tag, item))
    else:
        text = _apply_caps(str(replacewith), lower, upper)
        result.extend(p.replace('{' + tag + '}', text) for p in paths)
    return result


def _get_caps_options(tag):
    lower = False
    upper = False
    if tag.endswith('.lower'):
        lower = True
        tag = tag[0:-6]
    elif tag.endswith('.upper'):
        upper = True
        tag = tag[0:-6]
    return tag, lower, upper


def _apply_caps(original, lower, upper):
    if lower:
        return original.lower()
    if upper:
        return original.upper()
    return original


def _resolve_latestversion(dirname_template):
    """Resolve the 'latestversion' tag."""
    if '{latestversion}' not in dirname_template:
        return dirname_template

    # Find latest version
    part1, part2 = dirname_template.split('{latestversion}')
    part2 = part2.lstrip(os.sep)
    if os.path.exists(part1):
        versions = os.listdir(part1)
        versions.sort(reverse=True)
        for version in ['latest'] + versions:
            dirname = os.path.join(part1, version, part2)
            if os.path.isdir(dirname):
                return dirname

    return dirname_template


def _select_drs(input_type, drs, project):
    """Select the directory structure of input path."""
    cfg = get_project_config(project)
    input_path = cfg[input_type]
    if isinstance(input_path, str):
        return input_path

    structure = drs.get(project, 'default')
    if structure in input_path:
        return input_path[structure]

    raise KeyError(
        'drs {} for {} project not specified in config-developer file'.format(
            structure, project))


def get_rootpath(rootpath, project):
    """Select the rootpath."""
    if project in rootpath:
        return rootpath[project]
    if 'default' in rootpath:
        return rootpath['default']
    raise KeyError('default rootpath must be specified in config-user file')


def _find_input_dirs(variable, rootpath, drs):
    """Return a the full paths to input directories."""
    project = variable['project']

    root = get_rootpath(rootpath, project)
    path_template = _select_drs('input_dir', drs, project)

    dirnames = []
    for dirname_template in _replace_tags(path_template, variable):
        for base_path in root:
            dirname = os.path.join(base_path, dirname_template)
            dirname = _resolve_latestversion(dirname)
            matches = glob.glob(dirname)
            matches = [match for match in matches if os.path.isdir(match)]
            if matches:
                for match in matches:
                    logger.debug("Found %s", match)
                    dirnames.append(match)
            else:
                logger.debug("Skipping non-existent %s", dirname)

    return dirnames


def _get_filenames_glob(variable, drs):
    """Return patterns that can be used to look for input files."""
    path_template = _select_drs('input_file', drs, variable['project'])
    filenames_glob = _replace_tags(path_template, variable)
    return filenames_glob


def _find_input_files(variable, rootpath, drs):
    input_dirs = _find_input_dirs(variable, rootpath, drs)
    filenames_glob = _get_filenames_glob(variable, drs)
    files = find_files(input_dirs, filenames_glob)

    return files


def get_input_filelist(variable, rootpath, drs):
    """Return the full path to input files."""
    # change ensemble to fixed r0i0p0 for fx variables
    # this is needed and is not a duplicate effort
    if variable['project'] == 'CMIP5' and variable['frequency'] == 'fx':
        variable['ensemble'] = 'r0i0p0'
    files = _find_input_files(variable, rootpath, drs)
    # do time gating only for non-fx variables
    if variable['frequency'] != 'fx':
        files = select_files(files, variable['start_year'],
                             variable['end_year'])
    return files


def get_output_file(variable, preproc_dir):
    """Return the full path to the output (preprocessed) file."""
    cfg = get_project_config(variable['project'])

    # Join different experiment names
    if isinstance(variable.get('exp'), (list, tuple)):
        variable = dict(variable)
        variable['exp'] = '-'.join(variable['exp'])

    outfile = os.path.join(
        preproc_dir,
        variable['diagnostic'],
        variable['variable_group'],
        _replace_tags(cfg['output_file'], variable)[0] + '.nc',
    )

    return outfile


def get_statistic_output_file(variable, preproc_dir):
    """Get multi model statistic filename depending on settings."""
    template = os.path.join(
        preproc_dir,
        '{diagnostic}',
        '{variable_group}',
        '{dataset}_{mip}_{short_name}_{start_year}-{end_year}.nc',
    )

    outfile = template.format(**variable)

    return outfile
