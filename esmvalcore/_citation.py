"""Citation module."""
import os
import logging
import re
from pathlib import Path
import textwrap
import requests

from ._config import DIAGNOSTICS_PATH

if DIAGNOSTICS_PATH:
    REFERENCES_PATH = Path(DIAGNOSTICS_PATH) / 'references'
else:
    REFERENCES_PATH = ''

logger = logging.getLogger(__name__)

CMIP6_URL_STEM = 'https://cera-www.dkrz.de/WDCC/ui/cerasearch'

# it is the technical overview and should always be cited
ESMVALTOOL_PAPER = (
    '@article{righi19gmd,\n'
    '\tdoi = {10.5194/gmd-2019-226},\n'
    '\turl = {https://doi.org/10.5194%2Fgmd-2019-226},\n'
    '\tyear = 2019,\n'
    '\tmonth = {sep},\n'
    '\tpublisher = {Copernicus {GmbH}},\n'
    '\tauthor = {Mattia Righi and Bouwe Andela and Veronika Eyring '
    'and Axel Lauer and Valeriu Predoi and Manuel Schlund '
    'and Javier Vegas-Regidor and Lisa Bock and Björn Brötz '
    'and Lee de Mora and Faruk Diblen and Laura Dreyer '
    'and Niels Drost and Paul Earnshaw and Birgit Hassler '
    'and Nikolay Koldunov and Bill Little and Saskia Loosveldt Tomas '
    'and Klaus Zimmermann},\n'
    '\ttitle = {{ESMValTool} v2.0 '
    '{\\&}amp$\\mathsemicolon${\\#}8211$\\mathsemicolon$ '
    'Technical overview}\n'
    '}\n'
)


def _write_citation_file(filename, provenance):
    """
    Write citation information provided by the recorded provenance.

    Recipe and cmip6 data references are saved into one bibtex file.
    cmip6 data references are provided by CMIP6 data citation service.
    each cmip6 data reference has a json link. In the case of internet
    connection, cmip6 data references are saved into a bibtex file.
    Otherwise, cmip6 data reference links are saved into a text file.
    """
    product_name = os.path.splitext(filename)[0]
    info_urls = []
    json_urls = []
    product_tags = []
    for item in provenance.records:
        reference_attr = item.get_attribute('attribute:references')
        # get cmip6 citation info
        value = item.get_attribute('attribute:mip_era')
        if 'CMIP6' in value:
            url_prefix = _make_url_prefix(item.attributes)
            info_urls.append(_make_info_url(url_prefix))
            json_urls.append(_make_json_url(url_prefix))
        if reference_attr:
            # get recipe citation tags
            if item.identifier.namespace.prefix == 'recipe':
                product_tags.extend(reference_attr)
            # get diagnostics citation tags
            elif item.get_attribute('attribute:script_file'):
                product_tags.extend(reference_attr)
            else:
                info_urls += list(reference_attr)

    _save_citation_info(product_name, product_tags, json_urls, info_urls)


def _save_citation_info(product_name, product_tags, json_urls, info_urls):
    citation_entries = [ESMVALTOOL_PAPER]

    # save CMIP6 url_info, if any
    # save any refrences info that is not related to recipe or diagnostics
    title = [
        "Some citation information are found, "
        "which are not mentioned in the recipe or diagnostic."
    ]
    if info_urls:
        with open(f'{product_name}_data_citation_info.txt', 'w') as file:
            file.write('\n'.join(title + list(set(info_urls))))

    # convert json_urls to bibtex entries
    for json_url in json_urls:
        cmip_citation = _collect_cmip_citation(json_url)
        if cmip_citation:
            citation_entries.append(cmip_citation)

    # convert tags to bibtex entries
    if REFERENCES_PATH and product_tags:
        tags = _extract_tags(product_tags)
        for tag in tags:
            citation_entries.append(_collect_bibtex_citation(tag))

    with open(f'{product_name}_citation.bibtex', 'w') as file:
        file.write('\n'.join(citation_entries))


def _extract_tags(tags):
    """Extract tags that are recorded by provenance,
    as for example, "['acknow_project', 'acknow_author']".
    """
    pattern = re.compile(r'\w+')
    return list(set(pattern.findall(str(tags))))


def _get_response(url):
    """Return information from CMIP6 Data Citation service in json format."""
    json_data = False
    if url.lower().startswith('https'):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                json_data = response.json()
            else:
                logger.warning('Error in the CMIP6 citation link: %s', url)
        except IOError:
            logger.info(
                'No network connection,'
                'unable to retrieve CMIP6 citation information'
            )
    return json_data


def _json_to_bibtex(data):
    """Make a bibtex entry from CMIP6 Data Citation json data."""
    url = 'url not found'
    title = data.get('titles', ['title not found'])[0]
    publisher = data.get('publisher', 'publisher not found')
    year = data.get('publicationYear', 'publicationYear not found')
    authors = 'creators not found'
    doi = 'doi not found'

    if data.get('creators', ''):
        author_list = [
            item.get('creatorName', '') for item in data['creators']
        ]
        authors = ' and '.join(author_list)
        if not authors:
            authors = 'creators not found'

    if data.get('identifier', ''):
        doi = data.get('identifier').get('id', 'doi not found')
        url = f'https://doi.org/{doi}'

    bibtex_entry = textwrap.dedent(
        f"""
        @misc{{{url},
        \turl = {{{url}}},
        \ttitle = {{{title}}},
        \tpublisher = {{{publisher}}},
        \tyear = {year},
        \tauthor = {{{authors}}},
        \tdoi = {{{doi}}},
        }}
        """
    )
    return bibtex_entry


def _collect_bibtex_citation(tag):
    """Collect information from bibtex files."""
    bibtex_file = REFERENCES_PATH / f'{tag}.bibtex'
    if bibtex_file.is_file():
        entry = bibtex_file.read_text()
    else:
        logger.warning(
            'The reference file %s does not exist.', bibtex_file
        )
        entry = ''
    return entry


def _collect_cmip_citation(json_url):
    """Collect information from CMIP6 Data Citation Service."""
    json_data = _get_response(json_url)
    if json_data:
        bibtex_entry = _json_to_bibtex(json_data)
    else:
        bibtex_entry = ''
    return bibtex_entry


def _make_url_prefix(attribute):
    """Make url prefix based on CMIP6 Data Citation Service."""
    # the order of keys is important
    localpart = {
        'mip_era': '',
        'activity_id': '',
        'institution_id': '',
        'source_id': '',
        'experiment_id': '',
    }
    for key, value in attribute:
        if key.localpart in localpart:
            localpart[key.localpart] = value
    url_prefix = '.'.join(localpart.values())
    return url_prefix


def _make_json_url(url_prefix):
    """Make json url based on CMIP6 Data Citation Service."""
    json_url = f'{CMIP6_URL_STEM}/cerarest/exportcmip6?input={url_prefix}'
    return json_url


def _make_info_url(url_prefix):
    """Make info url based on CMIP6 Data Citation Service."""
    info_url = f'{CMIP6_URL_STEM}/cmip6?input=CMIP6.{url_prefix}'
    return info_url
