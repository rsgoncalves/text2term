import logging
import re
import sys
import pandas as pd
import bioregistry
import shortuuid
from owlready2 import *
from gensim.parsing import strip_non_alphanum, strip_multiple_whitespaces

STOP_WORDS = {'in', 'the', 'any', 'all', 'for', 'and', 'or', 'dx', 'on', 'fh', 'tx', 'only', 'qnorm', 'w', 'iqb',
              'ds', 'rd', 'rdgwas', 'average', 'weekly', 'monthly', 'daily'}

BASE_IRI = "http://ccb.hms.harvard.edu/t2t/"


def normalize_list(token_list):
    normalized_token_list = []
    for token in token_list:
        normalized_token_list.append(normalize(token))
    return normalized_token_list


def normalize(token):
    """
    Normalizes a given string by converting to lower case, removing non-word characters, stop words, white space
    :param token: Text to be normalized
    :return: Normalized string
    """
    token = re.sub(r"[\(\[].*?[\)\]]", "", token)  # remove text within parenthesis/brackets
    token = strip_non_alphanum(token).lower()
    token = token.replace("_", " ")
    token = " ".join(w for w in token.split() if w not in STOP_WORDS)
    token = strip_multiple_whitespaces(token)
    return token


def curie_from_iri(iri):
    return bioregistry.curie_from_iri(iri)


def label_from_iri(iri):
    if "#" in iri:
        return iri.split("#")[1]
    else:
        return iri.rsplit('/', 1)[1]


def remove_quotes(text):
    text = text.replace("\"", "")
    text = text.replace("\'", "")
    return text


def get_logger(name, level):
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def parse_list_file(file_path):
    file = open(file_path)
    lines = file.read().splitlines()
    file.close()
    return lines


def parse_csv_file(file_path, term_column_name, term_id_column_name, separator=','):
    data = pd.read_csv(file_path, sep=separator)
    if term_column_name not in data.columns:
        raise ValueError("Could not find specified column name for input terms: " + term_column_name)
    terms = data[term_column_name].values
    if term_id_column_name not in data.columns:
        term_ids = generate_iris(len(terms))
    elif data[term_id_column_name].isnull().values.all():
        term_ids = generate_iris(len(terms))
    else:
        term_ids = data[term_id_column_name].values
    return terms, term_ids


def get_ontology_from_labels(term_labels):
    onto_iri = BASE_IRI + "Ontology-" + generate_uuid()
    onto = owlready2.get_ontology(onto_iri)
    onto.metadata.comment.append("Created dynamically using text2term")
    onto.metadata.comment.append(datetime.datetime.now())
    for term_label in term_labels:
        with onto:
            new_term_iri = generate_iri()
            new_term = types.new_class(new_term_iri, (Thing,))
            new_term.label = term_label
    return onto


def generate_uuid():
    return str(shortuuid.ShortUUID().random(length=10))


def generate_iri():
    return BASE_IRI + "R" + generate_uuid()


def generate_iris(quantity):
    return [generate_iri() for _ in range(quantity)]


OBO_BASE_IRI = "http://purl.obolibrary.org/obo/"
BIOPORTAL_BASE_IRI = "http://purl.bioontology.org/ontology/"
ORPHANET_IRI = "http://www.orpha.net/ORDO/"
ONTOLOGY_IRIS = {"EFO": "http://www.ebi.ac.uk/efo/",
                 "Orphanet": ORPHANET_IRI,
                 "ORPHA": ORPHANET_IRI,
                 "MONDO": OBO_BASE_IRI,
                 "HP": OBO_BASE_IRI,
                 "UBERON": OBO_BASE_IRI,
                 "GO": OBO_BASE_IRI,
                 "DOID": OBO_BASE_IRI,
                 "CHEBI": OBO_BASE_IRI,
                 "OMIT": OBO_BASE_IRI,
                 "NCIT": OBO_BASE_IRI,
                 "MAXO": OBO_BASE_IRI,
                 "DRON": OBO_BASE_IRI,
                 "OAE": OBO_BASE_IRI,
                 "CIDO": OBO_BASE_IRI,
                 "OMIM": BIOPORTAL_BASE_IRI + "OMIM/",
                 "PATO": OBO_BASE_IRI,
                 "SNOMED": "http://snomed.info/id/"}
