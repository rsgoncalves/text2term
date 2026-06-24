"""Provides OntologyTermCollector class"""

import logging
import bioregistry
from owlready2 import *

from text2term import onto_utils
from text2term.term import OntologyTerm, OntologyTermType

# Matches the IRI reported by owlready2's namespace.py _load_properties() when it
# encounters "punning" (the same IRI declared as both a property and a class/individual),
# e.g.: "'http://purl.obolibrary.org/obo/STATO_0000416' belongs to more than one entity
# types (cannot be both a property and a class/an individual)!"
_PUNNING_ERROR_IRI_PATTERN = re.compile(r"^'(.+?)' belongs to more than one entity types")


class OntologyTermCollector:

    def __init__(self, ontology_iri, use_reasoning=False, log_level=logging.INFO):
        """
        Construct an ontology term collector for the ontology at the given IRI
        :param ontology_iri: IRI of the ontology (e.g., path of ontology document in the local file system, URL)
        :param use_reasoning: Use a reasoner to compute inferred class hierarchy
        """
        self.logger = onto_utils.get_logger(__name__, level=log_level)
        self.ontology = self._load_ontology(ontology_iri)
        if use_reasoning:
            self._classify_ontology(self.ontology)

    def get_ontology_terms(self, base_iris=(), exclude_deprecated=False, term_type=OntologyTermType.ANY,
                           include_related_synonyms=False, include_broad_synonyms=False):
        """
        Collect the terms described in the ontology at the specified IRI
        :param base_iris: Limit ontology term collection to terms whose IRIs start with any IRI given in this tuple
        :param exclude_deprecated: Exclude ontology terms stated as deprecated using owl:deprecated 'true'
        :param term_type: Type of term--can be 'class' or 'property' or 'any' (individuals may be added in the future)
        :param include_related_synonyms: Include related synonyms of each term, specified via oboInOwl#hasRelatedSynonym
        :param include_broad_synonyms: Include broad synonyms of each term, specified via oboInOwl#hasBroadSynonym
        :return: Dictionary of ontology term IRIs and their respective details in the specified ontology
        """
        self.logger.info("Collecting ontology term details...")
        start = time.time()
        ontology_terms = dict()
        if len(base_iris) > 0:
            for iri in base_iris:
                iri = iri.strip()
                query = iri + "*"
                self.logger.info("...collecting terms with IRIs starting in: " + iri)
                iris = list(default_world.search(iri=query))
                ontology_terms = (ontology_terms |
                                  self._get_ontology_terms(iris, self.ontology, exclude_deprecated, term_type,
                                                           include_related_synonyms=include_related_synonyms,
                                                           include_broad_synonyms=include_broad_synonyms))
        else:
            ontology_signature = self._get_ontology_signature(self.ontology)
            ontology_terms = self._get_ontology_terms(term_list=ontology_signature, ontology=self.ontology,
                                                      exclude_deprecated=exclude_deprecated, term_type=term_type,
                                                      include_related_synonyms=include_related_synonyms,
                                                      include_broad_synonyms=include_broad_synonyms)
        end = time.time()
        self.logger.info("...done: collected %i ontology terms (collection time: %.2fs)", len(ontology_terms),
                         end - start)
        return ontology_terms

    def _get_ontology_signature(self, ontology):
        signature = list(ontology.classes())
        signature.extend(list(ontology.properties()))
        # owlready2::ontology.classes() does not include classes in imported ontologies; we need to explicitly add them
        for imported_ontology in ontology.imported_ontologies:
            signature.extend(list(imported_ontology.classes()))
            signature.extend(list(imported_ontology.properties()))
        return signature

    def _get_ontology_terms(self, term_list, ontology, exclude_deprecated, term_type,
                            include_related_synonyms=False, include_broad_synonyms=False):
        ontology_terms = dict()
        for ontology_term in term_list:
            # Parse if should include ontology classes, properties, or both
            include = _filter_term_type(ontology_term, term_type, False)
            if include and ontology_term is not Thing and ontology_term is not Nothing:
                if (exclude_deprecated and not deprecated[ontology_term]) or (not exclude_deprecated):
                    iri = ontology_term.iri
                    labels = self._get_labels(ontology_term)
                    synonyms = self._get_synonyms(ontology_term, include_related_synonyms=include_related_synonyms,
                                                  include_broad_synonyms=include_broad_synonyms)
                    named_parents, complex_parents = self._get_parents(ontology_term)
                    children = self._get_children(ontology_term, ontology)
                    instances = self._get_instances(ontology_term, ontology)
                    definitions = self._get_definitions(ontology_term)
                    is_deprecated = deprecated[ontology_term] == [True]
                    if _filter_term_type(ontology_term, OntologyTermType.CLASS, False):
                        owl_term_type = OntologyTermType.CLASS
                    elif _filter_term_type(ontology_term, OntologyTermType.PROPERTY, False):
                        owl_term_type = OntologyTermType.PROPERTY
                    else:
                        owl_term_type = "undetermined"
                        self.logger.warning("Term has undetermined type %s %s", iri, labels)
                    if owl_term_type != "undetermined":
                        term_details = OntologyTerm(iri, labels, definitions=definitions, synonyms=synonyms,
                                                    parents=named_parents, children=children, instances=instances,
                                                    restrictions=complex_parents, deprecated=is_deprecated,
                                                    term_type=owl_term_type)
                        ontology_terms[iri] = term_details
                else:
                    self.logger.debug("Excluding deprecated ontology term: %s", ontology_term.iri)
        return ontology_terms

    def _get_parents(self, ontology_term):
        parents = dict()  # named/atomic superclasses except owl:Thing
        restrictions = dict()  # restrictions are class expressions such as 'pancreatitis disease_has_location pancreas'
        try:
            all_parents = ontology_term.is_a  # obtain direct parents of this entity
            for parent in all_parents:
                # exclude owl:Thing and Self
                if parent is not Thing and parent is not ontology_term:
                    if isinstance(parent, ThingClass):  # get named parents (i.e. classes with IRIs)
                        self._add_named_parent(parent, parents)
                    elif isinstance(parent, And):  # get conjuncts and add them to the respective structures
                        for conjunct in parent.Classes:
                            if isinstance(conjunct, ThingClass):  # if conjunct is a named class, add it to parents dict
                                self._add_named_parent(conjunct, parents)
                            else:
                                self._add_complex_parent(conjunct, restrictions)
                    elif isinstance(parent, Restriction):  # get complex parents, i.e. restrictions or class expressions
                        self._add_complex_parent(parent, restrictions)
        except (AttributeError, ValueError) as err:
            self.logger.debug(err)
        return parents, restrictions

    def _add_named_parent(self, parent, parents):
        if len(parent.label) > 0:
            parents.update({parent.iri: parent.label[0]})
        else:
            parents.update({parent.iri: onto_utils.label_from_iri(parent.iri)})

    def _add_complex_parent(self, parent, restrictions):
        property_iri = parent.property.iri
        if isinstance(parent.value, ThingClass):  # the filler is a named term (i.e., it has an IRI)
            value = parent.value.iri
        else:  # the filler is another complex class expression
            value = parent.value
        if property_iri in restrictions.keys():
            current_restrictions = restrictions[property_iri]
            current_restrictions.add(value)
            restrictions.update({property_iri: current_restrictions})
        else:
            restrictions.update({property_iri: str(value)})

    def _get_children(self, ontology_term, ontology):
        children = dict()
        try:
            for child in ontology.get_children_of(ontology_term):
                if len(child.iri) > 0:
                    if len(child.label) > 0:
                        children.update({child.iri: child.label[0]})
                    else:
                        children.update({child.iri: onto_utils.label_from_iri(child.iri)})
        except (TypeError, AttributeError, ValueError) as err:
            self.logger.debug(err)
        return children

    def _get_instances(self, ontology_term, ontology):
        instances = dict()
        try:
            for instance in ontology.get_instances_of(ontology_term):
                if len(instance.iri) > 0:
                    if len(instance.label) > 0:
                        instances.update({instance.iri: instance.label[0]})
                    else:
                        instances.update({instance.iri: onto_utils.label_from_iri(instance.iri)})
        except AttributeError as err:
            self.logger.debug(err)
        return instances

    def _get_labels(self, ontology_term):
        """
        Collect the labels of the given ontology term both given by rdfs:label and skos:prefLabel
        :param ontology_term: Ontology term
        :return: Collection of labels of the ontology term
        """
        labels = set()
        for rdfs_label in self._get_rdfs_labels(ontology_term):
            labels.add(rdfs_label)
        for skos_label in self._get_skos_pref_labels(ontology_term):
            labels.add(skos_label)
        if len(labels) == 0:
            label_from_iri = onto_utils.label_from_iri(ontology_term.iri)
            self.logger.debug("...ontology term %s has no labels (rdfs:label or skos:prefLabel). "
                              "Using a label based on the term IRI: %s", ontology_term.iri, label_from_iri)
            labels.add(label_from_iri)
        self.logger.debug("...collected %i labels and synonyms for %s", len(labels), ontology_term)
        return labels

    def _get_synonyms(self, ontology_term, include_related_synonyms=False, include_broad_synonyms=False):
        """
        Collect the synonyms of the given ontology term
        :param ontology_term: Ontology term
        :param include_broad_synonyms: true if broad (i.e. more generic) synonyms should be included, false otherwise
        :return: Collection of synonyms of the ontology term
        """
        synonyms = set()
        for synonym in self._get_obo_exact_synonyms(ontology_term):
            synonyms.add(synonym)
        for nci_synonym in self._get_nci_synonyms(ontology_term):
            synonyms.add(nci_synonym)
        for efo_alt_term in self._get_efo_alt_terms(ontology_term):
            synonyms.add(efo_alt_term)
        if include_related_synonyms:
            for synonym in self._get_obo_related_synonyms(ontology_term):
                synonyms.add(synonym)
        if include_broad_synonyms:
            for synonym in self._get_obo_broad_synonyms(ontology_term):
                synonyms.add(synonym)
        self.logger.debug("...collected %i synonyms for %s", len(synonyms), ontology_term)
        return synonyms

    def _get_rdfs_labels(self, ontology_term):
        """
        Collect labels of the given term that are specified using the standard rdfs:label annotation property
        :param ontology_term: Ontology term to collect labels from
        :return: Collection of RDFS labels
        """
        rdfs_labels = []
        try:
            for rdfs_label in ontology_term.label:
                rdfs_labels.append(rdfs_label)
        except (AttributeError, ValueError) as err:
            self.logger.debug(err)
        return rdfs_labels

    def _get_skos_pref_labels(self, ontology_term):
        """
        Collect labels of the given term that are specified using the skos:prefLabel annotation property
        :param ontology_term: Ontology term to collect labels from
        :return: Collection of SKOS preferred labels
        """
        skos_labels = []
        try:
            for skos_pref_label in ontology_term.prefLabel:
                skos_labels.append(skos_pref_label)
        except AttributeError as err:
            self.logger.debug(err)
        return skos_labels

    def _get_efo_alt_terms(self, ontology_term):
        efo_alt_terms = []
        try:
            for efo_alt_term in ontology_term.alternative_term:
                efo_alt_terms.append(efo_alt_term)
        except AttributeError as err:
            self.logger.debug(err)
        return efo_alt_terms

    def _get_obo_exact_synonyms(self, ontology_term):
        """
        Collect exact synonyms of the given term that are specified using the annotation property:
            <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym>.
        :param ontology_term: Ontology term to collect exact synonyms from
        :return: Collection of exact synonyms
        """
        synonyms = []
        try:
            for synonym in ontology_term.hasExactSynonym:
                if hasattr(synonym, 'iri'):
                    synonym = synonym.iri
                synonyms.append(synonym)
        except AttributeError as err:
            self.logger.debug(err)
        return synonyms

    def _get_obo_related_synonyms(self, ontology_term):
        """
        Collect related synonyms of the given term that are specified using the annotation property:
            <http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym>.
        :param ontology_term: Ontology term to collect related synonyms from
        :return: Collection of related synonyms
        """
        synonyms = []
        try:
            for synonym in ontology_term.hasRelatedSynonym:
                if hasattr(synonym, 'iri'):
                    synonym = synonym.iri
                synonyms.append(synonym)
        except AttributeError as err:
            self.logger.debug(err)
        return synonyms

    def _get_obo_broad_synonyms(self, ontology_term):
        """
        Collect broad synonyms of the given term that are specified using the annotation property:
            <http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym>.
        :param ontology_term: Ontology term to collect broad synonyms from
        :return: Collection of broad synonyms
        """
        synonyms = []
        try:
            for synonym in ontology_term.hasBroadSynonym:
                if hasattr(synonym, 'iri'):
                    synonym = synonym.iri
                synonyms.append(synonym)
        except AttributeError as err:
            self.logger.debug(err)
        return synonyms

    def _get_nci_synonyms(self, ontology_term):
        """
        Collect synonyms of the given term that are specified using the NCI Thesaurus annotation property:
         <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P90>.
        :param ontology_term: Ontology term to collect synonyms from
        :return: Collection of synonyms
        """
        nci_synonyms = []
        try:
            for synonym in ontology_term.P90:
                nci_synonyms.append(synonym)
        except AttributeError as err:
            self.logger.debug(err)
        return nci_synonyms

    def _get_definitions(self, ontology_term):
        """
         Get definitions (if any exist) of the given term as specified using either the skos:definition annotation
         property or the IAO_0000115 ('definition') property
         :param ontology_term: Ontology term to collect definition of
         :return: Set of term definition strings
         """
        definitions = set()
        for definition in self._get_skos_definition(ontology_term):
            definitions.add(definition)
        for definition in self._get_iao_definition(ontology_term):
            definitions.add(definition)
        return definitions

    def _get_iao_definition(self, ontology_term):
        definition = ""
        try:
            definition = ontology_term.IAO_0000115
        except AttributeError as err:
            self.logger.debug(err)
        return definition

    def _get_skos_definition(self, ontology_term):
        definition = ""
        try:
            definition = ontology_term.definition
        except AttributeError as err:
            self.logger.debug(err)
        return definition

    def _load_ontology(self, ontology_iri):
        """
        Load the ontology at the specified IRI.

        Some ontologies (e.g., CL, which imports STATO/OBI) contain "punned" entities--IRIs
        that are declared as both a property and a class/individual. Owlready2 enforces OWL DL
        and refuses to load such ontologies: internally, Ontology.load() parses the RDF/XML
        document into the world's quadstore, then calls self._load_properties(), and--only
        afterwards--loads imported ontologies and runs any ontology-declared Python module hooks.
        A punning TypeError raised inside _load_properties() would normally abort load() entirely,
        skipping those later steps.

        To recover without losing those later steps, we temporarily wrap (monkey-patch) the
        Ontology._load_properties method on the class itself for the duration of this call. The
        wrapper calls the original implementation; if it raises the punning TypeError, the wrapper
        strips the offending entity's triples directly from the quadstore and retries
        _load_properties() in a loop (an ontology can have any number of punned terms, often pulled
        in via different imports, and we keep going until they're all gone), then returns normally.
        From load()'s perspective, _load_properties() simply succeeded, so it proceeds to load
        imports etc. as usual. The patch is removed (restoring the original method) in a finally
        block, regardless of outcome.

        The loop has no cap on how many punned entities it will strip. The only thing it guards against
        is the same IRI being reported again right after being stripped, which would indicate the
        strip didn't actually take effect (a bug, or an IRI that owlready2 keeps regenerating); that
        case raises immediately rather than looping forever.

        :param ontology_iri: IRI of the ontology (e.g., path of ontology document in the local file system, URL)
        :return: Ontology document
        """
        self.logger.info("Loading ontology %s...", ontology_iri)
        start = time.time()
        owl_link = bioregistry.get_owl_download(ontology_iri)
        if owl_link is not None:
            ontology_iri = owl_link
        removed_iris = []

        original_load_properties = Ontology._load_properties

        def _patched_load_properties(onto_self):
            last_punned_iri = None
            while True:
                try:
                    return original_load_properties(onto_self)
                except TypeError as err:
                    punned_iri = _extract_punning_iri(err)
                    if punned_iri is None:
                        raise  # not a punning error we know how to recover from
                    if punned_iri == last_punned_iri:
                        self.logger.error("Stripping '%s' did not resolve its punning error; giving up "
                                          "to avoid looping forever.", punned_iri)
                        raise
                    self.logger.warning("Ontology contains a punned entity (declared as both a property "
                                        "and a class/individual): %s. Removing it and retrying.", punned_iri)
                    _strip_punned_entity(onto_self.world, punned_iri)
                    removed_iris.append(punned_iri)
                    last_punned_iri = punned_iri

        Ontology._load_properties = _patched_load_properties
        try:
            ontology = get_ontology(ontology_iri).load()
        finally:
            Ontology._load_properties = original_load_properties

        end = time.time()
        if removed_iris:
            self.logger.warning("Removed %i punned entit%s prior to loading: %s", len(removed_iris),
                                "y" if len(removed_iris) == 1 else "ies", removed_iris)
        self._log_ontology_metrics(ontology)
        self.logger.info("...done (ontology loading time: %.2fs)", end - start)
        return ontology

    def _classify_ontology(self, ontology):
        """
        Perform reasoning over the given ontology (consistency checking and classification)
        :param ontology: ontology instance
        """
        self.logger.info("Reasoning over ontology...")
        start = time.time()
        with ontology:  # entailments will be added to this ontology
            sync_reasoner(infer_property_values=True, ignore_unsupported_datatypes=True)
        end = time.time()
        self.logger.info("...done (reasoning time: %.2fs)", end - start)

    def close(self):
        # when multiple ontologies are loaded with owlready2, and they reference the same ontology term (IRI), a lookup
        # for that IRI returns the term from the first ontology loaded —> need to unload previously loaded ontologies
        try:
            self.ontology.destroy()
        except Exception as err:
            self.logger.debug("Unable to destroy ontology: ", err)

    def _log_ontology_metrics(self, ontology):
        self.logger.debug(" Ontology IRI: %s", ontology.base_iri)
        self.logger.debug(" Class count: %i", len(list(ontology.classes())))
        self.logger.debug(" Object property count: %i", len(list(ontology.object_properties())))
        self.logger.debug(" Data property count: %i", len(list(ontology.data_properties())))
        self.logger.debug(" Annotation property count: %i", len(list(ontology.annotation_properties())))


def _extract_punning_iri(error):
    """
    Parse the offending IRI out of the TypeError raised by owlready2's
    namespace.py::_load_properties() when it detects punning.
    :param error: the TypeError raised by owlready2
    :return: the punned entity's IRI, or None if the error message doesn't match the expected pattern
    """
    match = _PUNNING_ERROR_IRI_PATTERN.match(str(error))
    return match.group(1) if match else None


def _strip_punned_entity(world, iri):
    """
    Remove the given (punned) IRI from the ontology so that a subsequent retry of
    _load_properties() no longer sees it as declared as more than one entity type.

    Tries two strategies, in order:
      1. destroy_entity() -- owlready2's public, documented API for removing an entity and all
         its triples from the quadstore. This works here because by the time _load_properties()
         raises, the class/individual-side triples for the punned IRI have already been parsed
         and a Python entity object usually already exists for it (world[iri] / _get_by_storid);
         it is only the *property*-side interpretation that fails to materialize. If such an
         object exists, destroying it removes all of the IRI's triples (including the property
         declaration), which is exactly what we need.
      2. Direct deletion from the SQLite-backed quadstore (world.graph) as a fallback, in case no
         Python object could be resolved for the IRI. This reaches into internals that are less
         API-stable across owlready2 versions, so it is only used if strategy 1 doesn't apply.

    :param world: owlready2 World (or ontology.world) whose quadstore should be modified
    :param iri: IRI (string) of the punned entity to remove
    """
    entity = world[iri]  # World supports dict-like IRI lookup; returns None if no Python object exists yet
    if entity is not None:
        destroy_entity(entity)
        return

    # Fallback: no Python-side object could be resolved (e.g. only a bare property triple exists).
    # Delete the IRI's triples directly from the quadstore.
    try:
        storid = world._abbreviate(iri, False)
    except AttributeError as err:
        raise RuntimeError(
            "Could not resolve internal storage ID for IRI '%s' while trying to recover from a "
            "punning error (destroy_entity() also found no Python object for it). owlready2's "
            "internal API may have changed; inspect world._abbreviate() / world.graph in your "
            "installed version." % iri) from err
    if storid is None:
        # Entity not found in the store under this IRI; nothing to strip.
        return
    try:
        world.graph.execute("DELETE FROM quads WHERE s=? OR o=?", (storid, storid))
        world.graph.db.commit()
    except AttributeError as err:
        raise RuntimeError(
            "Could not delete triples for IRI '%s' while trying to recover from a punning error. "
            "owlready2's internal quadstore API may have changed; inspect world.graph in your "
            "installed version." % iri) from err


def filter_terms(onto_terms, iris=(), excl_deprecated=False, term_type=OntologyTermType.ANY):
    filtered_onto_terms = {}
    for base_iri, term in onto_terms.items():
        if isinstance(iris, str):
            begins_with_iri = (iris == ()) or base_iri.startswith(iris)
        else:
            begins_with_iri = (iris == ()) or any(base_iri.startswith(iri) for iri in iris)
        is_not_deprecated = (not excl_deprecated) or (not term.deprecated)
        include = _filter_term_type(term, term_type, True)
        if begins_with_iri and is_not_deprecated and include:
            filtered_onto_terms.update({base_iri: term})
    return filtered_onto_terms


def _filter_term_type(ontology_term, term_type, cached):
    if term_type == OntologyTermType.CLASS:
        if cached:
            return ontology_term.term_type == OntologyTermType.CLASS
        else:
            return isinstance(ontology_term, ThingClass)
    elif term_type == OntologyTermType.PROPERTY:
        if cached:
            return ontology_term.term_type == OntologyTermType.PROPERTY
        else:
            return isinstance(ontology_term, PropertyClass)
    elif term_type == OntologyTermType.ANY:
        return True
    else:
        raise ValueError("Invalid term-type option. Acceptable term types are: 'class' or 'property' or 'any'")
