import os
import unittest
import pandas as pd
import text2term
from text2term import OntologyTermType
from text2term import Mapper
from text2term import OntologyTermCollector

pd.set_option('display.max_columns', None)


class Text2TermTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Text2TermTestSuite, cls).setUpClass()
        print("Setting up test suite global variables...")
        cls.EFO_URL = "https://github.com/EBISPOT/efo/releases/download/v3.57.0/efo.owl"
        cls.SOURCE_TERM_ID_COLUMN = "Source Term ID"
        cls.MAPPED_TERM_CURIE_COLUMN = "Mapped Term CURIE"
        cls.MAPPING_SCORE_COLUMN = "Mapping Score"
        cls.TAGS_COLUMN = "Tags"

    @classmethod
    def tearDownClass(cls):
        super(Text2TermTestSuite, cls).tearDownClass()
        text2term.clear_cache()

    def test_caching_ontology_from_url(self):
        # Test caching an ontology loaded from a URL
        print("Test caching an ontology loaded from a URL...")
        efo_cache = text2term.cache_ontology(ontology_url=self.EFO_URL, ontology_acronym="EFO")
        print(f"Cache exists: {efo_cache.cache_exists()}\n")
        assert efo_cache.cache_exists() is True

        print("Test using the returned ontology cache object to map a list of terms...")
        mappings_efo_cache = efo_cache.map_terms(["asthma", "disease location", "food allergy"],
                                                 term_type=OntologyTermType.ANY)
        assert mappings_efo_cache.size > 0

    def test_caching_ontology_from_acronym(self):
        # Test caching an ontology by resolving its acronym using bioregistry
        print("Test caching an ontology by resolving its acronym using bioregistry...")
        clo_cache = text2term.cache_ontology(ontology_url="CLO", ontology_acronym="CLO")
        print(f"Cache exists: {clo_cache.cache_exists()}\n")
        assert clo_cache.cache_exists() is True

    def test_caching_ontology_set(self):
        ontology_registry_filepath = os.path.join("..", "text2term", "resources", "ontologies.csv")
        nr_ontologies_in_registry = len(pd.read_csv(ontology_registry_filepath))

        # Test caching the set of ontologies specified in resources/ontologies.csv
        caches = text2term.cache_ontology_set(ontology_registry_filepath)
        assert len(caches) == nr_ontologies_in_registry

    def test_mapping_to_cached_ontology(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a list of terms to EFO loaded from cache
        print("Test mapping a list of terms to EFO loaded from cache...")
        mappings_efo_cache = text2term.map_terms(["asthma", "disease location", "food allergy"], target_ontology="EFO",
                                                 use_cache=True, term_type=OntologyTermType.ANY)
        print(f"{mappings_efo_cache}\n")
        assert mappings_efo_cache.size > 0

        # Test mapping a list of terms to EFO loaded from a URL
        print("Test mapping a list of terms to EFO loaded from a URL...")
        mappings_efo_url = text2term.map_terms(["asthma", "disease location", "food allergy"],
                                               target_ontology=self.EFO_URL, term_type=OntologyTermType.ANY)
        print(f"{mappings_efo_url}\n")
        assert mappings_efo_url.size > 0

        # Test that mapping to cached ontology is the same as to ontology loaded from its URL
        print("Test that mapping to cached ontology is the same as to ontology loaded from its URL...")
        mappings_match = self.check_df_equals(self.drop_source_term_ids(mappings_efo_cache),
                                              self.drop_source_term_ids(mappings_efo_url))
        print(f"...{mappings_match}")
        assert mappings_match is True

    def test_mapping_to_cached_ontology_using_syntactic_mapper(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a list of terms to cached EFO using Jaro-Winkler syntactic similarity metric
        print("Test mapping a list of terms to cached ontology using Jaro-Winkler syntactic similarity metric...")
        df = text2term.map_terms(["asthma", "disease location", "food allergy"], "EFO", use_cache=True,
                                 mapper=text2term.Mapper.JARO_WINKLER, term_type=OntologyTermType.ANY)
        print(f"{df}\n")
        assert df.size > 0

    def test_mapping_using_ontology_acronym(self):
        # Test mapping a list of terms by specifying the target ontology acronym, which gets resolved by bioregistry
        print(
            "Test mapping a list of terms to EFO by specifying the ontology acronym, which gets resolved by bioregistry")
        df2 = text2term.map_terms(["contains", "asthma"], "MONDO", term_type=OntologyTermType.CLASS)
        print(f"{df2}\n")
        assert df2.size > 0

    def test_mapping_tagged_terms(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a dictionary of tagged terms to cached EFO, and include unmapped terms in the output
        print("Test mapping a dictionary of tagged terms to cached EFO, and include unmapped terms in the output...")
        df3 = text2term.map_terms(
            {"asthma": "disease", "allergy": ["ignore", "response"], "protein level": ["measurement"],
             "isdjfnsdfwd": None}, target_ontology="EFO", excl_deprecated=True, use_cache=True, incl_unmapped=True)
        print(f"{df3}\n")
        assert df3.size > 0
        assert df3[self.TAGS_COLUMN].str.contains("disease").any()
        assert df3[self.TAGS_COLUMN].str.contains("measurement").any()

    def test_preprocessing_from_file(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test processing tagged terms where the tags are provided in a file
        print("Test processing tagged terms where the tags are provided in a file...")
        tagged_terms = text2term.preprocess_tagged_terms("simple_preprocess.txt")
        df4 = text2term.map_terms(tagged_terms, target_ontology="EFO", use_cache=True, incl_unmapped=True)
        print(f"{df4}\n")
        assert df4.size > 0
        assert df4[self.TAGS_COLUMN].str.contains("disease").any()
        assert df4[self.TAGS_COLUMN].str.contains("important").any()

    def test_mapping_to_properties(self):
        # Test mapping a list of properties to EFO loaded from a URL and restrict search to properties
        print("Test mapping a list of properties to EFO loaded from a URL and restrict search to properties...")
        df5 = text2term.map_terms(source_terms=["contains", "location"], target_ontology=self.EFO_URL,
                                  term_type=OntologyTermType.PROPERTY)
        print(f"{df5}\n")
        assert df5.size > 0

        # Test mapping a list of properties to EFO loaded from cache and restrict search to properties
        print("Test mapping a list of properties to EFO loaded from cache and restrict search to properties...")
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df6 = text2term.map_terms(source_terms=["contains", "location"], target_ontology="EFO", use_cache=True,
                                  term_type=OntologyTermType.PROPERTY)
        print(f"{df6}\n")
        assert df6.size > 0

        # Test that mapping to properties in cached ontology is the same as to ontology loaded from its URL
        properties_df_match = self.check_df_equals(self.drop_source_term_ids(df5), self.drop_source_term_ids(df6))
        print(f"Properties match: {properties_df_match}")
        assert properties_df_match is True

    def test_mapping_zooma_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the Zooma mapper
        print("Test mapping a list of terms to multiple ontologies using the Zooma mapper...")
        df_zooma = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                       mapper=Mapper.ZOOMA, term_type=OntologyTermType.ANY)
        print(f"{df_zooma}\n")
        assert df_zooma.size > 0
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("EFO:").any()
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    def test_mapping_bioportal_ontologies_no_apikey(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper without API Key
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY)
        assert df_bioportal.empty is True

    def test_mapping_bioportal_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY,
                                           bioportal_apikey="8f0cbe43-2906-431a-9572-8600d3f4266e")
        print(f"{df_bioportal}\n")
        assert df_bioportal.size > 0
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("EFO:").any()
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    def test_term_collector(self):
        expected_nr_efo_terms = 50867
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms()
        assert len(terms) == expected_nr_efo_terms

    def test_term_collector_classes_only(self):
        expected_nr_efo_classes = 50643
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(term_type=OntologyTermType.CLASS)
        assert len(terms) == expected_nr_efo_classes

    def test_term_collector_properties_only(self):
        expected_nr_efo_properties = 224
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(term_type=OntologyTermType.PROPERTY)
        assert len(terms) == expected_nr_efo_properties

    def test_term_collector_iri_limit(self):
        efo_base_iri = "http://www.ebi.ac.uk/efo/"
        expected_nr_terms_with_efo_iri = 17383
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(base_iris=[efo_base_iri], term_type=OntologyTermType.ANY)
        assert len(terms) == expected_nr_terms_with_efo_iri

    def test_term_collector_iri_limit_properties_only(self):
        efo_base_iri = "http://www.ebi.ac.uk/efo/"
        expected_nr_properties_with_efo_iri = 29
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(base_iris=[efo_base_iri], term_type=OntologyTermType.PROPERTY)
        assert len(terms) == expected_nr_properties_with_efo_iri

    def test_mapping_with_min_score_filter(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        min_score = 0.6
        search_terms = ["asthma attack", "location"]

        print("Test mapping to cached EFO using Zooma mapper and min_score filter...")
        df_zooma = text2term.map_terms(search_terms, target_ontology="EFO,NCIT", mapper=Mapper.ZOOMA,
                                       term_type=OntologyTermType.ANY, min_score=min_score)
        assert (df_zooma[self.MAPPING_SCORE_COLUMN] >= min_score).all()

        print("Test mapping to cached EFO using TFIDF similarity metric and min_score filter...")
        df_tfidf = text2term.map_terms(search_terms, target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                       term_type=OntologyTermType.ANY, min_score=min_score)
        assert (df_tfidf[self.MAPPING_SCORE_COLUMN] >= min_score).all()

        print("Test mapping to cached EFO using Levenshtein similarity metric and min_score filter...")
        df_leven = text2term.map_terms(search_terms, target_ontology="EFO", use_cache=True, mapper=Mapper.LEVENSHTEIN,
                                       term_type=OntologyTermType.ANY, min_score=min_score)
        assert (df_leven[self.MAPPING_SCORE_COLUMN] >= min_score).all()

    def test_mapping_with_min_score_filter_empty_results(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        print("Test mapping to EFO using TFIDF similarity metric and min_score filter that results in no mappings...")
        df_tfidf = text2term.map_terms(["carbon monoxide"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                       term_type=OntologyTermType.ANY, min_score=0.99)
        assert df_tfidf.empty is True

    def test_include_unmapped_terms(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df = text2term.map_terms(["asthma", "margarita"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 incl_unmapped=True, min_score=0.8)
        assert df[self.TAGS_COLUMN].str.contains("unmapped").any()

    def test_include_unmapped_terms_when_mappings_df_is_empty(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df = text2term.map_terms(["mojito", "margarita"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 incl_unmapped=True, min_score=0.8)
        assert df[self.TAGS_COLUMN].str.contains("unmapped").any()

    def drop_source_term_ids(self, df):
        # Unless specified, source term IDs are randomly generated UUIDs. We have to drop the ID column to be able to
        # get a meaningful diff between two dataframes. Otherwise, the dataframes would always differ because of the IDs
        return df.drop(self.SOURCE_TERM_ID_COLUMN, axis=1)

    def check_df_equals(self, df, expected_df):
        # Use pandas::assert_frame_equal function to determine if two data frames are equal
        pd.testing.assert_frame_equal(df, expected_df, check_names=False, check_like=True)
        return True

    def ensure_cache_exists(self, ontology_name, ontology_url):
        if not text2term.cache_exists(ontology_name):
            text2term.cache_ontology(ontology_url=ontology_url, ontology_acronym=ontology_name)


if __name__ == '__main__':
    unittest.main()
