import os
import tempfile
import unittest
import pandas as pd
import text2term
import text2term.onto_cache as onto_cache
from text2term import OntologyTermType, onto_utils
from text2term import Mapper
from text2term import OntologyTermCollector
from text2term.bioportal_mapper import BioPortalAnnotatorMapper
from text2term.syntactic_mapper import SyntacticMapper
from text2term.zooma_mapper import ZoomaMapper
from owlready2 import EntityClass, IRIS

pd.set_option('display.max_columns', None)

RUN_LIVE_API_TESTS = os.getenv("TEXT2TERM_RUN_LIVE_API_TESTS") == "1"
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY", "")
skip_unless_live_api = unittest.skipUnless(
    RUN_LIVE_API_TESTS,
    "Set TEXT2TERM_RUN_LIVE_API_TESTS=1 to run live API integration tests.",
)
skip_unless_bioportal_api = unittest.skipUnless(
    RUN_LIVE_API_TESTS and BIOPORTAL_API_KEY,
    "Set TEXT2TERM_RUN_LIVE_API_TESTS=1 and BIOPORTAL_API_KEY to run live BioPortal integration tests.",
)


class Text2TermTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Text2TermTestSuite, cls).setUpClass()
        print("Setting up test suite global variables...")
        cls.SOURCE_TERM_ID_COLUMN = "Source Term ID"
        cls.MAPPED_TERM_CURIE_COLUMN = "Mapped Term CURIE"
        cls.MAPPING_SCORE_COLUMN = "Mapping Score"
        cls.TAGS_COLUMN = "Tags"
        cls.TEST_CACHE_FOLDER = ".test_cache"
        cls.TEST_FOLDER = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_ONTOLOGY_FILEPATH = os.path.join(cls.TEST_FOLDER, "test_ontology.owl")
        cls.TEST_ONTOLOGY_ACRONYM = "T2TTEST"
        cls.TEST_ONTOLOGY_BASE_IRI = "https://text2term.ontology/test/"
        cls.TEST_SOURCE_TERMS = ["cheese pizza", "fish pizza", "tomato topping"]
        cls.TEST_PROPERTY_TERMS = ["contains topping", "location"]
        
        # Configure the SSL verification to be disabled
        if hasattr(onto_cache, 'disable_ssl_verification'):
            onto_cache.disable_ssl_verification()
        
        # Ensure test environment is set up
        cls.setup_test_environment()

    @classmethod
    def tearDownClass(cls):
        super(Text2TermTestSuite, cls).tearDownClass()
        text2term.clear_cache(cache_folder=cls.TEST_CACHE_FOLDER)
        test_output_file_without_metadata = "test_output_without_metadata.csv"
        if os.path.exists(test_output_file_without_metadata):
            os.remove(test_output_file_without_metadata)
        test_output_file_with_metadata = "test_output_with_metadata.csv"
        if os.path.exists(test_output_file_with_metadata):
            os.remove("test_output_with_metadata.csv")

    @classmethod
    def setup_test_environment(cls):
        """Set up necessary test files and directories"""
        print("Setting up test environment...")
        
        # Create test cache folder
        if not os.path.exists(cls.TEST_CACHE_FOLDER):
            os.makedirs(cls.TEST_CACHE_FOLDER)
            print(f"Created test cache folder: {cls.TEST_CACHE_FOLDER}")
        
        cls.TEST_ONTOLOGY_REGISTRY = os.path.join(cls.TEST_FOLDER, "test_ontologies.csv")
        with open(cls.TEST_ONTOLOGY_REGISTRY, 'w') as f:
            f.write("acronym,url\n")
            f.write(f"{cls.TEST_ONTOLOGY_ACRONYM},{cls.TEST_ONTOLOGY_FILEPATH}\n")
            f.write(f"{cls.TEST_ONTOLOGY_ACRONYM}_ALT,{cls.TEST_ONTOLOGY_FILEPATH}\n")

        # Create test_input_with_tags.txt for the preprocessing test
        test_input_file_path = os.path.join(cls.TEST_FOLDER, "test_input_with_tags.txt")
        if not os.path.exists(test_input_file_path):
            with open(test_input_file_path, 'w') as f:
                f.write("cheese pizza;:;pizza\n")
                f.write("fish pizza;:;seafood,topping\n")
                f.write("pizza NOS;:;unspecified\n")
                f.write("mystery topping\n")
            print(f"Created sample input file: {test_input_file_path}")
        cls.TEST_INPUT_FILEPATH = test_input_file_path

        test_templates_file_path = os.path.join(cls.TEST_FOLDER, "test_templates.txt")
        if not os.path.exists(test_templates_file_path):
            with open(test_templates_file_path, 'w') as f:
                f.write("(.*) [Nn][Oo][Ss]\n")
                f.write("(.*) due to [A-z 0-9]+\n")
            print(f"Created sample template file: {test_templates_file_path}")
        cls.TEST_TEMPLATES_FILEPATH = test_templates_file_path
        print("Test environment setup complete")

    def test_caching_ontology_from_url(self):
        print("Test caching an ontology loaded from a local file...")
        ontology_cache = text2term.cache_ontology(ontology_url=self.TEST_ONTOLOGY_FILEPATH,
                                                  ontology_acronym=self.TEST_ONTOLOGY_ACRONYM,
                                                  cache_folder=self.TEST_CACHE_FOLDER)
        print(f"Cache exists: {ontology_cache.cache_exists()}\n")
        assert ontology_cache.cache_exists() is True

        print("Test using the returned ontology cache object to map a list of terms...")
        mappings_cache = ontology_cache.map_terms(self.TEST_SOURCE_TERMS, term_type=OntologyTermType.ANY)
        assert mappings_cache.size > 0

    def test_caching_ontology_from_acronym(self):
        print("Test caching an ontology with an acronym alias...")
        ontology_cache = text2term.cache_ontology(ontology_url=self.TEST_ONTOLOGY_FILEPATH,
                                                  ontology_acronym="LOCALPIZZA",
                                                  cache_folder=self.TEST_CACHE_FOLDER)
        print(f"Cache exists: {ontology_cache.cache_exists()}\n")
        assert ontology_cache.cache_exists() is True

    def test_cache_clearing(self):
        self.ensure_cache_exists(ontology_name="LOCALPIZZA", ontology_url=self.TEST_ONTOLOGY_FILEPATH)
        assert text2term.cache_exists(ontology_acronym="LOCALPIZZA", cache_folder=self.TEST_CACHE_FOLDER) is True
        text2term.clear_cache(ontology_acronym="LOCALPIZZA", cache_folder=self.TEST_CACHE_FOLDER)
        assert text2term.cache_exists(ontology_acronym="LOCALPIZZA", cache_folder=self.TEST_CACHE_FOLDER) is False

    def test_caching_ontology_set(self):
        nr_ontologies_in_registry = len(pd.read_csv(self.TEST_ONTOLOGY_REGISTRY))
        caches = text2term.cache_ontology_set(self.TEST_ONTOLOGY_REGISTRY, cache_folder=self.TEST_CACHE_FOLDER)
        assert len(caches) == nr_ontologies_in_registry

    def test_mapping_to_cached_ontology(self):
        text2term.clear_cache(cache_folder=self.TEST_CACHE_FOLDER)
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        print("Test mapping a list of terms to a local ontology loaded from cache...")
        mappings_cache = text2term.map_terms(self.TEST_SOURCE_TERMS, target_ontology=self.TEST_ONTOLOGY_ACRONYM,
                                             use_cache=True, term_type=OntologyTermType.ANY,
                                             cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{mappings_cache}\n")
        assert mappings_cache.size > 0

        print("Test mapping a list of terms to the same ontology loaded from a local file...")
        mappings_file = text2term.map_terms(self.TEST_SOURCE_TERMS, target_ontology=self.TEST_ONTOLOGY_FILEPATH,
                                            term_type=OntologyTermType.ANY)
        print(f"{mappings_file}\n")
        assert mappings_file.size > 0

        print("Test that mapping to cached ontology is the same as to ontology loaded from its file...")
        mappings_match = self.check_df_equals(self.drop_source_term_ids(mappings_cache),
                                              self.drop_source_term_ids(mappings_file))
        print(f"...{mappings_match}")
        assert mappings_match is True

    def test_mapping_to_cached_ontology_using_syntactic_mapper(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        print("Test mapping a list of terms to cached ontology using Jaro-Winkler syntactic similarity metric...")
        df = text2term.map_terms(self.TEST_SOURCE_TERMS, self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                                 mapper=text2term.Mapper.JARO_WINKLER, term_type=OntologyTermType.ANY,
                                 cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df}\n")
        assert df.size > 0

    def test_syntactic_mappers(self):
        mapper = SyntacticMapper(())
        for m in [Mapper.LEVENSHTEIN, Mapper.JARO, Mapper.JARO_WINKLER, Mapper.INDEL, Mapper.FUZZY, Mapper.JACCARD]:
            score = mapper.compare('heart', 'hearts', mapper=m)
            score_2 = mapper.compare('heart', 'lung', mapper=m)
            self.assertTrue(0.8 <= score <= 1.0)
            self.assertTrue(0.0 <= score_2 <= 0.1)

    def test_syntactic_mapper_inexistent(self):
        mapper = SyntacticMapper(())
        with self.assertRaises(ValueError):
            mapper.compare("heart", "hearts", mapper="mymapper")

    def test_mapping_using_ontology_acronym(self):
        print("Test mapping a list of terms to a local ontology file")
        df2 = text2term.map_terms(["cheese pizza", "fish pizza"], self.TEST_ONTOLOGY_FILEPATH,
                                  term_type=OntologyTermType.CLASS)
        print(f"{df2}\n")
        assert df2.size > 0

    def test_mapping_tagged_terms(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        print("Test mapping a dictionary of tagged terms to a cached local ontology...")
        df3 = text2term.map_terms(
            {"cheese pizza": "pizza", "ice cream": ["ignore", "dessert"], "fish pizza": ["seafood"],
             "isdjfnsdfwd": None}, target_ontology=self.TEST_ONTOLOGY_ACRONYM, excl_deprecated=True, use_cache=True,
            incl_unmapped=True,
            cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df3}\n")
        assert df3.size > 0
        assert df3[self.TAGS_COLUMN].str.contains("pizza").any()
        assert df3[self.TAGS_COLUMN].str.contains("seafood").any()

    def test_preprocessing_from_file(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        # Test processing tagged terms where the tags are provided in a file
        print("Test processing tagged terms where the tags are provided in a file...")
        tagged_terms = text2term.preprocess_tagged_terms(file_path=self.TEST_INPUT_FILEPATH)
        df4 = text2term.map_terms(tagged_terms, target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                                  incl_unmapped=True,
                                  cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df4}\n")
        assert df4.size > 0
        assert df4[self.TAGS_COLUMN].str.contains("pizza").any()
        assert df4[self.TAGS_COLUMN].str.contains("seafood").any()

    def test_preprocessing_tagged_terms(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        # Test processing tagged terms where the tags are provided in a file
        print("Test processing tagged terms where the tags are provided in a file...")
        tagged_terms = text2term.preprocess_tagged_terms(file_path=self.TEST_INPUT_FILEPATH,
                                                         template_path=self.TEST_TEMPLATES_FILEPATH)
        df = text2term.map_terms(tagged_terms, target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                                 incl_unmapped=True,
                                 cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df}\n")
        assert df.size > 0
        assert df[self.TAGS_COLUMN].str.contains("pizza").any()
        assert df[self.TAGS_COLUMN].str.contains("seafood").any()
        assert df["Source Term"].eq("pizza").any()
        assert not df["Source Term"].str.contains("pizza NOS").all()

    def test_preprocessing_terms(self):
        input_terms = ['Hypertension NOS', 'Diabetes mellitus due to underlying condition']
        result = text2term.preprocess_terms(terms=input_terms, template_path=self.TEST_TEMPLATES_FILEPATH)
        expected = {
            'Hypertension NOS': 'Hypertension',
            'Diabetes mellitus due to underlying condition': 'Diabetes mellitus'
        }
        self.assertEqual(result, expected)

    def test_preprocessing_terms_and_dedupe(self):
        input_terms = ['Hypertension NOS', 'Diabetes mellitus due to underlying condition',
                       'Hypertension due to unspecified condition', 'Hypertension']
        result = text2term.preprocess_terms(terms=input_terms, template_path=self.TEST_TEMPLATES_FILEPATH,
                                            rem_duplicates=True)
        expected = {
            'Hypertension': 'Hypertension',
            'Diabetes mellitus due to underlying condition': 'Diabetes mellitus'
        }
        self.assertEqual(result, expected)

    def test_preprocessing_blocklisted_terms(self):
        terms = ['Hypertension', 'Patient ID', 'Admission Date']
        blocklist_path = os.path.join(self.TEST_FOLDER, 'test_blocklist.txt')  # This file should contain 'Common Cold'
        result = text2term.preprocess_terms(terms=terms, template_path="", blocklist_path=blocklist_path)
        expected = {
            'Hypertension': 'Hypertension'
        }
        self.assertEqual(result, expected)

    def test_mapping_to_properties(self):
        print("Test mapping a list of properties to a local ontology file and restrict search to properties...")
        df5 = text2term.map_terms(source_terms=self.TEST_PROPERTY_TERMS, target_ontology=self.TEST_ONTOLOGY_FILEPATH,
                                  term_type=OntologyTermType.PROPERTY)
        print(f"{df5}\n")
        assert df5.size > 0

        print("Test mapping a list of properties to a cached local ontology and restrict search to properties...")
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        df6 = text2term.map_terms(source_terms=self.TEST_PROPERTY_TERMS, target_ontology=self.TEST_ONTOLOGY_ACRONYM,
                                  use_cache=True,
                                  term_type=OntologyTermType.PROPERTY, cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df6}\n")
        assert df6.size > 0

        # Test that mapping to properties in cached ontology is the same as to ontology loaded from its file
        properties_df_match = self.check_df_equals(self.drop_source_term_ids(df5), self.drop_source_term_ids(df6))
        print(f"Properties match: {properties_df_match}")
        assert properties_df_match is True

    @skip_unless_live_api
    def test_mapping_zooma_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the Zooma mapper
        print("Test mapping a list of terms to multiple ontologies using the Zooma mapper...")
        df_zooma = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="MONDO,NCIT",
                                       mapper=Mapper.ZOOMA, term_type=OntologyTermType.ANY)
        print(f"{df_zooma}\n")
        assert df_zooma.size > 0
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("MONDO:").any()
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    @skip_unless_live_api
    def test_mapping_zooma_empty_response(self):
        print("Test mapping a term that obviously has no ontology term to be mapped to...")
        df_zooma = text2term.map_terms(["ziggy"], target_ontology="NCIT",
                                       mapper=Mapper.ZOOMA, term_type=OntologyTermType.ANY)
        print(f"{df_zooma}\n")
        assert df_zooma.empty is True

    @skip_unless_live_api
    def test_mapping_zooma_bad_url(self):
        print("Test mapping terms to a wrong URL for Zooma mapper...")
        bp_mapper = ZoomaMapper()
        with self.assertRaises(Exception):
            bp_mapper._do_get_request(request_url="http://www.ebi.ac.uk/spot/zooma/v2/api/services/annotateBad")

    def test_mapping_bioportal_ontologies_no_apikey(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper without API Key
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="MONDO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY)
        assert df_bioportal.empty is True

    @skip_unless_bioportal_api
    def test_mapping_bioportal_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="MONDO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY,
                                           bioportal_apikey=BIOPORTAL_API_KEY)
        print(f"{df_bioportal}\n")
        assert df_bioportal.size > 0
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("MONDO:").any()
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    @skip_unless_bioportal_api
    def test_mapping_bioportal_empty_response(self):
        print("Test mapping a term that obviously has no ontology term to be mapped to...")
        df_bioportal = text2term.map_terms(["ziggy"], target_ontology="NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY,
                                           bioportal_apikey=BIOPORTAL_API_KEY)
        print(f"{df_bioportal}\n")
        assert df_bioportal.empty is True

    @skip_unless_bioportal_api
    def test_mapping_bioportal_bad_url(self):
        print("Test mapping terms to a wrong URL for BioPortal Annotator mapper...")
        bp_mapper = BioPortalAnnotatorMapper(bp_api_key=BIOPORTAL_API_KEY)
        with self.assertRaises(Exception):
            bp_mapper._do_get_request(request_url="http://data.bioontology.org/annotatorBad")

    def test_mapping_with_min_score_filter(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        min_score = 0.6
        search_terms = ["cheese pizza", "location"]

        print("Test mapping to cached local ontology using TFIDF similarity metric and min_score filter...")
        df_tfidf = text2term.map_terms(search_terms, target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                                       mapper=Mapper.TFIDF, term_type=OntologyTermType.ANY, min_score=min_score,
                                       cache_folder=self.TEST_CACHE_FOLDER)
        assert (df_tfidf[self.MAPPING_SCORE_COLUMN] >= min_score).all()

        print("Test mapping to cached local ontology using Levenshtein similarity metric and min_score filter...")
        df_leven = text2term.map_terms(search_terms, target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                                       mapper=Mapper.LEVENSHTEIN, term_type=OntologyTermType.ANY, min_score=min_score,
                                       cache_folder=self.TEST_CACHE_FOLDER)
        assert (df_leven[self.MAPPING_SCORE_COLUMN] >= min_score).all()

    def test_mapping_with_min_score_filter_empty_results(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        print("Test mapping to local ontology using TFIDF similarity metric and min_score filter that returns no mappings...")
        df_tfidf = text2term.map_terms(["carbon monoxide"], target_ontology=self.TEST_ONTOLOGY_ACRONYM,
                                       use_cache=True, mapper=Mapper.TFIDF, term_type=OntologyTermType.ANY,
                                       min_score=0.99, cache_folder=self.TEST_CACHE_FOLDER)
        assert df_tfidf.empty is True

    def test_include_unmapped_terms(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        df = text2term.map_terms(["cheese pizza", "margarita"], target_ontology=self.TEST_ONTOLOGY_ACRONYM,
                                 use_cache=True, mapper=Mapper.TFIDF, incl_unmapped=True, min_score=0.8,
                                 cache_folder=self.TEST_CACHE_FOLDER)
        assert df[self.TAGS_COLUMN].str.contains("unmapped").any()

    def test_exclude_metadata_from_output_file(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        test_output_file = "test_output_without_metadata.csv"
        text2term.map_terms(["cheese pizza"], target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                            mapper=Mapper.TFIDF, excl_metadata=True, save_mappings=True, output_file=test_output_file,
                            cache_folder=self.TEST_CACHE_FOLDER)
        with open(test_output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            assert not first_line.startswith('#'), "CSV output file should not start with metadata header"

    def test_include_metadata_in_output_file(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        test_output_file = "test_output_with_metadata.csv"
        text2term.map_terms(["cheese pizza"], target_ontology=self.TEST_ONTOLOGY_ACRONYM, use_cache=True,
                            mapper=Mapper.TFIDF, excl_metadata=False, save_mappings=True, output_file=test_output_file,
                            cache_folder=self.TEST_CACHE_FOLDER)
        with open(test_output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            assert first_line.startswith('#'), "CSV output file should contain metadata header"

    def test_include_unmapped_terms_when_mappings_df_is_empty(self):
        self.ensure_cache_exists(self.TEST_ONTOLOGY_ACRONYM, self.TEST_ONTOLOGY_FILEPATH)
        df = text2term.map_terms(["mojito", "margarita"], target_ontology=self.TEST_ONTOLOGY_ACRONYM,
                                 use_cache=True, mapper=Mapper.TFIDF, incl_unmapped=True, min_score=0.8,
                                 cache_folder=self.TEST_CACHE_FOLDER)
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
        if not text2term.cache_exists(ontology_name, cache_folder=self.TEST_CACHE_FOLDER):
            text2term.cache_ontology(ontology_url=ontology_url, ontology_acronym=ontology_name,
                                     cache_folder=self.TEST_CACHE_FOLDER)


class OntologyTermCollectorTestSuite(unittest.TestCase):

    def setUp(self):
        ontology_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ontology.owl")
        self.collector = OntologyTermCollector(ontology_file, use_reasoning=True)
        self.test_base_iri = "https://text2term.ontology/test/"

    def tearDown(self):
        self.collector.close()

    def test_collect_terms(self):
        terms = self.collector.get_ontology_terms()
        assert len(terms) == 14

    def test_collect_classes_only(self):
        terms = self.collector.get_ontology_terms(term_type=OntologyTermType.CLASS)
        assert len(terms) == 7

    def test_collect_properties_only(self):
        terms = self.collector.get_ontology_terms(term_type=OntologyTermType.PROPERTY)
        assert len(terms) == 7

    def test_collect_terms_limited_by_iri(self):
        terms = self.collector.get_ontology_terms(base_iris=[self.test_base_iri], term_type=OntologyTermType.ANY)
        assert len(terms) == 9

    def test_collect_properties_limited_by_iri(self):
        terms = self.collector.get_ontology_terms(base_iris=[self.test_base_iri],
                                                  term_type=OntologyTermType.PROPERTY)
        assert len(terms) == 2

    def test_collect_terms_excluding_deprecated(self):
        terms = self.collector.get_ontology_terms(exclude_deprecated=True, base_iris=(self.test_base_iri,))
        self.assertTrue(len(terms) == 8)

    def test_collect_terms_including_broad_synonyms(self):
        terms = self.collector.get_ontology_terms(exclude_deprecated=True, base_iris=(self.test_base_iri,),
                                                  include_broad_synonyms=True)
        self.assertTrue(any("Seafood Pizza" in term.synonyms for term in terms.values()))

    def test_collect_terms_including_narrow_synonyms(self):
        terms = self.collector.get_ontology_terms(exclude_deprecated=True, base_iris=(self.test_base_iri,),
                                                  include_related_synonyms=True)
        self.assertTrue(any("Fish Pizza" in term.synonyms for term in terms.values()))

    def test_filter_terms(self):
        terms = self.collector.get_ontology_terms()
        filtered = text2term.filter_terms(terms, iris=self.test_base_iri, excl_deprecated=False,
                                          term_type=OntologyTermType.CLASS)
        self.assertTrue(len(filtered) == 7)
        for term in filtered.values():
            self.assertEqual(term.term_type, OntologyTermType.CLASS)

    def test_filter_terms_invalid_type(self):
        terms = self.collector.get_ontology_terms()
        with self.assertRaises(ValueError):
            text2term.filter_terms(terms, term_type="invalid")

    def test_get_ontology_signature(self):
        sig = self.collector._get_ontology_signature(self.collector.ontology)
        self.assertTrue(all(isinstance(e, EntityClass) for e in sig))

    def test_classify_ontology(self):
        self.collector._classify_ontology(self.collector.ontology)
        test_term = IRIS["https://text2term.ontology/test/VanillaIceCream"]
        self.assertTrue(test_term.deprecated)


class OntoUtilsTestSuite(unittest.TestCase):

    def test_normalize_list(self):
        self.assertEqual(onto_utils.normalize_list([" Apple ", "Banana"]), ["apple", "banana"])

    def test_normalize(self):
        self.assertEqual(onto_utils.normalize(" TeSt "), "test")

    def test_remove_quotes(self):
        self.assertEqual(onto_utils.remove_quotes('"quoted"'), "quoted")
        self.assertEqual(onto_utils.remove_quotes("'quoted'"), "quoted")

    def test_remove_whitespace(self):
        self.assertEqual(onto_utils.remove_whitespace(" spaced string "), "spacedstring")

    def test_curie_from_iri(self):
        self.assertEqual(onto_utils.curie_from_iri("http://purl.obolibrary.org/obo/GO_0008150"), "GO:0008150")

    def test_label_from_iri(self):
        self.assertEqual(onto_utils.label_from_iri("http://example.org/ontology/Term_Label"), "Term_Label")

    def test_iri_from_tag(self):
        self.assertEqual(onto_utils.iri_from_tag("GO_0008150"), "http://purl.obolibrary.org/obo/GO_0008150")
        self.assertEqual(onto_utils.iri_from_tag("GO:0008150"), "http://purl.obolibrary.org/obo/GO_0008150")

    def test_get_logger(self):
        logger = onto_utils.get_logger("test_logger")
        self.assertTrue(hasattr(logger, "info"))

    def test_parse_list_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("apple\nbanana\n")
            f_path = f.name
        try:
            self.assertEqual(onto_utils.parse_list_file(f_path), ["apple", "banana"])
        finally:
            os.remove(f_path)

    def test_parse_csv_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("disease,disease code,patient count\n"
                    "asthma,abb1,30\n"
                    "bronchitis,baa1,21")
            f_path = f.name
        try:
            terms, term_ids = onto_utils.parse_csv_file(file_path=f_path,
                                                        term_column_name="disease",
                                                        term_id_column_name="disease code")
            self.assertEqual(terms.tolist(), ["asthma", "bronchitis"])
            self.assertEqual(term_ids.tolist(), ["abb1", "baa1"])
        finally:
            os.remove(f_path)

    def test_parse_tsv_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("disease\tdisease code\tpatient count\n"
                    "asthma\tabb1\t30\n"
                    "bronchitis\tbaa1\t21")
            f_path = f.name
        try:
            terms, term_ids = onto_utils.parse_tsv_file(file_path=f_path,
                                                        term_column_name= "disease",
                                                        term_id_column_name= "disease code")
            self.assertEqual(terms.tolist(), ["asthma", "bronchitis"])
            self.assertEqual(term_ids.tolist(), ["abb1", "baa1"])
        finally:
            os.remove(f_path)

    def test_get_ontology_from_labels(self):
        ontology_iri = "https://test.ontology.org/"
        terms = ["asthma", "bronchitis", "hypertension", "diabetes mellitus"]
        result = onto_utils.get_ontology_from_labels(terms, ontology_iri=ontology_iri)
        self.assertEqual(result.base_iri, ontology_iri)
        self.assertEqual(len(list(result.classes())), len(terms))

    def test_generate_uuid(self):
        some_uuid = onto_utils.generate_uuid(length=23)
        self.assertTrue(len(some_uuid) == 23)

    def test_generate_iri(self):
        iri = onto_utils.generate_iri()
        self.assertTrue(iri.startswith(onto_utils.BASE_IRI + "R"))

    def test_generate_iris(self):
        iris = onto_utils.generate_iris(3)
        self.assertEqual(len(iris), 3)
        self.assertTrue(all(i.startswith(onto_utils.BASE_IRI + "R") for i in iris))


if __name__ == '__main__':
    unittest.main()
