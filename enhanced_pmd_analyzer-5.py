# coding=utf-8
"""Enhanced Municipal Development Plan Analyzer - Production-Grade Implementation.

This module implements state-of-the-art techniques for comprehensive municipal plan analysis:
- Semantic cubes with knowledge graphs and ontological reasoning
- Multi-dimensional baseline analysis with automated extraction
- Advanced NLP for multimodal text mining and causal discovery
- Real-time monitoring with statistical process control
- Bayesian optimization for resource allocation
- Uncertainty quantification with Monte Carlo methods
"""

import json
import logging
import math
import statistics
import re
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any, Dict, List, Tuple, Optional, Sequence, Mapping, Union,
    Set, Callable, TypeVar, Generic, NamedTuple, Iterator, Protocol
)
import random
import itertools
import heapq
import bisect
import threading
from contextlib import contextmanager
import time
import pickle
import base64
import zlib
import warnings
warnings.filterwarnings('ignore')

# Production-grade scientific computing stack
import numpy as np
import pandas as pd
from scipy import stats, optimize, sparse, linalg
from scipy.spatial.distance import cosine, euclidean, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.signal import find_peaks, savgol_filter
import networkx as nx
from networkx.algorithms import community, centrality, shortest_paths

# Advanced machine learning and NLP
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.manifold import TSNE, UMAP
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import xgboost as xgb
import lightgbm as lgb

# Deep learning and transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTokenClassification,
    pipeline, BertModel, RobertaModel, DistilBertModel
)
from sentence_transformers import SentenceTransformer, util as st_util
import spacy
from spacy import displacy
import en_core_web_sm, es_core_news_sm

# Specialized libraries for advanced analytics
import networkx as nx
import igraph as ig
import graph_tool.all as gt
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
import owlready2
from pyld import jsonld
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from textstat import flesch_reading_ease, flesch_kincaid_grade
import yake
from keybert import KeyBERT
import bertopic
from gensim import models, corpora
from gensim.models import LdaModel, CoherenceModel, Word2Vec, FastText
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Time series and optimization
from statsmodels.tsa import stattools, arima, seasonal
from statsmodels.stats import diagnostic
import pmdarima as pm
from scipy.optimize import minimize, differential_evolution, basinhopping
import optuna
from hyperopt import hp, fmin, tpe, Trials
from bayesian_optimization import BayesianOptimization
import cvxpy as cp

# Causal inference and econometrics
from dowhy import CausalModel
from econml.dml import LinearDML, CausalForestDML
from causal_inference import CausalInference
import causalnex
from causalnex.structure import StructureLearner
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

# Geospatial and temporal analysis
import geopandas as gpd
import folium
from shapely.geometry import Point, Polygon, LineString
from geopy.distance import geodesic
import rasterio
from rasterio.plot import show
import contextily as ctx
from sklearn.neighbors import BallTree, KDTree

# Database and data engineering
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from great_expectations import DataContext
import pandera as pa_schema
from pandera import Column as PanderaColumn, DataFrameSchema

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Production constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. SEMANTIC CUBES WITH KNOWLEDGE GRAPHS AND ONTOLOGICAL CODING
# ---------------------------------------------------------------------------

@dataclass
class ValueChainLink:
    """Represents a link in the municipal development value chain."""
    name: str
    instruments: List[str]
    mediators: List[str]
    outputs: List[str] 
    outcomes: List[str]
    bottlenecks: List[str]
    lead_time_days: float
    conversion_rates: Dict[str, float]
    capacity_constraints: Dict[str, float]

class OntologicalKnowledgeGraph:
    """Production-grade ontological knowledge graph for municipal development domains."""
    
    def __init__(self, ontology_config: Optional[Dict[str, Any]] = None):
        self.config = ontology_config or self._load_production_ontology()
        
        # Initialize knowledge graph with multiple backends
        self.rdf_graph = Graph()
        self.networkx_graph = nx.MultiDiGraph()
        self.igraph_graph = ig.Graph(directed=True)
        
        # Namespaces for semantic web
        self.MDO = Namespace("http://municipaldev.org/ontology#")  # Municipal Development Ontology
        self.GEO = Namespace("http://www.opengis.net/ont/geosparql#")
        self.TIME = Namespace("http://www.w3.org/2006/time#")
        self.PROV = Namespace("http://www.w3.org/ns/prov#")
        
        # Initialize transformer models for semantic understanding
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        self.domain_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
        
        # Initialize NLP pipeline with multiple languages
        try:
            self.nlp_es = spacy.load("es_core_news_lg")
        except IOError:
            self.nlp_es = spacy.load("es_core_news_sm")
        
        try:
            self.nlp_en = spacy.load("en_core_web_lg") 
        except IOError:
            self.nlp_en = spacy.load("en_core_web_sm")
            
        # Initialize entity linking and relation extraction
        self.entity_linker = pipeline("ner", 
                                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                    aggregation_strategy="simple")
        
        self.relation_extractor = pipeline("text-classification",
                                         model="microsoft/DialoGPT-medium")
        
        # Initialize topic modeling
        self.topic_model = BERTopic(
            embedding_model=self.semantic_model,
            min_topic_size=10,
            nr_topics="auto",
            calculate_probabilities=True,
            verbose=False
        )
        
        # Build ontology
        self._build_ontological_structure()
        
        logger.info("Ontological Knowledge Graph initialized with production-grade components")
    
    def _load_production_ontology(self) -> Dict[str, Any]:
        """Load production ontology configuration based on international standards."""
        return {
            "value_chain_links": {
                "diagnostic_planning": ValueChainLink(
                    name="diagnostic_planning",
                    instruments=["territorial_diagnosis", "stakeholder_mapping", "needs_assessment", 
                               "capacity_analysis", "risk_assessment", "opportunity_identification"],
                    mediators=["technical_capacity", "participatory_processes", "information_systems",
                             "inter_institutional_coordination", "citizen_engagement"],
                    outputs=["diagnostic_report", "territorial_profile", "problem_tree", 
                           "stakeholder_matrix", "capacity_map", "risk_matrix"],
                    outcomes=["shared_territorial_vision", "prioritized_problems", 
                            "identified_opportunities", "stakeholder_alignment"],
                    bottlenecks=["data_availability", "technical_capacity_gaps", "stakeholder_resistance",
                               "time_constraints", "resource_limitations"],
                    lead_time_days=90,
                    conversion_rates={"diagnosis_to_strategy": 0.75, "problems_to_solutions": 0.60},
                    capacity_constraints={"technical_staff": 0.8, "financial_resources": 0.6, "time": 0.7}
                ),
                "strategic_planning": ValueChainLink(
                    name="strategic_planning", 
                    instruments=["strategic_framework", "theory_of_change", "results_matrix",
                               "intervention_logic", "risk_management_plan"],
                    mediators=["planning_methodology", "stakeholder_participation", "technical_assistance",
                             "political_leadership", "institutional_capacity"],
                    outputs=["development_plan", "sector_strategies", "program_portfolio",
                           "investment_plan", "monitoring_framework"],
                    outcomes=["strategic_alignment", "resource_optimization", "implementation_readiness",
                            "stakeholder_buy_in"],
                    bottlenecks=["political_changes", "resource_constraints", "capacity_limitations",
                               "coordination_failures", "external_dependencies"],
                    lead_time_days=120,
                    conversion_rates={"strategy_to_programs": 0.80, "programs_to_projects": 0.70},
                    capacity_constraints={"planning_expertise": 0.7, "stakeholder_time": 0.6, "resources": 0.8}
                ),
                "program_design": ValueChainLink(
                    name="program_design",
                    instruments=["program_logic", "intervention_design", "beneficiary_targeting",
                               "delivery_mechanisms", "quality_standards"],
                    mediators=["technical_expertise", "evidence_base", "stakeholder_input", 
                             "resource_availability", "implementation_capacity"],
                    outputs=["program_documents", "operational_manuals", "quality_frameworks",
                           "targeting_criteria", "delivery_protocols"],
                    outcomes=["program_feasibility", "implementation_readiness", "quality_assurance",
                            "stakeholder_alignment"],
                    bottlenecks=["design_complexity", "resource_constraints", "capacity_gaps",
                               "coordination_challenges", "regulatory_barriers"],
                    lead_time_days=75,
                    conversion_rates={"design_to_implementation": 0.85, "programs_to_outcomes": 0.65},
                    capacity_constraints={"design_expertise": 0.75, "resources": 0.70, "time": 0.80}
                ),
                "implementation": ValueChainLink(
                    name="implementation",
                    instruments=["project_management", "procurement_systems", "service_delivery",
                               "infrastructure_development", "capacity_building"],
                    mediators=["administrative_systems", "human_resources", "financial_management",
                             "supply_chains", "quality_control"],
                    outputs=["services_delivered", "infrastructure_built", "capacities_developed",
                           "beneficiaries_served", "results_achieved"],
                    outcomes=["improved_living_conditions", "enhanced_capabilities", "economic_development",
                            "social_cohesion", "environmental_sustainability"],
                    bottlenecks=["budget_execution", "procurement_delays", "capacity_constraints",
                               "coordination_failures", "external_factors"],
                    lead_time_days=365,
                    conversion_rates={"inputs_to_outputs": 0.75, "outputs_to_outcomes": 0.60},
                    capacity_constraints={"implementation_capacity": 0.65, "financial_execution": 0.70, "coordination": 0.60}
                ),
                "monitoring_evaluation": ValueChainLink(
                    name="monitoring_evaluation",
                    instruments=["indicator_systems", "data_collection", "analysis_methods",
                               "evaluation_frameworks", "feedback_mechanisms"],
                    mediators=["M&E_systems", "data_quality", "analytical_capacity",
                             "stakeholder_engagement", "learning_culture"],
                    outputs=["performance_reports", "evaluation_studies", "lessons_learned",
                           "recommendations", "corrective_actions"],
                    outcomes=["improved_performance", "enhanced_accountability", "evidence_based_decisions",
                            "continuous_improvement", "stakeholder_confidence"],
                    bottlenecks=["data_availability", "analytical_capacity", "utilization_gaps",
                               "resource_constraints", "institutional_resistance"],
                    lead_time_days=60,
                    conversion_rates={"monitoring_to_learning": 0.70, "learning_to_improvement": 0.55},
                    capacity_constraints={"M&E_expertise": 0.60, "data_systems": 0.65, "resources": 0.70}
                )
            },
            "cross_cutting_themes": {
                "governance": ["transparency", "accountability", "participation", "rule_of_law"],
                "equity": ["gender_equality", "social_inclusion", "poverty_reduction", "human_rights"],
                "sustainability": ["environmental_protection", "climate_adaptation", "resource_efficiency", "circular_economy"],
                "innovation": ["digital_transformation", "technological_adoption", "process_innovation", "social_innovation"]
            },
            "policy_domains": {
                "economic_development": ["competitiveness", "entrepreneurship", "employment", "productivity"],
                "social_development": ["education", "health", "housing", "social_protection"],
                "territorial_development": ["land_use", "infrastructure", "connectivity", "spatial_planning"],
                "environmental_management": ["natural_resources", "pollution_control", "biodiversity", "climate_change"],
                "institutional_development": ["public_administration", "governance", "transparency", "capacity_building"]
            }
        }
    
    def _build_ontological_structure(self):
        """Build comprehensive ontological structure using semantic web standards."""
        
        # Build RDF graph
        for link_name, link_obj in self.config["value_chain_links"].items():
            link_uri = self.MDO[link_name]
            self.rdf_graph.add((link_uri, RDF.type, self.MDO.ValueChainLink))
            self.rdf_graph.add((link_uri, RDFS.label, Literal(link_name)))
            
            # Add instruments, mediators, outputs, outcomes
            for instrument in link_obj.instruments:
                inst_uri = self.MDO[f"{link_name}_instrument_{instrument}"]
                self.rdf_graph.add((inst_uri, RDF.type, self.MDO.Instrument))
                self.rdf_graph.add((inst_uri, self.MDO.partOf, link_uri))
                self.rdf_graph.add((inst_uri, RDFS.label, Literal(instrument)))
            
            for mediator in link_obj.mediators:
                med_uri = self.MDO[f"{link_name}_mediator_{mediator}"]
                self.rdf_graph.add((med_uri, RDF.type, self.MDO.Mediator))
                self.rdf_graph.add((med_uri, self.MDO.partOf, link_uri))
                self.rdf_graph.add((med_uri, RDFS.label, Literal(mediator)))
            
            for output in link_obj.outputs:
                out_uri = self.MDO[f"{link_name}_output_{output}"]
                self.rdf_graph.add((out_uri, RDF.type, self.MDO.Output))
                self.rdf_graph.add((out_uri, self.MDO.producedBy, link_uri))
                self.rdf_graph.add((out_uri, RDFS.label, Literal(output)))
            
            for outcome in link_obj.outcomes:
                outc_uri = self.MDO[f"{link_name}_outcome_{outcome}"]
                self.rdf_graph.add((outc_uri, RDF.type, self.MDO.Outcome))
                self.rdf_graph.add((outc_uri, self.MDO.resultFrom, link_uri))
                self.rdf_graph.add((outc_uri, RDFS.label, Literal(outcome)))
        
        # Build NetworkX graph for analysis
        for link_name, link_obj in self.config["value_chain_links"].items():
            self.networkx_graph.add_node(link_name, 
                                       node_type="value_chain_link",
                                       lead_time=link_obj.lead_time_days,
                                       **link_obj.capacity_constraints)
            
            # Add causal relationships
            for conversion, rate in link_obj.conversion_rates.items():
                source, target = conversion.split("_to_")
                if target in [l for l in self.config["value_chain_links"].keys()]:
                    self.networkx_graph.add_edge(link_name, target, 
                                               edge_type="causal",
                                               conversion_rate=rate)
        
        logger.info(f"Built ontological structure with {len(self.rdf_graph)} RDF triples and {len(self.networkx_graph)} NetworkX nodes")
    
    def extract_semantic_cube(self, document_segments: List[str]) -> Dict[str, Any]:
        """Extract multidimensional semantic cube from document segments."""
        
        # Initialize semantic cube structure
        semantic_cube = {
            "dimensions": {
                "value_chain_links": defaultdict(list),
                "policy_domains": defaultdict(list),
                "cross_cutting_themes": defaultdict(list),
                "temporal": defaultdict(list),
                "spatial": defaultdict(list)
            },
            "measures": {
                "semantic_density": [],
                "coherence_scores": [],
                "coverage_indicators": [],
                "complexity_metrics": []
            },
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "total_segments": len(document_segments),
                "processing_parameters": {}
            }
        }
        
        # Process segments in batches for efficiency
        batch_size = 32
        segment_embeddings = []
        
        for i in range(0, len(document_segments), batch_size):
            batch = document_segments[i:i + batch_size]
            batch_embeddings = self.semantic_model.encode(batch, convert_to_tensor=True)
            segment_embeddings.extend(batch_embeddings.cpu().numpy())
        
        segment_embeddings = np.array(segment_embeddings)
        
        # Extract entities and relations from segments
        for idx, segment in enumerate(document_segments):
            
            # NLP processing
            doc_es = self.nlp_es(segment)
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                       for ent in doc_es.ents]
            
            # Map entities to ontological concepts
            ontological_mappings = self._map_entities_to_ontology(entities, segment)
            
            # Classify by value chain links
            link_scores = self._classify_value_chain_link(segment, segment_embeddings[idx])
            
            # Classify by policy domains
            domain_scores = self._classify_policy_domain(segment, entities)
            
            # Extract temporal and spatial references
            temporal_refs = self._extract_temporal_references(segment, doc_es)
            spatial_refs = self._extract_spatial_references(segment, doc_es)
            
            # Calculate semantic measures
            density = self._calculate_semantic_density(segment, entities)
            coherence = self._calculate_local_coherence(segment, idx, segment_embeddings)
            
            # Store in semantic cube
            segment_data = {
                "segment_id": idx,
                "text": segment,
                "embedding": segment_embeddings[idx].tolist(),
                "entities": entities,
                "ontological_mappings": ontological_mappings,
                "semantic_density": density,
                "local_coherence": coherence,
                "temporal_refs": temporal_refs,
                "spatial_refs": spatial_refs
            }
            
            # Assign to dimensions based on classification scores
            for link, score in link_scores.items():
                if score > 0.3:  # Threshold for inclusion
                    semantic_cube["dimensions"]["value_chain_links"][link].append(segment_data)
            
            for domain, score in domain_scores.items():
                if score > 0.3:
                    semantic_cube["dimensions"]["policy_domains"][domain].append(segment_data)
            
            # Temporal dimension
            if temporal_refs:
                for temp_ref in temporal_refs:
                    semantic_cube["dimensions"]["temporal"][temp_ref["type"]].append(segment_data)
            
            # Spatial dimension  
            if spatial_refs:
                for spat_ref in spatial_refs:
                    semantic_cube["dimensions"]["spatial"][spat_ref["type"]].append(segment_data)
            
            semantic_cube["measures"]["semantic_density"].append(density)
            semantic_cube["measures"]["coherence_scores"].append(coherence)
        
        # Calculate aggregate measures
        semantic_cube["measures"]["overall_coherence"] = np.mean(semantic_cube["measures"]["coherence_scores"])
        semantic_cube["measures"]["semantic_complexity"] = self._calculate_semantic_complexity(semantic_cube)
        semantic_cube["measures"]["coverage_completeness"] = self._calculate_coverage_completeness(semantic_cube)
        
        logger.info(f"Extracted semantic cube with {sum(len(dim) for dim in semantic_cube['dimensions'].values())} dimensional assignments")
        
        return semantic_cube
    
    def _map_entities_to_ontology(self, entities: List[Tuple], segment: str) -> Dict[str, Any]:
        """Map extracted entities to ontological concepts using semantic similarity."""
        mappings = {
            "direct_mappings": [],
            "semantic_mappings": [],
            "confidence_scores": {}
        }
        
        # Create embeddings for entities
        entity_texts = [entity[0] for entity in entities]
        if not entity_texts:
            return mappings
            
        entity_embeddings = self.domain_model.encode(entity_texts)
        
        # Get ontological concept embeddings
        ontology_concepts = []
        ontology_embeddings = []
        
        for link_name, link_obj in self.config["value_chain_links"].items():
            all_concepts = (link_obj.instruments + link_obj.mediators + 
                          link_obj.outputs + link_obj.outcomes)
            for concept in all_concepts:
                ontology_concepts.append(f"{link_name}:{concept}")
                
        if ontology_concepts:
            ontology_embeddings = self.domain_model.encode(ontology_concepts)
            
            # Calculate semantic similarity
            similarity_matrix = st_util.cos_sim(entity_embeddings, ontology_embeddings)
            
            # Find best mappings
            for i, entity in enumerate(entities):
                best_match_idx = similarity_matrix[i].argmax()
                best_score = float(similarity_matrix[i][best_match_idx])
                
                if best_score > 0.5:  # Threshold for semantic mapping
                    mappings["semantic_mappings"].append({
                        "entity": entity[0],
                        "ontological_concept": ontology_concepts[best_match_idx],
                        "similarity_score": best_score
                    })
                    mappings["confidence_scores"][entity[0]] = best_score
        
        return mappings
    
    def _classify_value_chain_link(self, segment: str, segment_embedding: np.ndarray) -> Dict[str, float]:
        """Classify segment by value chain link using semantic similarity."""
        link_scores = {}
        
        # Create embeddings for value chain link descriptions
        for link_name, link_obj in self.config["value_chain_links"].items():
            link_description = " ".join(link_obj.instruments + link_obj.mediators + 
                                      link_obj.outputs + link_obj.outcomes)
            link_embedding = self.semantic_model.encode([link_description])
            
            # Calculate semantic similarity
            similarity = float(st_util.cos_sim(segment_embedding.reshape(1, -1), link_embedding)[0][0])
            link_scores[link_name] = similarity
        
        return link_scores
    
    def _classify_policy_domain(self, segment: str, entities: List[Tuple]) -> Dict[str, float]:
        """Classify segment by policy domain using entity-based classification."""
        domain_scores = defaultdict(float)
        
        segment_lower = segment.lower()
        
        for domain, keywords in self.config["policy_domains"].items():
            score = 0.0
            for keyword in keywords:
                if keyword.lower() in segment_lower:
                    score += 1.0
                    
            # Normalize by number of keywords
            if keywords:
                domain_scores[domain] = score / len(keywords)
        
        return dict(domain_scores)
    
    def _extract_temporal_references(self, segment: str, doc) -> List[Dict[str, Any]]:
        """Extract temporal references from segment."""
        temporal_refs = []
        
        # Pattern-based temporal extraction
        temporal_patterns = {
            "year": r'\b(19|20)\d{2}\b',
            "period": r'\b(trimestre|semestre|año|período|etapa|fase)\b',
            "planning_horizon": r'\b(corto|mediano|largo)\s+plazo\b',
            "relative_time": r'\b(actual|presente|futuro|próximo|anterior)\b'
        }
        
        for temp_type, pattern in temporal_patterns.items():
            matches = re.finditer(pattern, segment.lower())
            for match in matches:
                temporal_refs.append({
                    "type": temp_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return temporal_refs
    
    def _extract_spatial_references(self, segment: str, doc) -> List[Dict[str, Any]]:
        """Extract spatial references from segment."""
        spatial_refs = []
        
        # Use spaCy's named entity recognition for locations
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                spatial_refs.append({
                    "type": "geographic_location",
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Pattern-based spatial extraction
        spatial_patterns = {
            "administrative_level": r'\b(municipal|departamental|regional|nacional|local)\b',
            "spatial_scale": r'\b(urbano|rural|territorial|zonal|sectorial)\b'
        }
        
        for spat_type, pattern in spatial_patterns.items():
            matches = re.finditer(pattern, segment.lower())
            for match in matches:
                spatial_refs.append({
                    "type": spat_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return spatial_refs
    
    def _calculate_semantic_density(self, segment: str, entities: List[Tuple]) -> float:
        """Calculate semantic density of segment."""
        words = segment.split()
        if not words:
            return 0.0
        
        entity_words = sum(len(entity[0].split()) for entity in entities)
        return entity_words / len(words)
    
    def _calculate_local_coherence(self, segment: str, segment_idx: int, 
                                 all_embeddings: np.ndarray) -> float:
        """Calculate local semantic coherence."""
        if len(all_embeddings) <= 1:
            return 1.0
        
        # Calculate similarity with neighboring segments
        similarities = []
        window_size = 3
        
        for i in range(max(0, segment_idx - window_size), 
                      min(len(all_embeddings), segment_idx + window_size + 1)):
            if i != segment_idx:
                sim = cosine(all_embeddings[segment_idx], all_embeddings[i])
                similarities.append(1 - sim)  # Convert distance to similarity
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_semantic_complexity(self, semantic_cube: Dict[str, Any]) -> float:
        """Calculate overall semantic complexity of the cube."""
        
        # Count unique concepts across dimensions
        unique_concepts = set()
        
        for dimension_name, dimension_data in semantic_cube["dimensions"].items():
            for category, segments in dimension_data.items():
                unique_concepts.add(f"{dimension_name}:{category}")
        
        # Calculate complexity based on concept diversity and interactions
        concept_count = len(unique_concepts)
        
        # Calculate interaction complexity
        dimension_sizes = [len(dim_data) for dim_data in semantic_cube["dimensions"].values()]
        interaction_complexity = np.prod(dimension_sizes) if dimension_sizes else 1
        
        # Normalize complexity score
        max_expected_concepts = 50  # Based on ontology size
        normalized_complexity = min(1.0, concept_count / max_expected_concepts)
        
        return float(normalized_complexity)
    
    def _calculate_coverage_completeness(self, semantic_cube: Dict[str, Any]) -> float:
        """Calculate coverage completeness across all dimensions."""
        
        total_expected_categories = (
            len(self.config["value_chain_links"]) +
            len(self.config["policy_domains"]) +
            len(self.config["cross_cutting_themes"])
        )
        
        covered_categories = 0
        for dimension_data in semantic_cube["dimensions"].values():
            covered_categories += len([cat for cat, segments in dimension_data.items() if segments])
        
        return min(1.0, covered_categories / total_expected_categories) if total_expected_categories > 0 else 0.0

# ---------------------------------------------------------------------------
# 2. PERFORMANCE INDICATORS WITH OPERATIONAL LOSS FUNCTIONS
# ---------------------------------------------------------------------------

class ValueChainPerformanceAnalyzer:
    """Advanced performance analysis for value chain links with operational loss functions."""
    
    def __init__(self, knowledge_graph: OntologicalKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.performance_history = defaultdict(list)
        self.bottleneck_detector = IsolationForest(contamination=0.1, random_state=RANDOM_SEED)
        self.lead_time_predictor = None
        self.conversion_rate_models = {}
        self._initialize_performance_models()
        
        logger.info("ValueChainPerformanceAnalyzer initialized with ML models")
    
    def _initialize_performance_models(self):
        """Initialize machine learning models for performance prediction."""
        
        # Lead time prediction model (Gaussian Process for uncertainty quantification)
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        self.lead_time_predictor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=RANDOM_SEED
        )
        
        # Conversion rate models for each link
        for link_name in self.knowledge_graph.config["value_chain_links"].keys():
            # Use XGBoost for robust prediction with feature interactions
            self.conversion_rate_models[link_name] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                objective='reg:squarederror'
            )
    
    def analyze_performance_indicators(self, semantic_cube: Dict[str, Any], 
                                     historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze comprehensive performance indicators with operational loss functions."""
        
        performance_analysis = {
            "value_chain_metrics": {},
            "bottleneck_analysis": {},
            "conversion_rates": {},
            "lead_time_analysis": {},
            "operational_loss_functions": {},
            "predictive_indicators": {},
            "recommendation_matrix": {}
        }
        
        # Extract performance features from semantic cube
        performance_features = self._extract_performance_features(semantic_cube)
        
        # Analyze each value chain link
        for link_name, link_config in self.knowledge_graph.config["value_chain_links"].items():
            link_segments = semantic_cube["dimensions"]["value_chain_links"].get(link_name, [])
            
            # Calculate throughput metrics
            throughput_metrics = self._calculate_throughput_metrics(link_segments, link_config)
            
            # Detect bottlenecks
            bottlenecks = self._detect_bottlenecks(link_segments, link_config, performance_features)
            
            # Calculate conversion rates
            conversion_rates = self._calculate_conversion_rates(link_segments, link_config, historical_data)
            
            # Predict lead times
            lead_time_analysis = self._analyze_lead_times(link_segments, link_config, performance_features)
            
            # Define operational loss functions
            loss_functions = self._define_operational_loss_functions(link_config, throughput_metrics)
            
            # Store results
            performance_analysis["value_chain_metrics"][link_name] = throughput_metrics
            performance_analysis["bottleneck_analysis"][link_name] = bottlenecks
            performance_analysis["conversion_rates"][link_name] = conversion_rates
            performance_analysis["lead_time_analysis"][link_name] = lead_time_analysis
            performance_analysis["operational_loss_functions"][link_name] = loss_functions
        
        # Generate predictive indicators
        performance_analysis["predictive_indicators"] = self._generate_predictive_indicators(
            performance_analysis, performance_features
        )
        
        # Generate optimization recommendations
        performance_analysis["recommendation_matrix"] = self._generate_optimization_recommendations(
            performance_analysis
        )
        
        logger.info(f"Performance analysis completed for {len(performance_analysis['value_chain_metrics'])} value chain links")
        
        return performance_analysis
    
    def _extract_performance_features(self, semantic_cube: Dict[str, Any]) -> pd.DataFrame:
        """Extract quantitative features for performance analysis."""
        
        features = []
        
        for dimension_name, dimension_data in semantic_cube["dimensions"].items():
            for category, segments in dimension_data.items():
                if segments:
                    # Text-based features
                    total_length = sum(len(seg["text"]) for seg in segments)
                    avg_length = total_length / len(segments)
                    entity_density = np.mean([seg["semantic_density"] for seg in segments])
                    coherence_score = np.mean([seg["local_coherence"] for seg in segments])
                    
                    # Semantic features
                    embeddings = np.array([seg["embedding"] for seg in segments])
                    semantic_centrality = np.mean(np.linalg.norm(embeddings, axis=1))
                    semantic_variance = np.var(embeddings.flatten())
                    
                    features.append({
                        "dimension": dimension_name,
                        "category": category,
                        "segment_count": len(segments),
                        "total_length": total_length,
                        "avg_length": avg_length,
                        "entity_density": entity_density,
                        "coherence_score": coherence_score,
                        "semantic_centrality": semantic_centrality,
                        "semantic_variance": semantic_variance
                    })
        
        return pd.DataFrame(features) if features else pd.DataFrame()
    
    def _calculate_throughput_metrics(self, segments: List[Dict], link_config: ValueChainLink) -> Dict[str, Any]:
        """Calculate throughput metrics for a value chain link."""
        
        if not segments:
            return {"throughput": 0.0, "capacity_utilization": 0.0, "efficiency_score": 0.0}
        
        # Calculate semantic throughput (information processing rate)
        total_semantic_content = sum(seg["semantic_density"] for seg in segments)
        avg_coherence = np.mean([seg["local_coherence"] for seg in segments])
        
        # Estimate capacity utilization
        theoretical_max_segments = 100  # Based on typical document structure
        capacity_utilization = len(segments) / theoretical_max_segments
        
        # Calculate efficiency score combining multiple factors
        efficiency_components = {
            "content_density": total_semantic_content / len(segments),
            "coherence": avg_coherence,
            "completeness": min(1.0, len(segments) / 20)  # Minimum expected segments
        }
        
        efficiency_score = np.mean(list(efficiency_components.values()))
        
        # Calculate throughput considering conversion rates
        base_throughput = len(segments) * avg_coherence
        adjusted_throughput = base_throughput * np.mean(list(link_config.conversion_rates.values()))
        
        return {
            "throughput": float(adjusted_throughput),
            "capacity_utilization": float(capacity_utilization),
            "efficiency_score": float(efficiency_score),
            "efficiency_components": efficiency_components,
            "segment_count": len(segments),
            "semantic_content": float(total_semantic_content)
        }
    
    def _detect_bottlenecks(self, segments: List[Dict], link_config: ValueChainLink, 
                          features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bottlenecks using statistical and ML approaches."""
        
        bottleneck_analysis = {
            "detected_bottlenecks": [],
            "bottleneck_scores": {},
            "capacity_constraints": {},
            "improvement_potential": {}
        }
        
        if not segments:
            return bottleneck_analysis
        
        # Extract features for anomaly detection
        segment_features = np.array([
            [seg["semantic_density"], seg["local_coherence"], len(seg["text"])]
            for seg in segments
        ])
        
        if len(segment_features) > 1:
            # Detect anomalies (potential bottlenecks)
            anomaly_scores = self.bottleneck_detector.fit_predict(segment_features)
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            
            for idx in anomaly_indices:
                bottleneck_analysis["detected_bottlenecks"].append({
                    "segment_id": idx,
                    "segment_text": segments[idx]["text"][:200],
                    "anomaly_score": float(self.bottleneck_detector.decision_function([segment_features[idx]])[0]),
                    "bottleneck_type": "content_anomaly"
                })
        
        # Analyze capacity constraints
        for constraint_type, constraint_value in link_config.capacity_constraints.items():
            if constraint_value < 0.7:  # Threshold for constraint identification
                bottleneck_analysis["capacity_constraints"][constraint_type] = {
                    "current_capacity": constraint_value,
                    "severity": "high" if constraint_value < 0.5 else "medium",
                    "improvement_needed": 1.0 - constraint_value
                }
        
        # Calculate bottleneck scores for predefined bottleneck types
        for bottleneck_type in link_config.bottlenecks:
            # Score based on text analysis
            bottleneck_mentions = sum(
                1 for seg in segments 
                if bottleneck_type.replace("_", " ").lower() in seg["text"].lower()
            )
            
            score = bottleneck_mentions / len(segments) if segments else 0.0
            bottleneck_analysis["bottleneck_scores"][bottleneck_type] = {
                "mention_frequency": score,
                "severity": "high" if score > 0.2 else "medium" if score > 0.1 else "low"
            }
        
        return bottleneck_analysis
    
    def _calculate_conversion_rates(self, segments: List[Dict], link_config: ValueChainLink,
                                  historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate and predict conversion rates between value chain stages."""
        
        conversion_analysis = {
            "current_rates": dict(link_config.conversion_rates),
            "predicted_rates": {},
            "rate_trends": {},
            "optimization_potential": {}
        }
        
        if historical_data is not None and not historical_data.empty:
            # Use historical data for trend analysis
            for conversion_type, current_rate in link_config.conversion_rates.items():
                if conversion_type in historical_data.columns:
                    historical_rates = historical_data[conversion_type].dropna()
                    
                    if len(historical_rates) > 2:
                        # Fit trend line
                        x = np.arange(len(historical_rates))
                        coeffs = np.polyfit(x, historical_rates, 1)
                        trend_slope = coeffs[0]
                        
                        # Predict next period rate
                        predicted_rate = coeffs[0] * len(historical_rates) + coeffs[1]
                        predicted_rate = max(0.0, min(1.0, predicted_rate))  # Bound between 0 and 1
                        
                        conversion_analysis["predicted_rates"][conversion_type] = predicted_rate
                        conversion_analysis["rate_trends"][conversion_type] = {
                            "trend_slope": float(trend_slope),
                            "trend_direction": "improving" if trend_slope > 0 else "declining" if trend_slope < 0 else "stable",
                            "confidence": min(1.0, len(historical_rates) / 10)  # More data = higher confidence
                        }
        
        # Calculate optimization potential
        for conversion_type, current_rate in link_config.conversion_rates.items():
            theoretical_max = 0.95  # Realistic maximum considering real-world constraints
            optimization_potential = theoretical_max - current_rate
            
            conversion_analysis["optimization_potential"][conversion_type] = {
                "current_rate": current_rate,
                "theoretical_max": theoretical_max,
                "improvement_potential": float(optimization_potential),
                "priority": "high" if optimization_potential > 0.3 else "medium" if optimization_potential > 0.15 else "low"
            }
        
        return conversion_analysis
    
    def _analyze_lead_times(self, segments: List[Dict], link_config: ValueChainLink,
                          features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and predict lead times with uncertainty quantification."""
        
        lead_time_analysis = {
            "baseline_lead_time": link_config.lead_time_days,
            "predicted_lead_time": link_config.lead_time_days,
            "uncertainty_bounds": {},
            "lead_time_drivers": {},
            "optimization_recommendations": []
        }
        
        # Create features for lead time prediction
        if segments:
            complexity_features = [
                len(segments),  # Number of segments (complexity)
                np.mean([seg["semantic_density"] for seg in segments]),  # Content density
                np.mean([seg["local_coherence"] for seg in segments]),  # Coherence
                sum(len(seg["text"]) for seg in segments)  # Total content volume
            ]
            
            # Estimate lead time based on complexity
            complexity_factor = np.mean(complexity_features[:3])  # Normalize by first 3 features
            volume_factor = complexity_features[3] / 10000  # Scale volume appropriately
            
            # Apply complexity and volume adjustments
            complexity_multiplier = 1 + (1 - complexity_factor) * 0.3  # Low coherence increases time
            volume_multiplier = 1 + volume_factor * 0.2  # More content increases time
            
            predicted_lead_time = link_config.lead_time_days * complexity_multiplier * volume_multiplier
            
            # Calculate uncertainty bounds using bootstrap-like approach
            feature_std = np.std(complexity_features[:3])
            uncertainty = predicted_lead_time * feature_std * 0.1  # 10% uncertainty factor
            
            lead_time_analysis.update({
                "predicted_lead_time": float(predicted_lead_time),
                "complexity_multiplier": float(complexity_multiplier),
                "volume_multiplier": float(volume_multiplier),
                "uncertainty_bounds": {
                    "lower": float(predicted_lead_time - uncertainty),
                    "upper": float(predicted_lead_time + uncertainty),
                    "confidence_interval": 0.8
                }
            })
            
            # Identify lead time drivers
            lead_time_analysis["lead_time_drivers"] = {
                "content_complexity": float(1 - complexity_factor),
                "content_volume": float(volume_factor),
                "process_efficiency": float(np.mean(list(link_config.conversion_rates.values())))
            }
            
            # Generate optimization recommendations
            if complexity_multiplier > 1.2:
                lead_time_analysis["optimization_recommendations"].append(
                    "Improve content coherence and semantic clarity to reduce processing time"
                )
            
            if volume_multiplier > 1.3:
                lead_time_analysis["optimization_recommendations"].append(
                    "Consider content consolidation or parallel processing to manage volume"
                )
        
        return lead_time_analysis
    
    def _define_operational_loss_functions(self, link_config: ValueChainLink,
                                         throughput_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Define operational loss functions for performance optimization."""
        
        loss_functions = {
            "throughput_loss": {},
            "quality_loss": {},
            "time_loss": {},
            "resource_loss": {},
            "composite_loss": {}
        }
        
        # Throughput loss function (quadratic penalty for low throughput)
        current_throughput = throughput_metrics["throughput"]
        target_throughput = 100.0  # Target throughput level
        throughput_gap = max(0, target_throughput - current_throughput)
        
        loss_functions["throughput_loss"] = {
            "current_throughput": current_throughput,
            "target_throughput": target_throughput,
            "loss_value": float(throughput_gap ** 2),
            "loss_type": "quadratic",
            "optimization_gradient": float(-2 * throughput_gap)
        }
        
        # Quality loss function (exponential penalty for low quality)
        current_efficiency = throughput_metrics["efficiency_score"]
        target_efficiency = 0.9
        quality_gap = max(0, target_efficiency - current_efficiency)
        
        loss_functions["quality_loss"] = {
            "current_efficiency": current_efficiency,
            "target_efficiency": target_efficiency,
            "loss_value": float(np.exp(quality_gap * 3) - 1),
            "loss_type": "exponential",
            "penalty_multiplier": 3.0
        }
        
        # Time loss function (linear penalty for delays)
        baseline_time = link_config.lead_time_days
        capacity_utilization = throughput_metrics["capacity_utilization"]
        time_multiplier = 1 + (1 - capacity_utilization) * 0.5  # Underutilization increases time
        
        loss_functions["time_loss"] = {
            "baseline_time": baseline_time,
            "time_multiplier": float(time_multiplier),
            "estimated_actual_time": float(baseline_time * time_multiplier),
            "loss_value": float(baseline_time * (time_multiplier - 1)),
            "loss_type": "linear"
        }
        
        # Resource loss function (based on capacity constraints)
        resource_losses = []
        for constraint_type, constraint_value in link_config.capacity_constraints.items():
            if constraint_value < 1.0:
                resource_loss = (1.0 - constraint_value) ** 1.5  # Convex penalty
                resource_losses.append(resource_loss)
        
        loss_functions["resource_loss"] = {
            "individual_losses": dict(zip(link_config.capacity_constraints.keys(), resource_losses)),
            "total_resource_loss": float(sum(resource_losses)),
            "loss_type": "convex_combination"
        }
        
        # Composite loss function (weighted combination)
        weights = {"throughput": 0.3, "quality": 0.3, "time": 0.2, "resource": 0.2}
        composite_loss = (
            weights["throughput"] * loss_functions["throughput_loss"]["loss_value"] +
            weights["quality"] * loss_functions["quality_loss"]["loss_value"] +
            weights["time"] * loss_functions["time_loss"]["loss_value"] +
            weights["resource"] * loss_functions["resource_loss"]["total_resource_loss"]
        )
        
        loss_functions["composite_loss"] = {
            "total_loss": float(composite_loss),
            "weights": weights,
            "loss_components": {
                "throughput_component": float(weights["throughput"] * loss_functions["throughput_loss"]["loss_value"]),
                "quality_component": float(weights["quality"] * loss_functions["quality_loss"]["loss_value"]),
                "time_component": float(weights["time"] * loss_functions["time_loss"]["loss_value"]),
                "resource_component": float(weights["resource"] * loss_functions["resource_loss"]["total_resource_loss"])
            }
        }
        
        return loss_functions
    
    def _generate_predictive_indicators(self, performance_analysis: Dict[str, Any],
                                      features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictive indicators for future performance."""
        
        predictive_indicators = {
            "performance_forecast": {},
            "risk_indicators": {},
            "opportunity_indicators": {},
            "early_warning_signals": {}
        }
        
        # Performance forecast
        for link_name, metrics in performance_analysis["value_chain_metrics"].items():
            current_efficiency = metrics["efficiency_score"]
            throughput = metrics["throughput"]
            
            # Simple trend-based forecast (in production, use more sophisticated models)
            efficiency_trend = 0.02 if current_efficiency > 0.7 else -0.01  # Positive/negative trend
            throughput_trend = throughput * 0.05 if throughput > 50 else throughput * -0.02
            
            predictive_indicators["performance_forecast"][link_name] = {
                "efficiency_forecast_3m": float(min(1.0, current_efficiency + efficiency_trend * 3)),
                "efficiency_forecast_6m": float(min(1.0, current_efficiency + efficiency_trend * 6)),
                "throughput_forecast_3m": float(max(0, throughput + throughput_trend * 3)),
                "throughput_forecast_6m": float(max(0, throughput + throughput_trend * 6)),
                "confidence_level": 0.7
            }
        
        # Risk indicators
        for link_name, bottlenecks in performance_analysis["bottleneck_analysis"].items():
            risk_score = 0.0
            risk_factors = []
            
            # High bottleneck scores indicate risk
            for bottleneck_type, scores in bottlenecks["bottleneck_scores"].items():
                if scores["severity"] == "high":
                    risk_score += 0.3
                    risk_factors.append(f"High {bottleneck_type} risk")
                elif scores["severity"] == "medium":
                    risk_score += 0.1
            
            # Capacity constraints indicate risk
            for constraint_type, constraint_info in bottlenecks["capacity_constraints"].items():
                if constraint_info["severity"] == "high":
                    risk_score += 0.4
                    risk_factors.append(f"Critical {constraint_type} constraint")
            
            predictive_indicators["risk_indicators"][link_name] = {
                "overall_risk_score": float(min(1.0, risk_score)),
                "risk_level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
                "risk_factors": risk_factors
            }
        
        # Opportunity indicators
        for link_name, conversion_rates in performance_analysis["conversion_rates"].items():
            opportunities = []
            opportunity_score = 0.0
            
            for conversion_type, optimization_info in conversion_rates["optimization_potential"].items():
                if optimization_info["priority"] == "high":
                    opportunity_score += 0.3
                    opportunities.append(f"Optimize {conversion_type}")
            
            predictive_indicators["opportunity_indicators"][link_name] = {
                "opportunity_score": float(opportunity_score),
                "opportunities": opportunities,
                "potential_impact": "high" if opportunity_score > 0.5 else "medium" if opportunity_score > 0.2 else "low"
            }
        
        return predictive_indicators
    
    def _generate_optimization_recommendations(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations using operations research principles."""
        
        recommendations = {
            "immediate_actions": [],
            "strategic_initiatives": [],
            "resource_allocation": {},
            "process_improvements": {},
            "performance_targets": {}
        }
        
        # Analyze all links for optimization opportunities
        for link_name, metrics in performance_analysis["value_chain_metrics"].items():
            efficiency = metrics["efficiency_score"]
            throughput = metrics["throughput"]
            
            # Immediate actions for underperforming links
            if efficiency < 0.5:
                recommendations["immediate_actions"].append({
                    "link": link_name,
                    "action": "Critical efficiency improvement required",
                    "priority": "urgent",
                    "expected_impact": "high"
                })
            
            if throughput < 30:
                recommendations["immediate_actions"].append({
                    "link": link_name,
                    "action": "Throughput optimization needed",
                    "priority": "high",
                    "expected_impact": "medium"
                })
        
        # Strategic initiatives based on bottleneck analysis
        for link_name, bottlenecks in performance_analysis["bottleneck_analysis"].items():
            for constraint_type, constraint_info in bottlenecks["capacity_constraints"].items():
                if constraint_info["severity"] == "high":
                    recommendations["strategic_initiatives"].append({
                        "link": link_name,
                        "initiative": f"Capacity building for {constraint_type}",
                        "timeline": "medium_term",
                        "investment_required": "high" if constraint_info["improvement_needed"] > 0.5 else "medium"
                    })
        
        # Resource allocation recommendations
        total_improvement_potential = 0
        link_priorities = {}
        
        for link_name, loss_functions in performance_analysis["operational_loss_functions"].items():
            composite_loss = loss_functions["composite_loss"]["total_loss"]
            link_priorities[link_name] = composite_loss
            total_improvement_potential += composite_loss
        
        # Allocate resources proportionally to loss magnitude
        for link_name, loss_value in link_priorities.items():
            if total_improvement_potential > 0:
                allocation_percentage = (loss_value / total_improvement_potential) * 100
                recommendations["resource_allocation"][link_name] = {
                    "allocation_percentage": float(allocation_percentage),
                    "justification": f"High improvement potential (loss: {loss_value:.2f})",
                    "recommended_focus": "high" if allocation_percentage > 30 else "medium" if allocation_percentage > 15 else "low"
                }
        
        return recommendations

# ---------------------------------------------------------------------------
# 3. MULTIMODAL TEXT MINING FOR CRITICAL LINK DIAGNOSIS
# ---------------------------------------------------------------------------

class MultimodalTextMiner:
    """Advanced multimodal text mining for critical value chain link diagnosis."""
    
    def __init__(self, knowledge_graph: OntologicalKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        
        # Initialize advanced NLP models
        self.transformer_model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Initialize multilingual models
        self.multilingual_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        
        # Initialize topic modeling
        self.topic_model = BERTopic(
            embedding_model=self.multilingual_model,
            min_topic_size=5,
            nr_topics="auto",
            calculate_probabilities=True
        )
        
        # Initialize keyword extraction
        self.keyword_extractor = yake.KeywordExtractor(
            lan="es",
            n=3,
            dedupLim=0.7,
            top=20,
            features=None
        )
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            return_all_scores=True
        )
        
        # Initialize relation extraction
        self.relation_extractor = pipeline(
            "token-classification",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        
        logger.info("MultimodalTextMiner initialized with state-of-the-art models")
    
    def diagnose_critical_links(self, semantic_cube: Dict[str, Any],
                              performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive multimodal diagnosis of critical value chain links."""
        
        diagnosis_results = {
            "critical_link_identification": {},
            "risk_assessment": {},
            "capacity_analysis": {},
            "intervention_recommendations": {},
            "evidence_extraction": {},
            "causal_analysis": {}
        }
        
        # Identify critical links based on performance
        critical_links = self._identify_critical_links(performance_analysis)
        
        # Perform detailed analysis for each critical link
        for link_name, criticality_score in critical_links.items():
            link_segments = semantic_cube["dimensions"]["value_chain_links"].get(link_name, [])
            
            # Comprehensive text analysis
            text_analysis = self._perform_comprehensive_text_analysis(link_segments)
            
            # Risk assessment
            risk_assessment = self._assess_link_risks(link_segments, text_analysis)
            
            # Capacity analysis
            capacity_analysis = self._analyze_link_capacity(link_segments, text_analysis)
            
            # Extract structured evidence
            evidence_extraction = self._extract_structured_evidence(link_segments)
            
            # Causal relationship analysis
            causal_analysis = self._analyze_causal_relationships(link_segments, text_analysis)
            
            # Generate intervention recommendations
            interventions = self._generate_intervention_recommendations(
                link_name, text_analysis, risk_assessment, capacity_analysis, causal_analysis
            )
            
            # Store results
            diagnosis_results["critical_link_identification"][link_name] = {
                "criticality_score": criticality_score,
                "segment_count": len(link_segments),
                "analysis_confidence": self._calculate_analysis_confidence(link_segments, text_analysis)
            }
            
            diagnosis_results["risk_assessment"][link_name] = risk_assessment
            diagnosis_results["capacity_analysis"][link_name] = capacity_analysis
            diagnosis_results["evidence_extraction"][link_name] = evidence_extraction
            diagnosis_results["causal_analysis"][link_name] = causal_analysis
            diagnosis_results["intervention_recommendations"][link_name] = interventions
        
        logger.info(f"Diagnosed {len(critical_links)} critical value chain links")
        
        return diagnosis_results
    
    def _identify_critical_links(self, performance_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Identify critical links based on multiple performance indicators."""
        
        critical_links = {}
        
        for link_name in performance_analysis["value_chain_metrics"].keys():
            criticality_score = 0.0
            
            # Performance metrics contribution
            metrics = performance_analysis["value_chain_metrics"][link_name]
            if metrics["efficiency_score"] < 0.5:
                criticality_score += 0.3
            if metrics["throughput"] < 30:
                criticality_score += 0.2
            if metrics["capacity_utilization"] < 0.6:
                criticality_score += 0.2
            
            # Bottleneck analysis contribution
            bottlenecks = performance_analysis["bottleneck_analysis"][link_name]
            high_severity_bottlenecks = sum(
                1 for scores in bottlenecks["bottleneck_scores"].values()
                if scores["severity"] == "high"
            )
            criticality_score += min(0.3, high_severity_bottlenecks * 0.1)
            
            # Operational loss function contribution
            if link_name in performance_analysis["operational_loss_functions"]:
                composite_loss = performance_analysis["operational_loss_functions"][link_name]["composite_loss"]["total_loss"]
                normalized_loss = min(1.0, composite_loss / 100)  # Normalize to [0,1]
                criticality_score += normalized_loss * 0.2
            
            # Only include links above criticality threshold
            if criticality_score > 0.4:
                critical_links[link_name] = min(1.0, criticality_score)
        
        return critical_links
    
    def _perform_comprehensive_text_analysis(self, segments: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive multimodal text analysis."""
        
        if not segments:
            return {"error": "No segments available for analysis"}
        
        # Combine all segment texts
        combined_text = " ".join([seg["text"] for seg in segments])
        segment_texts = [seg["text"] for seg in segments]
        
        analysis_results = {
            "linguistic_analysis": {},
            "semantic_analysis": {},
            "topic_modeling": {},
            "sentiment_analysis": {},
            "keyword_extraction": {},
            "readability_metrics": {},
            "entity_recognition": {},
            "relation_extraction": {}
        }
        
        # Linguistic analysis using spaCy
        doc = self.knowledge_graph.nlp_es(combined_text)
        
        analysis_results["linguistic_analysis"] = {
            "token_count": len(doc),
            "sentence_count": len(list(doc.sents)),
            "entity_count": len(doc.ents),
            "pos_distribution": dict(Counter([token.pos_ for token in doc])),
            "dependency_patterns": self._extract_dependency_patterns(doc),
            "linguistic_complexity": self._calculate_linguistic_complexity(doc)
        }
        
        # Semantic analysis using transformers
        embeddings = self.multilingual_model.encode(segment_texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        analysis_results["semantic_analysis"] = {
            "semantic_coherence": float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
            "semantic_diversity": float(np.std(embeddings.flatten())),
            "cluster_analysis": self._perform_semantic_clustering(embeddings, segment_texts),
            "semantic_centrality": self._calculate_semantic_centrality(embeddings, similarity_matrix)
        }
        
        # Topic modeling
        try:
            topics, probabilities = self.topic_model.fit_transform(segment_texts)
            topic_info = self.topic_model.get_topic_info()
            
            analysis_results["topic_modeling"] = {
                "topic_count": len(set(topics)),
                "topic_distribution": dict(Counter(topics)),
                "topic_coherence": self._calculate_topic_coherence(topics, probabilities),
                "dominant_topics": self._extract_dominant_topics(topic_info)
            }
        except Exception as e:
            logger.warning(f"Topic modeling failed: {e}")
            analysis_results["topic_modeling"] = {"error": str(e)}
        
        # Sentiment analysis
        sentiment_results = []
        for text in segment_texts[:50]:  # Limit for API efficiency
            try:
                sentiment = self.sentiment_analyzer(text[:512])  # Truncate for model limits
                sentiment_results.append(sentiment)
            except Exception:
                continue
        
        if sentiment_results:
            analysis_results["sentiment_analysis"] = {
                "overall_sentiment": self._aggregate_sentiment(sentiment_results),
                "sentiment_distribution": self._analyze_sentiment_distribution(sentiment_results),
                "sentiment_trends": self._analyze_sentiment_trends(sentiment_results)
            }
        
        # Keyword extraction
        keywords = self.keyword_extractor.extract_keywords(combined_text)
        analysis_results["keyword_extraction"] = {
            "top_keywords": [(kw, score) for kw, score in keywords[:20]],
            "keyword_density": len(keywords) / len(combined_text.split()) if combined_text.split() else 0,
            "domain_specific_terms": self._identify_domain_terms(keywords)
        }
        
        # Readability metrics
        analysis_results["readability_metrics"] = {
            "flesch_reading_ease": flesch_reading_ease(combined_text),
            "flesch_kincaid_grade": flesch_kincaid_grade(combined_text),
            "avg_sentence_length": np.mean([len(sent.text.split()) for sent in doc.sents]),
            "lexical_diversity": len(set(combined_text.lower().split())) / len(combined_text.split()) if combined_text.split() else 0
        }
        
        # Named entity recognition
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        analysis_results["entity_recognition"] = {
            "entities": entities,
            "entity_types": dict(Counter([ent[1] for ent in entities])),
            "entity_density": len(entities) / len(doc) if len(doc) > 0 else 0,
            "key_entities": self._identify_key_entities(entities)
        }
        
        # Relation extraction
        try:
            relations = self._extract_relations(combined_text)
            analysis_results["relation_extraction"] = {
                "relations": relations,
                "relation_types": dict(Counter([rel["type"] for rel in relations])),
                "relation_network": self._build_relation_network(relations)
            }
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            analysis_results["relation_extraction"] = {"error": str(e)}
        
        return analysis_results
    
    def _extract_dependency_patterns(self, doc) -> Dict[str, int]:
        """Extract dependency parsing patterns."""
        patterns = defaultdict(int)
        for token in doc:
            if token.dep_ and token.head:
                pattern = f"{token.pos_}--{token.dep_}-->{token.head.pos_}"
                patterns[pattern] += 1
        return dict(patterns)
    
    def _calculate_linguistic_complexity(self, doc) -> float:
        """Calculate linguistic complexity score."""
        # Multiple complexity indicators
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        dependency_depth = np.mean([self._get_dependency_depth(token) for token in doc])
        pos_diversity = len(set(token.pos_ for token in doc)) / len(set(token.pos_ for token in doc if token.pos_)) if len(doc) > 0 else 0
        
        # Normalize and combine
        complexity_score = (
            min(1.0, avg_sentence_length / 30) * 0.4 +
            min(1.0, dependency_depth / 10) * 0.3 +
            pos_diversity * 0.3
        )
        
        return float(complexity_score)
    
    def _get_dependency_depth(self, token) -> int:
        """Calculate dependency tree depth for a token."""
        depth = 0
        current = token
        while current.head != current and depth < 10:  # Prevent infinite loops
            depth += 1
            current = current.head
        return depth
    
    def _perform_semantic_clustering(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Perform semantic clustering of text segments."""
        if len(embeddings) < 2:
            return {"clusters": 0, "cluster_labels": []}
        
        # Determine optimal number of clusters
        n_clusters = min(8, max(2, len(embeddings) // 3))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate cluster characteristics
            centroid = np.mean(cluster_embeddings, axis=0)
            intra_cluster_distance = np.mean([
                cosine(embedding, centroid) for embedding in cluster_embeddings
            ])
            
            cluster_analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_indices),
                "representative_texts": cluster_texts[:3],
                "cohesion": float(1 - intra_cluster_distance),
                "keywords": self._extract_cluster_keywords(cluster_texts)
            }
        
        return {
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": float(silhouette_avg),
            "cluster_analysis": cluster_analysis
        }
    
    def _extract_cluster_keywords(self, cluster_texts: List[str]) -> List[str]:
        """Extract representative keywords for a cluster."""
        combined_text = " ".join(cluster_texts)
        try:
            keywords = self.keyword_extractor.extract_keywords(combined_text)
            return [kw for kw, score in keywords[:5]]
        except Exception:
            return []
    
    def _calculate_semantic_centrality(self, embeddings: np.ndarray, similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate semantic centrality measures."""
        if len(embeddings) < 2:
            return {"centrality_scores": [], "most_central": None}
        
        # Create graph from similarity matrix
        G = nx.from_numpy_array(similarity_matrix)
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Combine centrality measures
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                degree_centrality[node] * 0.3 +
                closeness_centrality[node] * 0.3 +
                betweenness_centrality[node] * 0.2 +
                eigenvector_centrality[node] * 0.2
            )
        
        # Find most central segments
        most_central = max(combined_centrality.items(), key=lambda x: x[1])
        
        return {
            "centrality_scores": list(combined_centrality.values()),
            "most_central_segment": int(most_central[0]),
            "centrality_score": float(most_central[1]),
            "centrality_distribution": {
                "mean": float(np.mean(list(combined_centrality.values()))),
                "std": float(np.std(list(combined_centrality.values())))
            }
        }
    
    def _calculate_topic_coherence(self, topics: List[int], probabilities: np.ndarray) -> float:
        """Calculate topic coherence score."""
        if len(set(topics)) <= 1:
            return 1.0
        
        # Calculate within-topic probability variance
        topic_coherences = []
        for topic_id in set(topics):
            topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
            if len(topic_indices) > 1:
                topic_probs = probabilities[topic_indices, topic_id] if probabilities.ndim > 1 else [1.0] * len(topic_indices)
                coherence = 1 - np.std(topic_probs) if len(topic_probs) > 1 else 1.0
                topic_coherences.append(coherence)
        
        return float(np.mean(topic_coherences)) if topic_coherences else 0.0
    
    def _extract_dominant_topics(self, topic_info: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract dominant topics from topic modeling results."""
        dominant_topics = []
        
        if not topic_info.empty:
            # Sort by count (assuming topic_info has 'Count' column)
            sorted_topics = topic_info.sort_values('Count', ascending=False) if 'Count' in topic_info.columns else topic_info
            
            for idx, row in sorted_topics.head(5).iterrows():
                topic_dict = row.to_dict()
                dominant_topics.append({
                    "topic_id": topic_dict.get('Topic', idx),
                    "count": topic_dict.get('Count', 0),
                    "representation": topic_dict.get('Representation', topic_dict.get('Name', 'Unknown'))
                })
        
        return dominant_topics
    
    def _aggregate_sentiment(self, sentiment_results: List[List[Dict]]) -> Dict[str, Any]:
        """Aggregate sentiment analysis results."""
        all_sentiments = []
        for result_list in sentiment_results:
            if result_list:  # Check if list is not empty
                # Take the first (highest confidence) sentiment
                sentiment = result_list[0]
                all_sentiments.append(sentiment)
        
        if not all_sentiments:
            return {"overall": "neutral", "confidence": 0.0}
        
        # Calculate overall sentiment
        sentiment_counts = Counter([s['label'] for s in all_sentiments])
        most_common_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Calculate average confidence
        avg_confidence = np.mean([s['score'] for s in all_sentiments])
        
        return {
            "overall": most_common_sentiment,
            "confidence": float(avg_confidence),
            "distribution": dict(sentiment_counts)
        }
    
    def _analyze_sentiment_distribution(self, sentiment_results: List[List[Dict]]) -> Dict[str, float]:
        """Analyze sentiment distribution across segments."""
        all_sentiments = []
        for result_list in sentiment_results:
            if result_list:
                all_sentiments.append(result_list[0]['label'])
        
        if not all_sentiments:
            return {}
        
        total_count = len(all_sentiments)
        distribution = Counter(all_sentiments)
        
        return {label: count / total_count for label, count in distribution.items()}
    
    def _analyze_sentiment_trends(self, sentiment_results: List[List[Dict]]) -> Dict[str, Any]:
        """Analyze sentiment trends across document segments."""
        sentiments = []
        scores = []
        
        for result_list in sentiment_results:
            if result_list:
                sentiment = result_list[0]
                sentiments.append(sentiment['label'])
                # Convert to numerical score for trend analysis
                if sentiment['label'] in ['POSITIVE', '4 stars', '5 stars']:
                    scores.append(1.0)
                elif sentiment['label'] in ['NEGATIVE', '1 star', '2 stars']:
                    scores.append(-1.0)
                else:
                    scores.append(0.0)
        
        if len(scores) < 2:
            return {"trend": "stable", "slope": 0.0}
        
        # Calculate trend using linear regression
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        trend_direction = "improving" if slope > 0.1 else "declining" if slope < -0.1 else "stable"
        
        return {
            "trend": trend_direction,
            "slope": float(slope),
            "correlation": float(r_value),
            "significance": float(p_value)
        }
    
    def _identify_domain_terms(self, keywords: List[Tuple[str, float]]) -> List[str]:
        """Identify domain-specific terms from extracted keywords."""
        domain_terms = []
        
        # Predefined domain vocabularies
        municipal_terms = {
            'desarrollo', 'municipal', 'territorio', 'población', 'gobierno',
            'administración', 'gestión', 'política', 'programa', 'proyecto',
            'estrategia', 'objetivo', 'meta', 'indicador', 'seguimiento',
            'evaluación', 'participación', 'ciudadanía', 'comunidad'
        }
        
        for keyword, score in keywords:
            keyword_lower = keyword.lower()
            if any(term in keyword_lower for term in municipal_terms):
                domain_terms.append(keyword)
        
        return domain_terms[:10]  # Return top 10 domain terms
    
    def _identify_key_entities(self, entities: List[Tuple]) -> List[Dict[str, Any]]:
        """Identify key entities based on frequency and importance."""
        entity_counts = Counter([ent[0].lower() for ent in entities])
        entity_types = defaultdict(list)
        
        for entity_text, entity_type, start, end in entities:
            entity_types[entity_type].append(entity_text)
        
        key_entities = []
        
        # Most frequent entities
        for entity, count in entity_counts.most_common(10):
            # Find the entity type
            entity_type = None
            for ent_text, ent_type, _, _ in entities:
                if ent_text.lower() == entity:
                    entity_type = ent_type
                    break
            
            key_entities.append({
                "text": entity,
                "type": entity_type,
                "frequency": count,
                "importance": "high" if count > 3 else "medium" if count > 1 else "low"
            })
        
        return key_entities
    
    def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations using pattern-based and NLP approaches."""
        relations = []
        
        # Use spaCy for dependency parsing
        doc = self.knowledge_graph.nlp_es(text)
        
        # Extract relations based on dependency patterns
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.head.pos_ == 'VERB':
                relations.append({
                    "subject": token.text,
                    "predicate": token.head.text,
                    "object": self._find_object(token.head, doc),
                    "type": "action_relation",
                    "confidence": 0.7
                })
        
        # Extract causal relations
        causal_patterns = [
            r'(\w+)\s+(causa|produce|genera|resulta en|lleva a)\s+(\w+)',
            r'debido a\s+(\w+),\s+(\w+)',
            r'(\w+)\s+impacta\s+(\w+)'
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                if len(match.groups()) >= 2:
                    relations.append({
                        "subject": match.group(1),
                        "predicate": "causes" if "causa" in match.group(0) else "impacts",
                        "object": match.group(-1),
                        "type": "causal_relation",
                        "confidence": 0.8
                    })
        
        return relations
    
    def _find_object(self, verb_token, doc) -> str:
        """Find the object of a verb in dependency tree."""
        for child in verb_token.children:
            if child.dep_ in ['dobj', 'pobj']:
                return child.text
        return ""
    
    def _build_relation_network(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a network graph from extracted relations."""
        G = nx.DiGraph()
        
        for relation in relations:
            subject = relation.get("subject", "")
            predicate = relation.get("predicate", "")
            obj = relation.get("object", "")
            
            if subject and obj:
                G.add_edge(subject, obj, 
                          relation=predicate, 
                          type=relation.get("type", "unknown"),
                          confidence=relation.get("confidence", 0.5))
        
        # Calculate network metrics
        if len(G.nodes()) > 0:
            density = nx.density(G)
            try:
                avg_clustering = nx.average_clustering(G.to_undirected())
            except:
                avg_clustering = 0.0
                
            centrality = nx.degree_centrality(G)
            most_central = max(centrality.items(), key=lambda x: x[1]) if centrality else ("", 0)
            
            return {
                "node_count": len(G.nodes()),
                "edge_count": len(G.edges()),
                "density": float(density),
                "average_clustering": float(avg_clustering),
                "most_central_entity": most_central[0],
                "centrality_score": float(most_central[1])
            }
        else:
            return {"node_count": 0, "edge_count": 0, "density": 0.0}
    
    def _assess_link_risks(self, segments: List[Dict], text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for a value chain link based on text analysis."""
        
        risk_assessment = {
            "overall_risk_level": "low",
            "risk_factors": [],
            "risk_indicators": {},
            "mitigation_recommendations": []
        }
        
        # Analyze sentiment-based risks
        sentiment = text_analysis.get("sentiment_analysis", {})
        if sentiment:
            if sentiment.get("overall") in ["NEGATIVE", "1 star", "2 stars"]:
                risk_assessment["risk_factors"].append("Negative sentiment indicates potential issues")
                risk_assessment["risk_indicators"]["sentiment_risk"] = "high"
            
            sentiment_trend = sentiment.get("sentiment_trends", {})
            if sentiment_trend.get("trend") == "declining":
                risk_assessment["risk_factors"].append("Declining sentiment trend")
                risk_assessment["risk_indicators"]["trend_risk"] = "medium"
        
        # Analyze linguistic complexity risks
        linguistic = text_analysis.get("linguistic_analysis", {})
        if linguistic.get("linguistic_complexity", 0) > 0.8:
            risk_assessment["risk_factors"].append("High linguistic complexity may indicate unclear communication")
            risk_assessment["risk_indicators"]["communication_risk"] = "medium"
        
        # Analyze semantic coherence risks
        semantic = text_analysis.get("semantic_analysis", {})
        if semantic.get("semantic_coherence", 1.0) < 0.5:
            risk_assessment["risk_factors"].append("Low semantic coherence indicates fragmented approach")
            risk_assessment["risk_indicators"]["coherence_risk"] = "high"
        
        # Analyze keyword-based risks
        keywords = text_analysis.get("keyword_extraction", {})
        risk_keywords = ['problema', 'dificultad', 'limitación', 'restricción', 'falta', 'carencia', 'déficit']
        keyword_list = [kw for kw, score in keywords.get("top_keywords", [])]
        
        risk_mentions = sum(1 for keyword in keyword_list if any(risk_term in keyword.lower() for risk_term in risk_keywords))
        if risk_mentions > 3:
            risk_assessment["risk_factors"].append("High frequency of risk-related terms")
            risk_assessment["risk_indicators"]["content_risk"] = "medium"
        
        # Calculate overall risk level
        high_risks = sum(1 for indicator in risk_assessment["risk_indicators"].values() if indicator == "high")
        medium_risks = sum(1 for indicator in risk_assessment["risk_indicators"].values() if indicator == "medium")
        
        if high_risks > 1:
            risk_assessment["overall_risk_level"] = "high"
        elif high_risks > 0 or medium_risks > 2:
            risk_assessment["overall_risk_level"] = "medium"
        else:
            risk_assessment["overall_risk_level"] = "low"
        
        # Generate mitigation recommendations
        for risk_type, risk_level in risk_assessment["risk_indicators"].items():
            if risk_level in ["high", "medium"]:
                if risk_type == "sentiment_risk":
                    risk_assessment["mitigation_recommendations"].append(
                        "Address underlying issues causing negative sentiment through stakeholder engagement"
                    )
                elif risk_type == "coherence_risk":
                    risk_assessment["mitigation_recommendations"].append(
                        "Improve document structure and logical flow to enhance coherence"
                    )
                elif risk_type == "communication_risk":
                    risk_assessment["mitigation_recommendations"].append(
                        "Simplify language and improve clarity of communication"
                    )
        
        return risk_assessment
    
    def _analyze_link_capacity(self, segments: List[Dict], text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capacity indicators for a value chain link."""
        
        capacity_analysis = {
            "capacity_indicators": {},
            "capacity_gaps": [],
            "capacity_strengths": [],
            "development_recommendations": []
        }
        
        # Analyze content volume as capacity indicator
        total_content = sum(len(seg["text"]) for seg in segments)
        avg_segment_length = total_content / len(segments) if segments else 0
        
        capacity_analysis["capacity_indicators"]["content_volume"] = {
            "total_words": total_content,
            "avg_segment_length": avg_segment_length,
            "capacity_level": "high" if total_content > 5000 else "medium" if total_content > 2000 else "low"
        }
        
        # Analyze semantic diversity as capacity indicator
        semantic = text_analysis.get("semantic_analysis", {})
        semantic_diversity = semantic.get("semantic_diversity", 0)
        
        capacity_analysis["capacity_indicators"]["conceptual_diversity"] = {
            "diversity_score": semantic_diversity,
            "capacity_level": "high" if semantic_diversity > 0.3 else "medium" if semantic_diversity > 0.15 else "low"
        }
        
        # Analyze entity recognition as institutional capacity indicator
        entities = text_analysis.get("entity_recognition", {})
        entity_density = entities.get("entity_density", 0)
        entity_types_count = len(entities.get("entity_types", {}))
        
        capacity_analysis["capacity_indicators"]["institutional_references"] = {
            "entity_density": entity_density,
            "entity_types": entity_types_count,
            "capacity_level": "high" if entity_types_count > 5 else "medium" if entity_types_count > 2 else "low"
        }
        
        # Analyze topic modeling as thematic capacity
        topics = text_analysis.get("topic_modeling", {})
        if not topics.get("error"):
            topic_count = topics.get("topic_count", 0)
            topic_coherence = topics.get("topic_coherence", 0)
            
            capacity_analysis["capacity_indicators"]["thematic_capacity"] = {
                "topic_count": topic_count,
                "topic_coherence": topic_coherence,
                "capacity_level": "high" if topic_count > 3 and topic_coherence > 0.7 else "medium" if topic_count > 1 else "low"
            }
        
        # Identify capacity gaps and strengths
        for indicator_name, indicator_data in capacity_analysis["capacity_indicators"].items():
            capacity_level = indicator_data.get("capacity_level", "low")
            
            if capacity_level == "low":
                capacity_analysis["capacity_gaps"].append({
                    "area": indicator_name,
                    "severity": "high",
                    "description": f"Low capacity detected in {indicator_name.replace('_', ' ')}"
                })
            elif capacity_level == "high":
                capacity_analysis["capacity_strengths"].append({
                    "area": indicator_name,
                    "description": f"Strong capacity identified in {indicator_name.replace('_', ' ')}"
                })
        
        # Generate development recommendations
        if len(capacity_analysis["capacity_gaps"]) > len(capacity_analysis["capacity_strengths"]):
            capacity_analysis["development_recommendations"].append(
                "Priority focus on capacity building across multiple dimensions"
            )
        
        for gap in capacity_analysis["capacity_gaps"]:
            if gap["area"] == "content_volume":
                capacity_analysis["development_recommendations"].append(
                    "Enhance documentation and content development processes"
                )
            elif gap["area"] == "conceptual_diversity":
                capacity_analysis["development_recommendations"].append(
                    "Expand conceptual framework and analytical perspectives"
                )
            elif gap["area"] == "institutional_references":
                capacity_analysis["development_recommendations"].append(
                    "Strengthen institutional coordination and stakeholder engagement"
                )
        
        return capacity_analysis
    
    def _extract_structured_evidence(self, segments: List[Dict]) -> Dict[str, Any]:
        """Extract structured evidence using advanced NLP techniques."""
        
        evidence_extraction = {
            "quantitative_evidence": [],
            "qualitative_evidence": [],
            "causal_evidence": [],
            "temporal_evidence": [],
            "spatial_evidence": [],
            "evidence_quality_score": 0.0
        }
        
        if not segments:
            return evidence_extraction
        
        combined_text = " ".join([seg["text"] for seg in segments])
        
        # Extract quantitative evidence
        number_patterns = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\$\s*\d+(?:,\d+)*(?:\.\d+)?',  # Currency
            r'\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:millones?|miles?|billones?)?',  # Numbers with scale
            r'\b(?:año|años?)\s+\d{4}',  # Years
            r'\b\d+\s*(?:días?|meses?|semanas?|años?)',  # Time periods
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                evidence_extraction["quantitative_evidence"].append({
                    "value": match,
                    "type": "numerical",
                    "context": self._extract_context(combined_text, match)
                })
        
        # Extract qualitative evidence using entity recognition and keywords
        doc = self.knowledge_graph.nlp_es(combined_text)
        
        # Policy and program mentions
        policy_patterns = [
            r'\b(?:programa|proyecto|estrategia|política|iniciativa)\s+[A-Z][^.]*',
            r'\b(?:implementar|desarrollar|ejecutar|realizar)\s+[^.]*',
            r'\b(?:objetivo|meta|propósito|fin)\s*:\s*[^.]*'
        ]
        
        for pattern in policy_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                evidence_extraction["qualitative_evidence"].append({
                    "content": match[:200],  # Limit length
                    "type": "policy_reference",
                    "confidence": 0.8
                })
        
        # Extract causal evidence
        causal_indicators = [
            'causa', 'produce', 'genera', 'resulta en', 'lleva a', 'impacta',
            'debido a', 'por causa de', 'como resultado de', 'gracias a'
        ]
        
        sentences = [sent.text for sent in doc.sents]
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in causal_indicators):
                evidence_extraction["causal_evidence"].append({
                    "sentence": sentence,
                    "causal_indicators": [ind for ind in causal_indicators if ind in sentence.lower()],
                    "type": "causal_statement"
                })
        
        # Extract temporal evidence
        temporal_expressions = []
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                temporal_expressions.append({
                    "expression": ent.text,
                    "label": ent.label_,
                    "context": self._extract_context(combined_text, ent.text)
                })
        
        evidence_extraction["temporal_evidence"] = temporal_expressions
        
        # Extract spatial evidence
        spatial_expressions = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities and locations
                spatial_expressions.append({
                    "location": ent.text,
                    "label": ent.label_,
                    "context": self._extract_context(combined_text, ent.text)
                })
        
        evidence_extraction["spatial_evidence"] = spatial_expressions
        
        # Calculate evidence quality score
        quality_factors = {
            "quantitative_richness": min(1.0, len(evidence_extraction["quantitative_evidence"]) / 10),
            "qualitative_depth": min(1.0, len(evidence_extraction["qualitative_evidence"]) / 15),
            "causal_clarity": min(1.0, len(evidence_extraction["causal_evidence"]) / 8),
            "temporal_specificity": min(1.0, len(evidence_extraction["temporal_evidence"]) / 5),
            "spatial_context": min(1.0, len(evidence_extraction["spatial_evidence"]) / 5)
        }
        
        evidence_extraction["evidence_quality_score"] = np.mean(list(quality_factors.values()))
        evidence_extraction["quality_breakdown"] = quality_factors
        
        return evidence_extraction
    
    def _extract_context(self, text: str, target: str, window_size: int = 50) -> str:
        """Extract context around a target phrase."""
        try:
            index = text.lower().find(target.lower())
            if index == -1:
                return ""
            
            start = max(0, index - window_size)
            end = min(len(text), index + len(target) + window_size)
            
            return text[start:end].strip()
        except Exception:
            return ""
    
    def _analyze_causal_relationships(self, segments: List[Dict], text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causal relationships using advanced causal inference techniques."""
        
        causal_analysis = {
            "causal_chains": [],
            "causal_strength": {},
            "confounding_factors": [],
            "causal_gaps": [],
            "intervention_points": []
        }
        
        if not segments:
            return causal_analysis
        
        # Extract causal evidence from structured evidence
        evidence = text_analysis.get("evidence_extraction", {})
        causal_evidence = evidence.get("causal_evidence", [])
        
        # Build causal chains
        causal_chains = []
        for evidence_item in causal_evidence:
            sentence = evidence_item.get("sentence", "")
            indicators = evidence_item.get("causal_indicators", [])
            
            # Parse causal structure
            causal_chain = self._parse_causal_structure(sentence, indicators)
            if causal_chain:
                causal_chains.append(causal_chain)
        
        causal_analysis["causal_chains"] = causal_chains
        
        # Analyze causal strength using relation extraction
        relations = text_analysis.get("relation_extraction", {}).get("relations", [])
        causal_relations = [rel for rel in relations if rel.get("type") == "causal_relation"]
        
        for relation in causal_relations:
            subject = relation.get("subject", "")
            object_val = relation.get("object", "")
            confidence = relation.get("confidence", 0.5)
            
            causal_analysis["causal_strength"][f"{subject}->{object_val}"] = confidence
        
        # Identify potential confounding factors
        entities = text_analysis.get("entity_recognition", {}).get("entities", [])
        frequent_entities = [ent for ent in entities if ent[0] and len(ent[0]) > 3]  # Filter short entities
        
        # Look for entities that appear in multiple causal contexts
        entity_causal_contexts = defaultdict(int)
        for chain in causal_chains:
            for entity in frequent_entities:
                entity_text = entity[0].lower()
                if entity_text in chain.get("cause", "").lower() or entity_text in chain.get("effect", "").lower():
                    entity_causal_contexts[entity[0]] += 1
        
        # Entities appearing in multiple causal contexts might be confounders
        potential_confounders = [entity for entity, count in entity_causal_contexts.items() if count > 2]
        causal_analysis["confounding_factors"] = potential_confounders[:5]  # Top 5 potential confounders
        
        # Identify causal gaps (missing links in theory of change)
        value_chain_links = list(self.knowledge_graph.config["value_chain_links"].keys())
        mentioned_links = []
        
        combined_text = " ".join([seg["text"] for seg in segments]).lower()
        for link in value_chain_links:
            if link.replace("_", " ") in combined_text:
                mentioned_links.append(link)
        
        missing_links = set(value_chain_links) - set(mentioned_links)
        for missing_link in missing_links:
            causal_analysis["causal_gaps"].append({
                "missing_element": missing_link,
                "type": "value_chain_gap",
                "description": f"Limited reference to {missing_link.replace('_', ' ')} in causal reasoning"
            })
        
        # Identify intervention points based on causal analysis
        # Look for causes that appear frequently (high leverage points)
        cause_frequency = defaultdict(int)
        for chain in causal_chains:
            cause = chain.get("cause", "")
            if cause:
                cause_frequency[cause] += 1
        
        high_leverage_causes = [cause for cause, freq in cause_frequency.items() if freq > 1]
        for cause in high_leverage_causes[:3]:  # Top 3 intervention points
            causal_analysis["intervention_points"].append({
                "leverage_point": cause,
                "frequency": cause_frequency[cause],
                "type": "high_impact_cause",
                "recommendation": f"Focus interventions on {cause} for maximum systemic impact"
            })
        
        return causal_analysis
    
    def _parse_causal_structure(self, sentence: str, indicators: List[str]) -> Optional[Dict[str, Any]]:
        """Parse causal structure from a sentence."""
        try:
            # Simple pattern-based parsing
            for indicator in indicators:
                if indicator in sentence.lower():
                    parts = sentence.lower().split(indicator)
                    if len(parts) >= 2:
                        cause = parts[0].strip()[-50:]  # Last 50 chars before indicator
                        effect = parts[1].strip()[:50]   # First 50 chars after indicator
                        
                        return {
                            "cause": cause,
                            "effect": effect,
                            "indicator": indicator,
                            "confidence": 0.7,
                            "full_sentence": sentence
                        }
            return None
        except Exception:
            return None
    
    def _generate_intervention_recommendations(self, link_name: str, text_analysis: Dict[str, Any],
                                             risk_assessment: Dict[str, Any], capacity_analysis: Dict[str, Any],
                                             causal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive intervention recommendations."""
        
        recommendations = {
            "immediate_interventions": [],
            "strategic_interventions": [],
            "capacity_building": [],
            "system_interventions": [],
            "monitoring_requirements": []
        }
        
        # Immediate interventions based on risks
        high_risks = [factor for factor in risk_assessment["risk_factors"] if "high" in factor.lower()]
        for risk in high_risks[:3]:  # Top 3 high risks
            recommendations["immediate_interventions"].append({
                "intervention": f"Address {risk.lower()}",
                "priority": "urgent",
                "timeline": "1-3 months",
                "resources_required": "medium"
            })
        
        # Strategic interventions based on causal analysis
        intervention_points = causal_analysis.get("intervention_points", [])
        for point in intervention_points:
            recommendations["strategic_interventions"].append({
                "intervention": point["recommendation"],
                "leverage_point": point["leverage_point"],
                "priority": "high",
                "timeline": "6-12 months",
                "expected_impact": "systemic"
            })
        
        # Capacity building recommendations
        capacity_gaps = capacity_analysis.get("capacity_gaps", [])
        for gap in capacity_gaps:
            if gap["severity"] == "high":
                recommendations["capacity_building"].append({
                    "area": gap["area"],
                    "intervention": f"Strengthen {gap['area'].replace('_', ' ')} capacity",
                    "approach": "training_and_systems",
                    "timeline": "3-6 months"
                })
        
        # System-level interventions based on causal gaps
        causal_gaps = causal_analysis.get("causal_gaps", [])
        for gap in causal_gaps:
            recommendations["system_interventions"].append({
                "gap": gap["missing_element"],
                "intervention": f"Strengthen integration with {gap['missing_element'].replace('_', ' ')}",
                "type": "cross_cutting",
                "timeline": "long_term"
            })
        
        # Monitoring requirements
        evidence_quality = text_analysis.get("evidence_extraction", {}).get("evidence_quality_score", 0)
        if evidence_quality < 0.6:
            recommendations["monitoring_requirements"].append({
                "requirement": "Improve evidence collection and documentation",
                "focus": "data_quality",
                "frequency": "continuous"
            })
        
        # Add specific monitoring for high-risk areas
        risk_indicators = risk_assessment.get("risk_indicators", {})
        for risk_type, level in risk_indicators.items():
            if level == "high":
                recommendations["monitoring_requirements"].append({
                    "requirement": f"Monitor {risk_type.replace('_', ' ')} indicators",
                    "focus": risk_type,
                    "frequency": "monthly"
                })
        
        return recommendations
    
    def _calculate_analysis_confidence(self, segments: List[Dict], text_analysis: Dict[str, Any]) -> float:
        """Calculate confidence level for the analysis."""
        
        confidence_factors = []
        
        # Data sufficiency
        segment_count = len(segments)
        data_sufficiency = min(1.0, segment_count / 20)  # 20 segments as ideal
        confidence_factors.append(data_sufficiency)
        
        # Evidence quality
        evidence_quality = text_analysis.get("evidence_extraction", {}).get("evidence_quality_score", 0.5)
        confidence_factors.append(evidence_quality)
        
        # Semantic coherence
        semantic_coherence = text_analysis.get("semantic_analysis", {}).get("semantic_coherence", 0.5)
        confidence_factors.append(semantic_coherence)
        
        # Analysis completeness (availability of different analysis types)
        analysis_completeness = 0
        analysis_types = ["linguistic_analysis", "semantic_analysis", "sentiment_analysis", 
                         "entity_recognition", "relation_extraction"]
        
        for analysis_type in analysis_types:
            if analysis_type in text_analysis and not text_analysis[analysis_type].get("error"):
                analysis_completeness += 0.2
        
        confidence_factors.append(analysis_completeness)
        
        # Calculate weighted average
        weights = [0.3, 0.25, 0.25, 0.2]  # Data, evidence, coherence, completeness
        overall_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return min(1.0, max(0.0, overall_confidence))

# ---------------------------------------------------------------------------
# 4. RELATIONAL MODEL NORMALIZATION AND PROCESS MINING
# ---------------------------------------------------------------------------

class RelationalModelNormalizer:
    """Advanced relational model normalization for portfolio→program→project→activity→product hierarchy."""
    
    def __init__(self, knowledge_graph: OntologicalKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.schema_validator = None
        self.process_miner = None
        self._initialize_schemas()
        
        logger.info("RelationalModelNormalizer initialized with formal schemas")
    
    def _initialize_schemas(self):
        """Initialize database schemas and validation rules."""
        
        # Define schema using pandera for validation
        self.portfolio_schema = DataFrameSchema({
            "portfolio_id": PanderaColumn(str, required=True, unique=True),
            "portfolio_name": PanderaColumn(str, required=True),
            "strategic_objective": PanderaColumn(str, required=True),
            "budget_total": PanderaColumn(float, ge=0),
            "start_date": PanderaColumn(pd.Timestamp),
            "end_date": PanderaColumn(pd.Timestamp),
            "responsible_entity": PanderaColumn(str, required=True),
            "status": PanderaColumn(str, isin=["planned", "active", "completed", "cancelled"]),
            "created_at": PanderaColumn(pd.Timestamp, required=True),
            "updated_at": PanderaColumn(pd.Timestamp, required=True)
        })
        
        self.program_schema = DataFrameSchema({
            "program_id": PanderaColumn(str, required=True, unique=True),
            "portfolio_id": PanderaColumn(str, required=True),  # Foreign key
            "program_name": PanderaColumn(str, required=True),
            "program_objective": PanderaColumn(str, required=True),
            "budget_allocated": PanderaColumn(float, ge=0),
            "expected_outcomes": PanderaColumn(str),
            "target_population": PanderaColumn(str),
            "geographic_scope": PanderaColumn(str),
            "start_date": PanderaColumn(pd.Timestamp),
            "end_date": PanderaColumn(pd.Timestamp),
            "responsible_unit": PanderaColumn(str, required=True),
            "status": PanderaColumn(str, isin=["design", "implementation", "monitoring", "evaluation", "closed"]),
            "created_at": PanderaColumn(pd.Timestamp, required=True),
            "updated_at": PanderaColumn(pd.Timestamp, required=True)
        })
        
        self.project_schema = DataFrameSchema({
            "project_id": PanderaColumn(str, required=True, unique=True),
            "program_id": PanderaColumn(str, required=True),  # Foreign key
            "project_name": PanderaColumn(str, required=True),
            "project_description": PanderaColumn(str, required=True),
            "budget_assigned": PanderaColumn(float, ge=0),
            "expected_products": PanderaColumn(str),
            "implementation_strategy": PanderaColumn(str),
            "risk_level": PanderaColumn(str, isin=["low", "medium", "high"]),
            "start_date": PanderaColumn(pd.Timestamp),
            "end_date": PanderaColumn(pd.Timestamp),
            "project_manager": PanderaColumn(str, required=True),
            "status": PanderaColumn(str, isin=["planning", "execution", "monitoring", "closure"]),
            "completion_percentage": PanderaColumn(float, ge=0, le=100),
            "created_at": PanderaColumn(pd.Timestamp, required=True),
            "updated_at": PanderaColumn(pd.Timestamp, required=True)
        })
        
        self.activity_schema = DataFrameSchema({
            "activity_id": PanderaColumn(str, required=True, unique=True),
            "project_id": PanderaColumn(str, required=True),  # Foreign key
            "activity_name": PanderaColumn(str, required=True),
            "activity_description": PanderaColumn(str, required=True),
            "activity_type": PanderaColumn(str, required=True),
            "budget_activity": PanderaColumn(float, ge=0),
            "duration_days": PanderaColumn(int, ge=1),
            "dependencies": PanderaColumn(str),  # JSON string of dependency IDs
            "resources_required": PanderaColumn(str),  # JSON string of resources
            "value_chain_link": PanderaColumn(str, required=True),
            "start_date": PanderaColumn(pd.Timestamp),
            "end_date": PanderaColumn(pd.Timestamp),
            "responsible_person": PanderaColumn(str, required=True),
            "status": PanderaColumn(str, isin=["not_started", "in_progress", "completed", "blocked", "cancelled"]),
            "completion_percentage": PanderaColumn(float, ge=0, le=100),
            "created_at": PanderaColumn(pd.Timestamp, required=True),
            "updated_at": PanderaColumn(pd.Timestamp, required=True)
        })
        
        self.product_schema = DataFrameSchema({
            "product_id": PanderaColumn(str, required=True, unique=True),
            "activity_id": PanderaColumn(str, required=True),  # Foreign key
            "product_name": PanderaColumn(str, required=True),
            "product_description": PanderaColumn(str, required=True),
            "product_type": PanderaColumn(str, required=True),
            "quality_standards": PanderaColumn(str),
            "verification_criteria": PanderaColumn(str, required=True),
            "target_quantity": PanderaColumn(float, ge=0),
            "actual_quantity": PanderaColumn(float, ge=0),
            "unit_measure": PanderaColumn(str, required=True),
            "delivery_date": PanderaColumn(pd.Timestamp),
            "quality_score": PanderaColumn(float, ge=0, le=10),
            "beneficiaries_reached": PanderaColumn(int, ge=0),
            "value_chain_contribution": PanderaColumn(str, required=True),
            "status": PanderaColumn(str, isin=["planned", "in_production", "delivered", "verified", "rejected"]),
            "created_at": PanderaColumn(pd.Timestamp, required=True),
            "updated_at": PanderaColumn(pd.Timestamp, required=True)
        })
    
    def normalize_programmatic_structure(self, semantic_cube: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize semantic cube into relational model structure."""
        
        normalization_results = {
            "normalized_tables": {},
            "referential_integrity": {},
            "value_chain_mappings": {},
            "process_flows": {},
            "data_quality_report": {}
        }
        
        # Extract and normalize data from semantic cube
        portfolios_df = self._extract_portfolios(semantic_cube)
        programs_df = self._extract_programs(semantic_cube, portfolios_df)
        projects_df = self._extract_projects(semantic_cube, programs_df)
        activities_df = self._extract_activities(semantic_cube, projects_df)
        products_df = self._extract_products(semantic_cube, activities_df)
        
        # Validate schemas
        validation_results = self._validate_schemas({
            "portfolios": portfolios_df,
            "programs": programs_df,
            "projects": projects_df,
            "activities": activities_df,
            "products": products_df
        })
        
        # Store normalized tables
        normalization_results["normalized_tables"] = {
            "portfolios": portfolios_df,
            "programs": programs_df,
            "projects": projects_df,
            "activities": activities_df,
            "products": products_df
        }
        
        # Check referential integrity
        normalization_results["referential_integrity"] = self._check_referential_integrity(
            normalization_results["normalized_tables"]
        )
        
        # Create value chain mappings
        normalization_results["value_chain_mappings"] = self._create_value_chain_mappings(
            activities_df, products_df
        )
        
        # Generate process flows
        normalization_results["process_flows"] = self._generate_process_flows(
            normalization_results["normalized_tables"]
        )
        
        # Data quality report
        normalization_results["data_quality_report"] = {
            "validation_results": validation_results,
            "completeness_scores": self._calculate_completeness_scores(normalization_results["normalized_tables"]),
            "consistency_checks": self._perform_consistency_checks(normalization_results["normalized_tables"])
        }
        
        logger.info("Programmatic structure normalized into relational model")
        
        return normalization_results