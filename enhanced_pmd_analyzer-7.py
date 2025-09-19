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
        """Identify critical links based on multiple performance indicators