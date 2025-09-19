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
        concept_count = len(