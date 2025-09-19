#!/usr/bin/env python3
"""
Semantic Cubes Ontological Coding Module

This module implements an advanced ontological knowledge graph system that combines
RDF/OWL ontology management with state-of-the-art NLP processing for semantic
relationship extraction and representation.

Author: MICROFARFAN777
"""

import logging
import re
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib

# RDF/OWL and Knowledge Graph dependencies
try:
    import rdflib
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, BNode, Literal
    from rdflib.namespace import FOAF, DC, DCTERMS
    from rdflib.plugins.stores import memory
except ImportError:
    raise ImportError("rdflib is required. Install with: pip install rdflib")

# NLP and Transformer dependencies
try:
    import spacy
    from spacy.lang.en import English
except ImportError:
    raise ImportError("spaCy is required. Install with: pip install spacy")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    SentenceTransformer = None
    np = None

# Optional dependencies for enhanced functionality
try:
    import networkx as nx
except ImportError:
    nx = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticEntity:
    """Represents a semantic entity extracted from text."""
    text: str
    label: str
    uri: str
    confidence: float
    start: int
    end: int
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class SemanticRelation:
    """Represents a semantic relationship between entities."""
    subject: SemanticEntity
    predicate: str
    object: Union[SemanticEntity, str]
    confidence: float
    context: str = ""
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class SemanticCube:
    """
    Represents a semantic cube containing structured semantic information
    extracted from documents or structured data.
    """
    
    def __init__(self, cube_id: str, source: str = ""):
        self.cube_id = cube_id
        self.source = source
        self.entities: List[SemanticEntity] = []
        self.relations: List[SemanticRelation] = []
        self.metadata: Dict[str, Any] = {}
        self.confidence_score: float = 0.0
    
    def add_entity(self, entity: SemanticEntity) -> None:
        """Add an entity to the semantic cube."""
        self.entities.append(entity)
    
    def add_relation(self, relation: SemanticRelation) -> None:
        """Add a relation to the semantic cube."""
        self.relations.append(relation)
    
    def get_rdf_triples(self, namespace: Namespace) -> List[Tuple]:
        """Convert semantic cube to RDF triples."""
        triples = []
        
        # Add entity triples
        for entity in self.entities:
            entity_uri = URIRef(entity.uri)
            triples.append((entity_uri, RDF.type, URIRef(namespace + entity.label)))
            triples.append((entity_uri, RDFS.label, Literal(entity.text)))
            
            for prop, value in entity.properties.items():
                triples.append((entity_uri, URIRef(namespace + prop), Literal(value)))
        
        # Add relation triples
        for relation in self.relations:
            subject_uri = URIRef(relation.subject.uri)
            predicate_uri = URIRef(namespace + relation.predicate)
            
            if isinstance(relation.object, SemanticEntity):
                object_uri = URIRef(relation.object.uri)
            else:
                object_uri = Literal(relation.object)
            
            triples.append((subject_uri, predicate_uri, object_uri))
        
        return triples


class OntologicalKnowledgeGraph:
    """
    Advanced ontological knowledge graph system that combines RDF/OWL ontology
    management with NLP-powered semantic extraction and reasoning capabilities.
    """
    
    def __init__(self, base_uri: str = "http://microfarfan777.com/ontology/"):
        """
        Initialize the ontological knowledge graph.
        
        Args:
            base_uri: Base URI for the ontology namespace
        """
        self.base_uri = base_uri
        self.graph = Graph()
        self.namespace = Namespace(base_uri)
        self.semantic_cubes: Dict[str, SemanticCube] = {}
        
        # Initialize NLP components
        self._init_nlp_pipeline()
        self._init_transformer_models()
        
        # Bind common namespaces
        self._bind_namespaces()
        
        # Initialize ontology schema
        self._init_ontology_schema()
        
        logger.info(f"Initialized OntologicalKnowledgeGraph with base URI: {base_uri}")
    
    def _init_nlp_pipeline(self):
        """Initialize spaCy NLP pipeline with enhanced components."""
        try:
            # Try to load transformer model first, fallback to smaller model
            try:
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("Loaded transformer-based spaCy model")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded small spaCy model (transformer model not available)")
                except OSError:
                    # Create basic English pipeline
                    self.nlp = English()
                    logger.warning("Using basic English pipeline (no trained model available)")
                    
        except Exception as e:
            logger.error(f"Failed to initialize NLP pipeline: {e}")
            self.nlp = None
    
    def _init_transformer_models(self):
        """Initialize transformer models for semantic similarity and embeddings."""
        self.sentence_transformer = None
        if SentenceTransformer is not None:
            try:
                # Initialize sentence transformer for semantic similarity
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer model")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
    
    def _bind_namespaces(self):
        """Bind common RDF namespaces."""
        self.graph.bind("owl", OWL)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("dc", DC)
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("onto", self.namespace)
    
    def _init_ontology_schema(self):
        """Initialize basic ontology schema with core classes and properties."""
        # Define core classes
        classes = [
            "Document", "Entity", "Person", "Organization", "Location", 
            "Concept", "Event", "SemanticCube", "Relationship"
        ]
        
        for cls in classes:
            class_uri = self.namespace[cls]
            self.graph.add((class_uri, RDF.type, OWL.Class))
            self.graph.add((class_uri, RDFS.label, Literal(cls)))
        
        # Define core properties
        properties = [
            ("hasEntity", "Document", "Entity"),
            ("hasRelation", "Document", "Relationship"),
            ("mentions", "Document", "Entity"),
            ("relatedTo", "Entity", "Entity"),
            ("hasConfidence", "Entity", None),
            ("extractedFrom", "Entity", "Document"),
            ("hasSemanticType", "Entity", None)
        ]
        
        for prop, domain, range_cls in properties:
            prop_uri = self.namespace[prop]
            self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((prop_uri, RDFS.label, Literal(prop)))
            
            if domain:
                self.graph.add((prop_uri, RDFS.domain, self.namespace[domain]))
            if range_cls:
                self.graph.add((prop_uri, RDFS.range, self.namespace[range_cls]))
    
    def validate_schema(self, triples: List[Tuple]) -> Tuple[bool, List[str]]:
        """
        Validate RDF triples against the ontology schema.
        
        Args:
            triples: List of RDF triples to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Create temporary graph for validation
        temp_graph = Graph()
        for triple in triples:
            temp_graph.add(triple)
        
        # Basic validation checks
        for subject, predicate, obj in triples:
            # Check if predicate is defined in schema
            if not list(self.graph.triples((predicate, RDF.type, None))):
                errors.append(f"Undefined predicate: {predicate}")
            
            # Check domain/range constraints
            domain_triples = list(self.graph.triples((predicate, RDFS.domain, None)))
            range_triples = list(self.graph.triples((predicate, RDFS.range, None)))
            
            if domain_triples:
                expected_domain = domain_triples[0][2]
                subject_types = list(temp_graph.triples((subject, RDF.type, None)))
                if not any(obj_type[2] == expected_domain for obj_type in subject_types):
                    errors.append(f"Domain constraint violation for {predicate}")
        
        return len(errors) == 0, errors
    
    def extract_entities_nlp(self, text: str, doc_uri: str = None) -> List[SemanticEntity]:
        """
        Extract entities from text using NLP pipeline.
        
        Args:
            text: Input text to process
            doc_uri: URI of the source document
            
        Returns:
            List of extracted semantic entities
        """
        if not self.nlp:
            logger.error("NLP pipeline not initialized")
            return []
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Generate URI for entity
            entity_hash = hashlib.md5(ent.text.encode()).hexdigest()[:8]
            entity_uri = f"{self.base_uri}entity/{ent.label_}_{entity_hash}"
            
            semantic_entity = SemanticEntity(
                text=ent.text,
                label=ent.label_,
                uri=entity_uri,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                start=ent.start_char,
                end=ent.end_char,
                properties={
                    "entity_type": ent.label_,
                    "source_document": doc_uri or "unknown"
                }
            )
            
            entities.append(semantic_entity)
        
        return entities
    
    def extract_relations_nlp(self, text: str, entities: List[SemanticEntity]) -> List[SemanticRelation]:
        """
        Extract semantic relations from text using dependency parsing.
        
        Args:
            text: Input text to process
            entities: List of entities to find relations between
            
        Returns:
            List of extracted semantic relations
        """
        if not self.nlp:
            return []
        
        relations = []
        doc = self.nlp(text)
        
        # Simple relation extraction based on dependency parsing
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                # Find if token corresponds to any entity
                for entity in entities:
                    if entity.start <= token.idx < entity.end:
                        # Look for related entities through dependency relations
                        head = token.head
                        for other_entity in entities:
                            if other_entity != entity and other_entity.start <= head.idx < other_entity.end:
                                relation = SemanticRelation(
                                    subject=entity,
                                    predicate=f"dependency_{token.dep_}",
                                    object=other_entity,
                                    confidence=0.8,
                                    context=text[max(0, token.idx-50):token.idx+50],
                                    properties={
                                        "dependency_type": token.dep_,
                                        "head_token": head.text
                                    }
                                )
                                relations.append(relation)
        
        return relations
    
    def process_document(self, text: str, document_id: str = None, metadata: Dict[str, Any] = None) -> SemanticCube:
        """
        Process a document and extract semantic information into a semantic cube.
        
        Args:
            text: Document text to process
            document_id: Unique identifier for the document
            metadata: Additional metadata about the document
            
        Returns:
            SemanticCube containing extracted semantic information
        """
        if not document_id:
            document_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        doc_uri = f"{self.base_uri}document/{document_id}"
        cube = SemanticCube(document_id, doc_uri)
        cube.metadata = metadata or {}
        
        # Extract entities using NLP
        entities = self.extract_entities_nlp(text, doc_uri)
        for entity in entities:
            cube.add_entity(entity)
        
        # Extract relations
        relations = self.extract_relations_nlp(text, entities)
        for relation in relations:
            cube.add_relation(relation)
        
        # Calculate overall confidence score
        if entities:
            cube.confidence_score = sum(e.confidence for e in entities) / len(entities)
        
        # Store semantic cube
        self.semantic_cubes[document_id] = cube
        
        logger.info(f"Processed document {document_id}: {len(entities)} entities, {len(relations)} relations")
        return cube
    
    def add_structured_data(self, data: Dict[str, Any], cube_id: str = None) -> SemanticCube:
        """
        Process structured data and convert it to semantic relationships.
        
        Args:
            data: Structured data dictionary
            cube_id: Unique identifier for the semantic cube
            
        Returns:
            SemanticCube containing the structured data as semantic relationships
        """
        if not cube_id:
            cube_id = f"structured_{hashlib.md5(str(data).encode()).hexdigest()[:12]}"
        
        cube = SemanticCube(cube_id, f"{self.base_uri}structured/{cube_id}")
        
        def process_dict(obj: Dict[str, Any], parent_uri: str = None) -> List[SemanticEntity]:
            entities = []
            
            for key, value in obj.items():
                entity_uri = f"{parent_uri}#{key}" if parent_uri else f"{self.base_uri}entity/{key}"
                
                if isinstance(value, dict):
                    # Nested dictionary - create entity and process recursively
                    entity = SemanticEntity(
                        text=key,
                        label="StructuredEntity",
                        uri=entity_uri,
                        confidence=1.0,
                        start=0,
                        end=len(key),
                        properties={"data_type": "dict", "parent": parent_uri}
                    )
                    entities.append(entity)
                    
                    # Process nested entities
                    nested_entities = process_dict(value, entity_uri)
                    entities.extend(nested_entities)
                    
                elif isinstance(value, list):
                    # List - create entity and relations to list items
                    entity = SemanticEntity(
                        text=key,
                        label="StructuredEntity",
                        uri=entity_uri,
                        confidence=1.0,
                        start=0,
                        end=len(key),
                        properties={"data_type": "list", "list_length": len(value)}
                    )
                    entities.append(entity)
                    
                    for i, item in enumerate(value):
                        item_entity = SemanticEntity(
                            text=str(item),
                            label="ListItem",
                            uri=f"{entity_uri}_item_{i}",
                            confidence=1.0,
                            start=0,
                            end=len(str(item)),
                            properties={"list_index": i, "parent_list": entity_uri}
                        )
                        entities.append(item_entity)
                        
                        relation = SemanticRelation(
                            subject=entity,
                            predicate="hasListItem",
                            object=item_entity,
                            confidence=1.0,
                            properties={"list_index": i}
                        )
                        cube.add_relation(relation)
                
                else:
                    # Simple value - create entity
                    entity = SemanticEntity(
                        text=f"{key}: {value}",
                        label="DataProperty",
                        uri=entity_uri,
                        confidence=1.0,
                        start=0,
                        end=len(str(value)),
                        properties={"key": key, "value": value, "data_type": type(value).__name__}
                    )
                    entities.append(entity)
            
            return entities
        
        # Process the structured data
        entities = process_dict(data)
        for entity in entities:
            cube.add_entity(entity)
        
        self.semantic_cubes[cube_id] = cube
        return cube
    
    def store_semantic_cube(self, cube: SemanticCube, validate: bool = True) -> bool:
        """
        Store a semantic cube in the RDF graph.
        
        Args:
            cube: SemanticCube to store
            validate: Whether to validate against schema
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert semantic cube to RDF triples
            triples = cube.get_rdf_triples(self.namespace)
            
            # Validate if requested
            if validate:
                is_valid, errors = self.validate_schema(triples)
                if not is_valid:
                    logger.warning(f"Schema validation failed for cube {cube.cube_id}: {errors}")
                    # Continue anyway, but log warnings
            
            # Add triples to graph
            for triple in triples:
                self.graph.add(triple)
            
            # Add cube metadata
            cube_uri = URIRef(f"{self.base_uri}cube/{cube.cube_id}")
            self.graph.add((cube_uri, RDF.type, self.namespace.SemanticCube))
            self.graph.add((cube_uri, RDFS.label, Literal(f"Semantic Cube {cube.cube_id}")))
            self.graph.add((cube_uri, self.namespace.hasConfidence, Literal(cube.confidence_score)))
            
            for key, value in cube.metadata.items():
                self.graph.add((cube_uri, URIRef(f"{self.base_uri}metadata/{key}"), Literal(value)))
            
            logger.info(f"Stored semantic cube {cube.cube_id} with {len(triples)} triples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store semantic cube {cube.cube_id}: {e}")
            return False
    
    def query_graph(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query against the knowledge graph.
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            results = self.graph.query(sparql_query)
            return [dict(row.asdict()) for row in results]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []
    
    def find_similar_entities(self, query_text: str, threshold: float = 0.7) -> List[Tuple[SemanticEntity, float]]:
        """
        Find entities similar to query text using semantic similarity.
        
        Args:
            query_text: Text to find similar entities for
            threshold: Minimum similarity threshold
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        if not self.sentence_transformer:
            logger.warning("Sentence transformer not available")
            return []
        
        similar_entities = []
        query_embedding = self.sentence_transformer.encode([query_text])
        
        for cube in self.semantic_cubes.values():
            for entity in cube.entities:
                entity_embedding = self.sentence_transformer.encode([entity.text])
                similarity = np.dot(query_embedding[0], entity_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(entity_embedding[0])
                )
                
                if similarity >= threshold:
                    similar_entities.append((entity, float(similarity)))
        
        # Sort by similarity score (descending)
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return similar_entities
    
    def export_graph(self, format: str = "turtle", file_path: str = None) -> str:
        """
        Export the knowledge graph to various formats.
        
        Args:
            format: Output format ("turtle", "xml", "json-ld", "n3")
            file_path: Optional file path to save to
            
        Returns:
            Serialized graph as string
        """
        try:
            serialized = self.graph.serialize(format=format)
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(serialized)
                logger.info(f"Graph exported to {file_path} in {format} format")
            
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return ""
    
    def import_graph(self, data: str, format: str = "turtle") -> bool:
        """
        Import RDF data into the knowledge graph.
        
        Args:
            data: RDF data as string
            format: Input format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph.parse(data=data, format=format)
            logger.info(f"Successfully imported RDF data in {format} format")
            return True
        except Exception as e:
            logger.error(f"Failed to import RDF data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            "total_triples": len(self.graph),
            "total_cubes": len(self.semantic_cubes),
            "total_entities": sum(len(cube.entities) for cube in self.semantic_cubes.values()),
            "total_relations": sum(len(cube.relations) for cube in self.semantic_cubes.values()),
            "namespaces": list(self.graph.namespaces()),
            "cube_statistics": {
                cube_id: {
                    "entities": len(cube.entities),
                    "relations": len(cube.relations),
                    "confidence": cube.confidence_score
                }
                for cube_id, cube in self.semantic_cubes.items()
            }
        }
        
        return stats


# Example usage and demonstration
def demonstrate_ontological_knowledge_graph():
    """Demonstrate the capabilities of the OntologicalKnowledgeGraph."""
    
    # Initialize the knowledge graph
    kg = OntologicalKnowledgeGraph()
    
    # Example 1: Process unstructured text document
    sample_text = """
    John Smith is a software engineer at Microsoft Corporation in Seattle.
    He specializes in artificial intelligence and machine learning.
    Microsoft is a technology company founded by Bill Gates and Paul Allen.
    Seattle is located in Washington State, United States.
    """
    
    print("Processing unstructured text document...")
    cube1 = kg.process_document(sample_text, "doc_001", {"source": "example", "type": "text"})
    
    # Store the semantic cube in the graph
    kg.store_semantic_cube(cube1)
    
    # Example 2: Process structured data
    structured_data = {
        "person": {
            "name": "John Smith",
            "profession": "Software Engineer",
            "company": "Microsoft",
            "skills": ["AI", "Machine Learning", "Python"],
            "location": {
                "city": "Seattle",
                "state": "Washington",
                "country": "USA"
            }
        }
    }
    
    print("Processing structured data...")
    cube2 = kg.add_structured_data(structured_data, "struct_001")
    kg.store_semantic_cube(cube2)
    
    # Example 3: Query the knowledge graph
    print("Querying the knowledge graph...")
    sparql_query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object .
    }
    LIMIT 10
    """
    
    results = kg.query_graph(sparql_query)
    print(f"Found {len(results)} results from SPARQL query")
    
    # Example 4: Find similar entities
    if kg.sentence_transformer:
        print("Finding similar entities...")
        similar = kg.find_similar_entities("software developer", threshold=0.5)
        print(f"Found {len(similar)} similar entities")
    
    # Example 5: Export graph
    print("Exporting graph...")
    turtle_output = kg.export_graph("turtle")
    print(f"Exported graph ({len(turtle_output)} characters)")
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"Knowledge Graph Statistics: {stats}")
    
    return kg


if __name__ == "__main__":
    # Demonstrate the functionality
    kg = demonstrate_ontological_knowledge_graph()