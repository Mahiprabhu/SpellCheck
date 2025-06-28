import os
from dataclasses import dataclass

@dataclass
class DomainConfig:
    terms_path: str
    ontology_path: str
    description: str

class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.DATABASE_URL = os.path.join(self.BASE_DIR, 'spellcheck.db')
        
        # Dictionary paths
        self.ENGLISH_DICT_PATH = os.path.join(self.DATA_DIR, 'words_alpha.txt')
        self.PHYSICS_TERMS_PATH = os.path.join(self.DATA_DIR, 'physics_terms.txt')
        self.MEDICAL_TERMS_PATH = os.path.join(self.DATA_DIR, 'medical_terms.txt')
        self.LEGAL_TERMS_PATH = os.path.join(self.DATA_DIR, 'legal_terms.txt')
        
        # Verify dictionary files exist
        if not os.path.exists(self.ENGLISH_DICT_PATH):
            raise FileNotFoundError(f"English dictionary not found at {self.ENGLISH_DICT_PATH}")
        if not os.path.exists(self.PHYSICS_TERMS_PATH):
            print(f"Warning: Physics terms not found at {self.PHYSICS_TERMS_PATH}")
        
        # Domain configurations
        self.DOMAINS = {
            'physics': DomainConfig(
                terms_path=self.PHYSICS_TERMS_PATH,
                ontology_path=os.path.join(self.DATA_DIR, 'physics_ontology.ttl'),
                description='Physics terminology checker'
            ),
            'medical': DomainConfig(
                terms_path=self.MEDICAL_TERMS_PATH,
                ontology_path=os.path.join(self.DATA_DIR, 'medical_ontology.ttl'),
                description='Medical terminology checker'
            ),
            'legal': DomainConfig(
                terms_path=self.LEGAL_TERMS_PATH,
                ontology_path=os.path.join(self.DATA_DIR, 'legal_ontology.ttl'),
                description='Legal terminology checker'
            )
        }
        
        # Model configurations
        self.MODELS = {
            'bert': {
                'name': 'BERT',
                'description': 'Transformer-based contextual model'
            },
            'lstm': {
                'name': 'LSTM',
                'description': 'Long Short-Term Memory neural network'
            },
            'ensemble': {
                'name': 'BERT+LSTM Ensemble',
                'description': 'Combines BERT and LSTM predictions'
            }
        }