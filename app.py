from flask import Flask, render_template, request, jsonify
from collections import OrderedDict
from transformers import BertTokenizer
import torch
import re
import sqlite3
from rdflib import Graph
import os
from config import Config
from flask_cors import CORS
import numpy as np
from torch import nn
import torch.optim as optim
from werkzeug.utils import secure_filename
import docx
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
import json
from difflib import get_close_matches
from torch.utils.data import Dataset, DataLoader
from models import PhysicsSpellcheckLSTM, SpellcheckCNN, load_pretrained_bert

app = Flask(__name__)
CORS(app)
config = Config()

# --- Initialization --- #
print("Initializing spellchecker system...")

# Check/create data directory
if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)
    print(f"Created data directory at {config.DATA_DIR}")

# Initialize database if not exists
if not os.path.exists(config.DATABASE_URL):
    from init_db import init_database
    init_database()
    print("Initialized database")

# Load domain configuration
DOMAIN_CONFIG = {}
try:
    with open(os.path.join(config.DATA_DIR, 'domain_config.json'), 'r') as f:
        DOMAIN_CONFIG = json.load(f)
    print("Loaded domain configuration")
except FileNotFoundError:
    print("Warning: domain_config.json not found - using default configuration")
    DOMAIN_CONFIG = {
        "ontologies": {
            "physics": "physics_ontology.ttl",
            "medical": "medical_ontology.ttl",
            "legal": "legal_ontology.ttl"
        },
        "domains": {
            "physics": {
                "dictionary": "physics_terms.txt",
                "common_misspellings": {
                    "quantan": "quantum",
                    "relativty": "relativity",
                    "physiks": "physics",
                    "fotan": "photon",
                    "particl": "particle",
                    "lite": "light"
                }
            }
        }
    }

# Load dictionaries
DICTIONARIES = {}
COMMON_MISSPELLINGS = {}

try:
    # English dictionary
    with open(config.ENGLISH_DICT_PATH, 'r', encoding='utf-8') as f:
        DICTIONARIES['english'] = {line.strip().lower() for line in f if line.strip()}
    
    # Domain dictionaries
    for domain in ['physics', 'medical', 'legal']:
        dict_path = os.path.join(config.DATA_DIR, DOMAIN_CONFIG['domains'].get(domain, {}).get('dictionary', f"{domain}_terms.txt"))
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                DICTIONARIES[domain] = {line.strip().lower() for line in f if line.strip()}
    
    # Common misspellings
    COMMON_MISSPELLINGS = DOMAIN_CONFIG['domains'].get('physics', {}).get('common_misspellings', {})
    
    print(f"Loaded dictionaries: English ({len(DICTIONARIES['english'])}), Physics ({len(DICTIONARIES.get('physics', set()))})")
except Exception as e:
    print(f"Error loading dictionaries: {str(e)}")
    DICTIONARIES = {
        'english': set(),
        'physics': set(),
        'medical': set(),
        'legal': set()
    }
    COMMON_MISSPELLINGS = {}

# Load physics terms and initialize vocabulary
# def load_physics_terms():
#     with open(os.path.join(config.DATA_DIR, 'physics_terms.txt'), 'r') as f:
#         terms = [line.strip().lower() for line in f if line.strip()]
#     return ['<PAD>', '<UNK>'] + terms

def load_physics_terms():
    with open(os.path.join(config.DATA_DIR, 'physics_terms.txt'), 'r') as f:
        # Remove empty lines and duplicates
        terms = list(OrderedDict.fromkeys(
            line.strip().lower() for line in f if line.strip()
        ))
    return ['<PAD>', '<UNK>'] + terms  # Total should now match trained model

physics_terms = load_physics_terms()
vocab = {word: idx for idx, word in enumerate(physics_terms)}
reverse_vocab = {idx: word for word, idx in vocab.items()}
vocab_size = len(physics_terms)

print(f"Vocabulary size: {vocab_size}")

# --- Model Initialization --- #
print("\nInitializing models...")

# BERT Model
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = load_pretrained_bert()
    
    bert_finetuned_path = os.path.join(config.DATA_DIR, 'bert_finetuned_physics.pth')
    if os.path.exists(bert_finetuned_path):
        bert_model.load_state_dict(torch.load(bert_finetuned_path))
        print("Loaded fine-tuned BERT model")
    else:
        print("Using base BERT model (not fine-tuned)")
except Exception as e:
    print(f"Error initializing BERT: {str(e)}")
    raise

# LSTM Model
try:
    lstm_model = PhysicsSpellcheckLSTM(vocab_size)
    lstm_weights_path = os.path.join(config.DATA_DIR, 'lstm_physics.pth')
    
    if os.path.exists(lstm_weights_path):
        saved_data = torch.load(lstm_weights_path)
        lstm_model.load_state_dict(saved_data['model_state_dict'])
        print("Loaded trained LSTM model")
    else:
        for param in lstm_model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        print("Initialized LSTM with random weights (not trained)")
except Exception as e:
    print(f"Error initializing LSTM: {str(e)}")
    raise

# CNN Model
try:
    cnn_weights_path = os.path.join(config.DATA_DIR, 'cnn_physics.pth')
    if os.path.exists(cnn_weights_path):
        saved_data = torch.load(cnn_weights_path)
        
        # Initialize character vocabulary
        chars = list("abcdefghijklmnopqrstuvwxyz'-")
        cnn_char_vocab = {c: i+2 for i, c in enumerate(chars)}
        cnn_char_vocab['<PAD>'] = 0
        cnn_char_vocab['<UNK>'] = 1
        
        cnn_model = SpellcheckCNN(
            vocab_size=vocab_size,
            char_vocab_size=len(cnn_char_vocab)
        )
        cnn_model.load_state_dict(saved_data['model_state_dict'])
        cnn_model.eval()
        print("Loaded trained CNN model")
    else:
        print("CNN model weights not found")
        cnn_model = None
except Exception as e:
    print(f"Error initializing CNN: {str(e)}")
    cnn_model = None

# --- Core Functions --- #
def get_bert_suggestions(sentence: str, word: str, domain: str, top_k: int = 5) -> List[Dict]:
    """Get contextual suggestions from BERT"""
    try:
        masked = re.sub(r'\b' + re.escape(word) + r'\b', '[MASK]', sentence, count=1)
        inputs = tokenizer(masked, return_tensors='pt')
        
        mask_positions = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if not mask_positions:
            return []
            
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        suggestions = []
        for mask_pos in mask_positions:
            logits = outputs.logits[0, mask_pos.item()]
            probs = torch.softmax(logits, dim=-1)
            top_k_tokens = torch.topk(probs, top_k*3)
            
            for token_id, score in zip(top_k_tokens.indices, top_k_tokens.values):
                suggestion = tokenizer.decode([token_id]).strip()
                lower_sugg = suggestion.lower()
                lower_word = word.lower()
                
                if (not suggestion or len(suggestion) <= 1 or not suggestion.isalpha() or lower_sugg == lower_word):
                    continue
                
                score_val = score.item() * 100
                if domain != 'general' and lower_sugg in DICTIONARIES.get(domain, set()):
                    score_val = min(100, score_val * 1.5)
                if lower_word in COMMON_MISSPELLINGS and lower_sugg == COMMON_MISSPELLINGS[lower_word]:
                    score_val = min(100, score_val * 2.0)
                
                if word[0].isupper():
                    suggestion = suggestion.capitalize()
                
                suggestions.append({
                    'token': suggestion,
                    'score': min(100, int(score_val)),
                    'source': 'bert'
                })
        
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)[:top_k]
    except Exception as e:
        print(f"BERT suggestion error: {str(e)}")
        return []

def get_lstm_suggestions(word: str, domain: str, top_k: int = 5) -> List[Dict]:
    """Get suggestions from LSTM based on similarity"""
    try:
        word_lower = word.lower()
        matches = get_close_matches(word_lower, list(vocab.keys()), n=top_k*2, cutoff=0.6)
        
        suggestions = []
        for match in matches:
            if match == word_lower:
                continue
                
            score = int(100 * (1 - (len(match) - len(word_lower))/max(len(match), len(word_lower))))
            if domain != 'general' and match in DICTIONARIES.get(domain, set()):
                score = min(100, score + 20)
            
            token = match.capitalize() if word[0].isupper() else match
            suggestions.append({
                'token': token,
                'score': score,
                'source': 'lstm'
            })
        
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)[:top_k]
    except Exception as e:
        print(f"LSTM suggestion error: {str(e)}")
        return []

def get_cnn_suggestions(word: str, domain: str, top_k: int = 5) -> List[Dict]:
    """Get suggestions from CNN based on character patterns"""
    if not cnn_model:
        return []
    
    try:
        # Convert word to character IDs
        char_ids = []
        for c in word.lower():
            char_id = cnn_char_vocab.get(c, cnn_char_vocab['<UNK>'])
            char_ids.append(char_id)
        
        # Pad/truncate to 16 characters
        char_ids = char_ids[:16] + [0] * (16 - len(char_ids))
        input_tensor = torch.tensor([char_ids], dtype=torch.long)

        with torch.no_grad():
            logits = cnn_model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k*5)
        
        suggestions = []
        for i in range(top_k*5):
            token_id = top_indices[0][i].item()
            score = top_probs[0][i].item() * 100
            suggestion = reverse_vocab.get(token_id, "")
            
            if not suggestion or suggestion in ['<PAD>', '<UNK>'] or suggestion == word.lower():
                continue
                
            if abs(len(suggestion) - len(word)) > 2:
                continue
                
            if domain != 'general' and suggestion in DICTIONARIES.get(domain, set()):
                score = min(100, score * 1.5)
                
            if word[0].isupper():
                suggestion = suggestion.capitalize()
            
            if score >= 20:
                suggestions.append({
                    'token': suggestion,
                    'score': int(score),
                    'source': 'cnn'
                })

        if not suggestions and top_k*5 > 0:
            for i in range(top_k*5):
                token_id = top_indices[0][i].item()
                suggestion = reverse_vocab.get(token_id, "")
                if suggestion and suggestion not in ['<PAD>', '<UNK>']:
                    suggestions.append({
                        'token': suggestion.capitalize() if word[0].isupper() else suggestion,
                        'score': 20,
                        'source': 'cnn'
                    })
                    break

        return suggestions[:top_k]
    except Exception as e:
        print(f"CNN suggestion error: {str(e)}")
        return []

def enhanced_ensemble_suggestions(bert_sugs, lstm_sugs, cnn_sugs, domain: str) -> List[Dict]:
    """Combine suggestions from all models"""
    combined = defaultdict(dict)
    
    for model_name, suggestions in [('bert', bert_sugs), ('lstm', lstm_sugs), ('cnn', cnn_sugs)]:
        if not suggestions:
            continue
            
        for sugg in suggestions:
            token = sugg['token'].lower()
            if token not in combined:
                combined[token] = {
                    'token': sugg['token'],
                    'scores': {'bert': 0, 'lstm': 0, 'cnn': 0},
                    'sources': []
                }
            
            weight = 0.5 if model_name == 'bert' else 0.3 if model_name == 'lstm' else 0.2
            combined[token]['scores'][model_name] = sugg['score'] * weight
            combined[token]['sources'].append(model_name)
    
    results = []
    for token, data in combined.items():
        total_score = sum(data['scores'].values())
        if domain != 'general' and token in DICTIONARIES.get(domain, set()):
            total_score = min(100, total_score * 1.2)
        
        results.append({
            'token': data['token'],
            'score': int(total_score),
            'sources': data['sources']
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

def get_ontology_links(term: str, domain: str) -> List[Dict]:
    """Get ontology relationships for a term"""
    if domain not in DOMAIN_CONFIG.get('ontologies', {}):
        return []
    
    ontology_file = os.path.join(config.DATA_DIR, DOMAIN_CONFIG['ontologies'][domain])
    if not os.path.exists(ontology_file):
        return []
    
    try:
        g = Graph()
        g.parse(ontology_file, format='turtle')
        
        results = []
        query = f"""
        SELECT ?rel ?other WHERE {{
            ?term rdfs:label "{term.lower()}"@en .
            ?term ?rel ?other .
            FILTER(STRSTARTS(STR(?rel), "http://example.com/{domain}#"))
        }}
        """
        
        for row in g.query(query):
            results.append({
                'relation': str(row.rel).split('#')[-1],
                'term': str(row.other).split('#')[-1]
            })
        
        return results
    except Exception as e:
        print(f"Ontology query error: {str(e)}")
        return []
    
def spellcheck_text(text: str, domain: str = 'physics', model_type: str = 'ensemble') -> Dict:
    results = {
        'general_errors': [],
        'domain_errors': [],
        'warnings': []
    }
    
    if not text.strip():
        return results
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        words = re.findall(r"\b[\w']+\b", sentence)
        
        for word in words:
            if len(word) <= 2 or word.replace("'", "").isdigit():
                continue
                
            lower_word = word.lower()
            in_english = lower_word in DICTIONARIES['english']
            in_domain = domain != 'general' and lower_word in DICTIONARIES.get(domain, set())
            
            if in_english and (domain == 'general' or in_domain):
                continue
                
            suggestions = []
            if model_type == 'bert':
                suggestions = get_bert_suggestions(sentence, word, domain)
            elif model_type == 'lstm':
                suggestions = get_lstm_suggestions(word, domain)
            elif model_type == 'cnn':
                suggestions = get_cnn_suggestions(word, domain)
            else:
                bert_sugs = get_bert_suggestions(sentence, word, domain)
                lstm_sugs = get_lstm_suggestions(word, domain)
                cnn_sugs = get_cnn_suggestions(word, domain)
                suggestions = enhanced_ensemble_suggestions(bert_sugs, lstm_sugs, cnn_sugs, domain)
            
            if suggestions:
                best = suggestions[0]
                error_type = 'general'
                
                if in_english and domain != 'general' and not in_domain:
                    if best['token'].lower() in DICTIONARIES.get(domain, set()):
                        error_type = 'domain'
                
                error_entry = {
                    'original': word,
                    'correction': best['token'],
                    'confidence': best['score'],
                    'context': sentence,
                    'type': error_type,
                    'model': model_type
                }
                
                if error_type == 'general':
                    results['general_errors'].append(error_entry)
                else:
                    if domain != 'general':
                        error_entry['ontology'] = get_ontology_links(best['token'], domain)
                    results['domain_errors'].append(error_entry)
            elif not in_english:
                results['warnings'].append({
                    'word': word,
                    'message': 'No suggestions available',
                    'context': sentence
                })
    
    return results


def read_docx(file_path: str) -> str:
    """Read text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return ""

# --- API Endpoints --- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/spellcheck', methods=['POST'])
def api_spellcheck():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid request'}), 400
            
        text = data['text']
        domain = data.get('domain', 'physics')
        model_type = data.get('model', 'ensemble')
        
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
            
        result = spellcheck_text(text, domain, model_type)
        return jsonify(result)
        
    except Exception as e:
        print(f"Spellcheck error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/spellcheck-file', methods=['POST'])
def api_spellcheck_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        domain = request.form.get('domain', 'physics')
        model_type = request.form.get('model', 'ensemble')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif file.filename.endswith('.docx'):
            temp_path = os.path.join('/tmp', secure_filename(file.filename))
            file.save(temp_path)
            text = read_docx(temp_path)
            os.remove(temp_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
            
        if not text.strip():
            return jsonify({'error': 'File is empty'}), 400
            
        result = spellcheck_text(text, domain, model_type)
        return jsonify(result)
        
    except Exception as e:
        print(f"File processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Modify the existing api_feedback endpoint
@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    try:
        data = request.json
        if not data or 'original' not in data:
            return jsonify({'error': 'Invalid feedback data'}), 400
        
        # Log feedback to SQLite
        conn = sqlite3.connect(config.DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO feedback 
                      (original, correction, user_correction, is_correct, domain, model_type) 
                      VALUES (?, ?, ?, ?, ?, ?)''',
                    (data['original'], 
                     data.get('correction'),
                     data.get('user_correction'),
                     data.get('is_correct', False),
                     data.get('domain', 'general'),
                     data.get('model_type', 'bert')))
        
        # Continual learning trigger (simplified)
        if data.get('is_correct') is False and data.get('user_correction'):
            update_misspellings(data['original'], data['user_correction'])
        
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        return jsonify({'error': 'Failed to save feedback'}), 500

def update_misspellings(wrong: str, correct: str):
    """Update common misspellings dynamically"""
    try:
        COMMON_MISSPELLINGS[wrong.lower()] = correct.lower()
        with open(os.path.join(config.DATA_DIR, 'common_misspellings.json'), 'w') as f:
            json.dump(COMMON_MISSPELLINGS, f)
        print(f"Updated misspellings: {wrong} → {correct}")
    except Exception as e:
        print(f"Failed to update misspellings: {str(e)}")

@app.route('/api/debug/dictionaries')
def debug_dictionaries():
    return jsonify({
        'english_size': len(DICTIONARIES['english']),
        'physics_size': len(DICTIONARIES.get('physics', set())),
        'common_misspellings': COMMON_MISSPELLINGS
    })

@app.route('/api/ontology/<term>', methods=['GET'])
def get_ontology(term):
    domain = request.args.get('domain', 'physics')
    links = get_ontology_links(term, domain)
    return jsonify({
        'term': term,
        'relations': links,
        'visualization': generate_ontology_graph(term, links) 
    })

def generate_ontology_graph(root_term: str, relations: List[Dict]) -> Dict:
    """Format for frontend visualization"""
    nodes = {root_term: {'id': 1, 'label': root_term, 'group': 1}}
    links = []
    
    for i, rel in enumerate(relations, start=2):
        nodes[rel['term']] = {'id': i, 'label': rel['term'], 'group': 2}
        links.append({
            'source': 1,
            'target': i,
            'label': rel['relation']
        })
    
    return {'nodes': list(nodes.values()), 'links': links}


print("relativity in physics terms:", "relativity" in DICTIONARIES['physics'])
print("farad in physics terms:", "farad" in DICTIONARIES['physics'])

if __name__ == '__main__':
    # Run test cases
    test_cases = [
        ("quantan mechanics", "physics"),
        ("relativty theory", "physics"),
        ("physiks is hard", "physics"),
        ("fotan particle", "physics"),
        ("normal english words", "general")
    ]
    
    for text, domain in test_cases:
        print(f"\nTesting: '{text}' (domain: {domain})")
        result = spellcheck_text(text, domain)
        print("General errors:", len(result['general_errors']))
        print("Domain errors:", len(result['domain_errors']))
        for error in result['general_errors'] + result['domain_errors']:
            print(f"{error['original']} → {error['correction']} [{error['confidence']}%]")
    
    def test_cnn_model():
        if not cnn_model:
            print("[DEBUG] CNN model not available for testing")
            return
        
        print("\n[DEBUG] Running CNN model test cases...")
        test_words = [
            "quantan",    # Should suggest "quantum"
            "relativty",  # Should suggest "relativity"
            "physiks",    # Should suggest "physics"
            "fotan",      # Should suggest "photon"
            "lite",       # Should suggest "light"
            "particl"     # Should suggest "particle"
        ]
        
        for word in test_words:
            print(f"\n[DEBUG] Testing CNN with '{word}':")
            suggestions = get_cnn_suggestions(word, "physics")
            if suggestions:
                print(f"[DEBUG] Suggestions for '{word}':")
                for sugg in suggestions:
                    print(f"  → {sugg['token']} ({sugg['score']}%)")
            else:
                print(f"[DEBUG] No suggestions for '{word}'")

    # Call the test function when the server starts
    test_cnn_model()
    app.run(debug=True)