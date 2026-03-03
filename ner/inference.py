
#!/usr/bin/env python3
"""
NER Inference Script for OpenCHS AI Pipeline
Performs Named Entity Recognition on child helpline conversations.
"""

import json
import os
import argparse
from typing import List, Dict, Tuple, Union
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification


class NERInference:
    """Named Entity Recognition inference class for child helpline conversations."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the NER inference model.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        # Load model components
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.label_mappings = self._load_label_mappings()
        
        print(f"NER model loaded successfully on {self.device}")
        print(f"Available entity types: {list(self.label_mappings['label_to_id'].keys())}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        tokenizer_path = os.path.join(self.model_path, "tokenizer_config.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.model_path}")
        return AutoTokenizer.from_pretrained(self.model_path)
    
    def _load_model(self) -> AutoModelForTokenClassification:
        """Load the trained model."""
        # Check for different model file formats
        model_files = [
            os.path.join(self.model_path, "pytorch_model.bin"),
            os.path.join(self.model_path, "model.safetensors"),
            os.path.join(self.model_path, "config.json")  # At minimum, config should exist
        ]
        
        config_exists = os.path.exists(os.path.join(self.model_path, "config.json"))
        if not config_exists:
            raise FileNotFoundError(f"Model config not found at {self.model_path}")
        
        # Use transformers' automatic model loading (handles both .bin and .safetensors)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_label_mappings(self) -> Dict:
        """Load label mappings."""
        mappings_path = os.path.join(self.model_path, "label_mappings.json")
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Label mappings not found at {mappings_path}")
        
        with open(mappings_path, 'r') as f:
            return json.load(f)
    
    def predict_single(self, text: str, return_confidence: bool = False) -> Union[List[Tuple], List[Dict]]:
        """
        Predict entities for a single text.
        
        Args:
            text: Input text to analyze
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of entities with their labels and positions
        """
        if not text.strip():
            return []
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True
        ).to(self.device)
        
        offset_mapping = inputs.pop('offset_mapping').cpu().numpy()[0]
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Convert to CPU numpy arrays
        predictions = predictions.cpu().numpy()[0]
        predicted_labels = predicted_labels.cpu().numpy()[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].cpu().numpy()[0])
        
        # Extract entities
        entities = self._extract_entities(
            text, tokens, predicted_labels, predictions, 
            offset_mapping, return_confidence
        )
        
        return entities
    
    def predict_batch(self, texts: List[str], return_confidence: bool = False) -> List[List[Union[Tuple, Dict]]]:
        """
        Predict entities for a batch of texts.
        
        Args:
            texts: List of input texts
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of entity lists for each input text
        """
        results = []
        for text in texts:
            entities = self.predict_single(text, return_confidence)
            results.append(entities)
        return results
    
    def _extract_entities(self, original_text: str, tokens: List[str], 
                         predicted_labels: np.ndarray, predictions: np.ndarray,
                         offset_mapping: np.ndarray, return_confidence: bool) -> List[Union[Tuple, Dict]]:
        """Extract entities from model predictions."""
        id_to_label = self.label_mappings['id_to_label']
        entities = []
        current_entity = None
        
        for i, (token, label_id) in enumerate(zip(tokens, predicted_labels)):
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            label = id_to_label[str(label_id)]
            confidence = float(np.max(predictions[i]))
            
            # Skip 'O' (Outside) labels
            if label == 'O':
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, original_text, return_confidence))
                    current_entity = None
                continue
            
            # Get token position in original text
            if i < len(offset_mapping):
                start, end = offset_mapping[i]
                
                # Handle subword tokens (starting with ##)
                if token.startswith('##'):
                    if current_entity and current_entity['label'] == label:
                        # Extend current entity
                        current_entity['end'] = int(end)
                        current_entity['tokens'].append(token[2:])  # Remove ##
                        current_entity['confidences'].append(confidence)
                    continue
                
                # Start new entity or continue existing one
                if current_entity and current_entity['label'] == label:
                    # Extend current entity if same label
                    current_entity['end'] = int(end)
                    current_entity['tokens'].append(token)
                    current_entity['confidences'].append(confidence)
                else:
                    # Finalize previous entity if exists
                    if current_entity:
                        entities.append(self._finalize_entity(current_entity, original_text, return_confidence))
                    
                    # Start new entity
                    current_entity = {
                        'label': label,
                        'start': int(start),
                        'end': int(end),
                        'tokens': [token],
                        'confidences': [confidence]
                    }
        
        # Finalize last entity if exists
        if current_entity:
            entities.append(self._finalize_entity(current_entity, original_text, return_confidence))
        
        return entities
    
    def _finalize_entity(self, entity_data: Dict, original_text: str, return_confidence: bool) -> Union[Tuple, Dict]:
        """Convert entity data to final format."""
        # Extract text from original string using positions
        text = original_text[entity_data['start']:entity_data['end']].strip()
        
        # Calculate average confidence
        avg_confidence = np.mean(entity_data['confidences'])
        
        if return_confidence:
            return {
                'text': text,
                'label': entity_data['label'],
                'start': entity_data['start'],
                'end': entity_data['end'],
                'confidence': float(avg_confidence)
            }
        else:
            return (text, entity_data['label'])
    
    def analyze_conversation(self, conversation: str) -> Dict:
        """
        Analyze a complete conversation and provide structured output.
        
        Args:
            conversation: Full conversation text
            
        Returns:
            Dictionary with conversation analysis
        """
        entities = self.predict_single(conversation, return_confidence=True)
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            label = entity['label']
            if label not in entity_groups:
                entity_groups[label] = []
            entity_groups[label].append(entity)
        
        # Calculate statistics
        stats = {
            'total_entities': len(entities),
            'entity_types': len(entity_groups),
            'entities_by_type': {label: len(ents) for label, ents in entity_groups.items()}
        }
        
        return {
            'entities': entities,
            'entity_groups': entity_groups,
            'statistics': stats,
            'conversation_length': len(conversation),
            'conversation_preview': conversation[:200] + "..." if len(conversation) > 200 else conversation
        }


def main():
    """Command line interface for NER inference."""
    parser = argparse.ArgumentParser(description="Named Entity Recognition Inference for Child Helplines")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model directory")
    parser.add_argument("--text", "-t", help="Single text to analyze")
    parser.add_argument("--file", "-f", help="File containing texts to analyze (one per line)")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--confidence", "-c", action="store_true", help="Include confidence scores")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to use for inference")
    parser.add_argument("--format", choices=["simple", "detailed", "conversation"], default="simple",
                       help="Output format")
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Initialize model
    try:
        ner_model = NERInference(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    results = []
    
    # Process single text
    if args.text:
        if args.format == "conversation":
            result = ner_model.analyze_conversation(args.text)
        else:
            entities = ner_model.predict_single(args.text, args.confidence)
            result = {"text": args.text, "entities": entities}
        results.append(result)
    
    # Process file
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(texts)} texts from {args.file}")
            
            for i, text in enumerate(texts):
                if args.format == "conversation":
                    result = ner_model.analyze_conversation(text)
                else:
                    entities = ner_model.predict_single(text, args.confidence)
                    result = {"text": text[:100] + "..." if len(text) > 100 else text, 
                             "entities": entities}
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(texts)} texts")
        
        except FileNotFoundError:
            print(f"File not found: {args.file}")
            return
        except Exception as e:
            print(f"Error processing file: {e}")
            return
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        # Print to console
        for i, result in enumerate(results):
            if args.format == "simple":
                print(f"\nText {i+1}: {result['text']}")
                print("Entities:")
                for entity in result['entities']:
                    if args.confidence:
                        print(f"  - {entity['text']} [{entity['label']}] (confidence: {entity['confidence']:.3f})")
                    else:
                        print(f"  - {entity[0]} [{entity[1]}]")
            
            elif args.format == "detailed":
                print(json.dumps(result, indent=2, ensure_ascii=False))
            
            elif args.format == "conversation":
                print(f"\n{'='*50}")
                print(f"CONVERSATION ANALYSIS {i+1}")
                print(f"{'='*50}")
                print(f"Length: {result['conversation_length']} characters")
                print(f"Total entities: {result['statistics']['total_entities']}")
                print(f"Entity types: {result['statistics']['entity_types']}")
                print("\nEntities by type:")
                for entity_type, count in result['statistics']['entities_by_type'].items():
                    print(f"  {entity_type}: {count}")
                print(f"\nPreview: {result['conversation_preview']}")


if __name__ == "__main__":
    main()
