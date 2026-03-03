
#!/usr/bin/env python3
"""
Usage examples for NER inference script.
Examples specific to child helpline conversations.
"""

from inference import NERInference
import json

def example_basic_usage():
    """Basic usage example."""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # Initialize the model
    model_path = "./ner-distilbert-en-synthetic-v2"  # Update with your model path
    ner_model = NERInference(model_path)
    
    # Example conversation from child helpline
    conversation = """
    Hello, is this the child helpline? Yes, this is Kimani. How can I assist you today? 
    I'm Peter Rogendo from Mikindani, Mombasa. I have a concern about my 20 year-old daughter, 
    Kamili. She's been having difficulties at school with her academics. She goes to Technical University of Mombasa
    """
    
    # Simple prediction
    entities = ner_model.predict_single(conversation)
    
    print("Detected entities:")
    for entity_text, entity_label in entities:
        print(f"  {entity_text} -> {entity_label}")


def example_with_confidence():
    """Example with confidence scores."""
    print("\n=== CONFIDENCE SCORES EXAMPLE ===")
    
    model_path = "./ner-distilbert-en-synthetic-v2"
    ner_model = NERInference(model_path)
    
    text = "Hi, I'm Amara from Dar es Salaam. I'm 15 years old and need help."
    
    entities = ner_model.predict_single(text, return_confidence=True)
    
    print("Entities with confidence:")
    for entity in entities:
        print(f"  {entity['text']} -> {entity['label']} (confidence: {entity['confidence']:.3f})")


def example_batch_processing():
    """Example of batch processing multiple conversations."""
    print("\n=== BATCH PROCESSING EXAMPLE ===")
    
    model_path = "./ner-distilbert-en-synthetic-v2"
    ner_model = NERInference(model_path)
    
    conversations = [
        "Hello, I'm Sarah from Nairobi. My son David is 12 years old.",
        "Hi, this is counselor John. How can I help you today?",
        "I'm calling from Mombasa about a 10-year-old girl named Grace."
    ]
    
    batch_results = ner_model.predict_batch(conversations)
    
    for i, (conversation, entities) in enumerate(zip(conversations, batch_results)):
        print(f"\nConversation {i+1}: {conversation[:50]}...")
        print("Entities:")
        for entity_text, entity_label in entities:
            print(f"  {entity_text} -> {entity_label}")


def example_conversation_analysis():
    """Example of full conversation analysis."""
    print("\n=== CONVERSATION ANALYSIS EXAMPLE ===")
    
    model_path = "./ner-distilbert-en-synthetic-v2"
    ner_model = NERInference(model_path)
    
    full_conversation = """
    Hello, is this the child helpline? Yes, this is Asubuhi, I'm here to help. 
    I'm Modesta from Mafinga, Iringa. I'm 11 years old and I need someone to talk to. 
    Lately, I've been feeling really sad and overwhelmed with everything at home and school. 
    My parents are constantly arguing and it makes me feel anxious all the time. 
    At school, I struggle to keep up with my classmates in class, which makes me feel 
    stupid and worthless. The teachers don't seem to notice or care.
    """
    
    analysis = ner_model.analyze_conversation(full_conversation)
    
    print("Conversation Analysis:")
    print(f"Total entities: {analysis['statistics']['total_entities']}")
    print(f"Entity types found: {analysis['statistics']['entity_types']}")
    print("\nBreakdown by entity type:")
    for entity_type, count in analysis['statistics']['entities_by_type'].items():
        print(f"  {entity_type}: {count}")
    
    print(f"\nConversation length: {analysis['conversation_length']} characters")


def example_pipeline_integration():
    """Example of integrating NER into OpenCHS AI pipeline."""
    print("\n=== PIPELINE INTEGRATION EXAMPLE ===")
    
    class OpenCHSPipelineComponent:
        def __init__(self, model_path):
            self.ner_model = NERInference(model_path)
        
        def process_helpline_call(self, conversation_text: str) -> dict:
            """Process a helpline call and extract structured information."""
            
            # Get NER analysis
            analysis = self.ner_model.analyze_conversation(conversation_text)
            
            # Extract key information for case management
            extracted_info = {
                'caller_info': self._extract_caller_info(analysis['entity_groups']),
                'victim_info': self._extract_victim_info(analysis['entity_groups']),
                'location_info': self._extract_location_info(analysis['entity_groups']),
                'incident_type': self._extract_incident_type(analysis['entity_groups']),
                'entities_summary': analysis['statistics']
            }
            
            return extracted_info
        
        def _extract_caller_info(self, entity_groups):
            """Extract caller information."""
            callers = entity_groups.get('CALLER', [])
            ages = entity_groups.get('AGE', [])
            
            return {
                'names': [e['text'] for e in callers],
                'ages': [e['text'] for e in ages],
                'count': len(callers)
            }
        
        def _extract_victim_info(self, entity_groups):
            """Extract victim information."""
            victims = entity_groups.get('VICTIM', [])
            return {
                'names': [e['text'] for e in victims],
                'count': len(victims)
            }
        
        def _extract_location_info(self, entity_groups):
            """Extract location information."""
            locations = entity_groups.get('LOCATION', [])
            return {
                'locations': [e['text'] for e in locations],
                'count': len(locations)
            }
        
        def _extract_incident_type(self, entity_groups):
            """Extract incident type information."""
            incidents = entity_groups.get('INCIDENT_TYPE', [])
            return {
                'types': [e['text'] for e in incidents],
                'count': len(incidents)
            }
    
    # Demo usage
    model_path = "./ner-distilbert-en-synthetic-v2"
    pipeline = OpenCHSPipelineComponent(model_path)
    
    sample_call = """
    Hi, I'm Sarah from Nairobi. I'm calling about my 12-year-old daughter Emma. 
    She's been experiencing bullying at school and it's affecting her mental health.
    """
    
    result = pipeline.process_helpline_call(sample_call)
    
    print("Structured case information:")
    print(json.dumps(result, indent=2))


def example_quality_control():
    """Example of using NER for quality control of transcriptions."""
    print("\n=== QUALITY CONTROL EXAMPLE ===")
    
    model_path = "./ner-distilbert-en-synthetic-v2"
    ner_model = NERInference(model_path)
    
    def check_transcription_quality(transcription: str) -> dict:
        """Check if transcription contains expected entities for quality control."""
        entities = ner_model.predict_single(transcription, return_confidence=True)
        
        # Define expected entities for a typical child helpline call
        expected_entity_types = ['CALLER', 'COUNSELOR', 'AGE', 'LOCATION']
        found_entity_types = set([e['label'] for e in entities])
        
        quality_score = len(found_entity_types.intersection(expected_entity_types)) / len(expected_entity_types)
        
        return {
            'quality_score': quality_score,
            'found_entities': len(entities),
            'found_entity_types': list(found_entity_types),
            'missing_expected_types': list(set(expected_entity_types) - found_entity_types),
            'low_confidence_entities': [e for e in entities if e['confidence'] < 0.7]
        }
    
    # Test with a good transcription
    good_transcription = "Hi, I'm Mary from Lagos. I'm 25 and calling about my 8-year-old son."
    quality_check = check_transcription_quality(good_transcription)
    
    print("Quality check results:")
    print(f"Quality score: {quality_check['quality_score']:.2f}")
    print(f"Found entities: {quality_check['found_entities']}")
    print(f"Missing expected types: {quality_check['missing_expected_types']}")


if __name__ == "__main__":
    print("NER Inference Usage Examples for OpenCHS AI Pipeline")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_with_confidence()
        example_batch_processing()
        example_conversation_analysis()
        example_pipeline_integration()
        example_quality_control()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure your trained model is available at the specified path.")
        print("Update the model_path variable in the examples to match your model location.")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Make sure you have all required dependencies installed.")
