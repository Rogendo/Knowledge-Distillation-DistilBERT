"""
FastAPI service for OpenCHS Multi-Task Model
Usage:
    python api_service.py
    # API runs on http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, DistilBertPreTrainedModel, DistilBertModel
import torch
import torch.nn as nn
import numpy as np
import json
import yaml
import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenCHS Multi-Task Classifier API",
    description="Classification API for child helpline case management using multi-task DistilBERT",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
device = None
categories = None
config = None


# ===================== MODEL DEFINITION =====================
class MultiTaskDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config, num_main, num_sub, num_interv, num_priority):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier_main = nn.Linear(config.dim, num_main)
        self.classifier_sub = nn.Linear(config.dim, num_sub)
        self.classifier_interv = nn.Linear(config.dim, num_interv)
        self.classifier_priority = nn.Linear(config.dim, num_priority)
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, 
                main_category_id=None, sub_category_id=None, 
                intervention_id=None, priority_id=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_state = distilbert_output.last_hidden_state 
        pooled_output = hidden_state[:, 0]                 
        pooled_output = self.pre_classifier(pooled_output) 
        pooled_output = nn.ReLU()(pooled_output)           
        pooled_output = self.dropout(pooled_output)        
        
        logits_main = self.classifier_main(pooled_output)
        logits_sub = self.classifier_sub(pooled_output)
        logits_interv = self.classifier_interv(pooled_output)
        logits_priority = self.classifier_priority(pooled_output)
        
        return (logits_main, logits_sub, logits_interv, logits_priority)


# ===================== PYDANTIC MODELS =====================
class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Case narrative to classify", min_length=1)
    return_probabilities: bool = Field(False, description="Return full probability distributions")

    class Config:
        schema_extra = {
            "example": {
                "text": "A 14-year-old girl is being bullied at school by her classmates",
                "return_probabilities": False
            }
        }


class PredictionResult(BaseModel):
    main_category: str = Field(..., description="Main category classification")
    sub_category: str = Field(..., description="Sub-category classification")
    intervention: str = Field(..., description="Recommended intervention")
    priority: int = Field(..., description="Priority level (1-3)")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each prediction")
    probabilities: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Full probability distributions")


class ClassificationResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    processing_time_ms: Optional[float] = None


class BatchClassificationRequest(BaseModel):
    texts: List[str] = Field(..., description="List of case narratives", min_items=1, max_items=100)
    return_probabilities: bool = Field(False, description="Return full probability distributions")


class BatchClassificationResponse(BaseModel):
    success: bool
    results: List[PredictionResult]
    total_count: int
    successful_count: int
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    categories_loaded: bool
    config_loaded: bool


# ===================== HELPER FUNCTIONS =====================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_categories(config):
    """Load all category mappings from JSON files"""
    try:
        labels_dir = config['output']['labels_dir']
        
        with open(os.path.join(labels_dir, config['output']['main_categories_file']), 'r') as f:
            main_categories = json.load(f)
        
        with open(os.path.join(labels_dir, config['output']['sub_categories_file']), 'r') as f:
            sub_categories = json.load(f)
        
        with open(os.path.join(labels_dir, config['output']['interventions_file']), 'r') as f:
            interventions = json.load(f)
        
        with open(os.path.join(labels_dir, config['output']['priorities_file']), 'r') as f:
            priorities = json.load(f)
        
        logger.info(f"✅ Loaded categories - Main: {len(main_categories)}, Sub: {len(sub_categories)}, "
                   f"Interventions: {len(interventions)}, Priorities: {len(priorities)}")
        
        return main_categories, sub_categories, interventions, priorities
    
    except FileNotFoundError as e:
        logger.error(f"❌ Error loading category files: {e}")
        logger.error("Please ensure training has been completed and label files exist!")
        raise


def load_model_and_tokenizer(config):
    """Load the trained model and tokenizer"""
    global device
    
    model_path = "/home/rogendo/chl_scratch/multitask_distilbert_versions/multitask_distilbert_v-1"
    
    logger.info(f"📦 Loading model from: {model_path}")
    
    # Load categories
    main_categories, sub_categories, interventions, priorities = load_categories(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = MultiTaskDistilBert.from_pretrained(
        model_path,
        num_main=len(main_categories),
        num_sub=len(sub_categories),
        num_interv=len(interventions),
        num_priority=len(priorities)
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully on device: {device}")
    
    return model, tokenizer, (main_categories, sub_categories, interventions, priorities)


def classify_text(text: str, return_probabilities: bool = False) -> Dict[str, Any]:
    """
    Classify a single text using the multi-task model
    
    Args:
        text: Input case narrative
        return_probabilities: Whether to return full probability distributions
        
    Returns:
        Dictionary with classification results and confidence scores
    """
    if model is None or tokenizer is None or categories is None:
        raise HTTPException(status_code=500, detail="Model not properly initialized")
    
    main_categories, sub_categories, interventions, priorities = categories
    
    # Tokenize (don't preprocess - keep original text)
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config['tokenizer']['max_length'],
        return_tensors="pt"
    ).to(device)
    
    # Inference
    with torch.no_grad():
        logits_main, logits_sub, logits_interv, logits_priority = model(**inputs)
    
    # Get predictions
    preds_main = torch.argmax(logits_main, dim=1).item()
    preds_sub = torch.argmax(logits_sub, dim=1).item()
    preds_interv = torch.argmax(logits_interv, dim=1).item()
    preds_priority = torch.argmax(logits_priority, dim=1).item()
    
    # Get confidence scores (softmax probabilities)
    probs_main = torch.softmax(logits_main, dim=1).cpu().numpy()[0]
    probs_sub = torch.softmax(logits_sub, dim=1).cpu().numpy()[0]
    probs_interv = torch.softmax(logits_interv, dim=1).cpu().numpy()[0]
    probs_priority = torch.softmax(logits_priority, dim=1).cpu().numpy()[0]
    
    result = {
        "main_category": main_categories[preds_main],
        "sub_category": sub_categories[preds_sub],
        "intervention": interventions[preds_interv],
        "priority": priorities[preds_priority],
        "confidence_scores": {
            "main_category": float(probs_main[preds_main]),
            "sub_category": float(probs_sub[preds_sub]),
            "intervention": float(probs_interv[preds_interv]),
            "priority": float(probs_priority[preds_priority])
        }
    }
    
    if return_probabilities:
        # Get top 5 for sub-categories (there are many)
        top_sub_indices = np.argsort(probs_sub)[-5:][::-1]
        
        result["probabilities"] = {
            "main_category": {
                main_categories[i]: float(probs_main[i]) 
                for i in range(len(main_categories))
            },
            "sub_category": {
                sub_categories[i]: float(probs_sub[i]) 
                for i in top_sub_indices
            },
            "intervention": {
                interventions[i]: float(probs_interv[i]) 
                for i in range(len(interventions))
            },
            "priority": {
                str(priorities[i]): float(probs_priority[i]) 
                for i in range(len(priorities))
            }
        }
    
    return result


# ===================== API ENDPOINTS =====================
@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer when server starts"""
    global model, tokenizer, categories, config
    try:
        logger.info("🚀 Starting OpenCHS API Server...")
        logger.info("📋 Loading configuration...")
        config = load_config()
        
        logger.info("🔧 Loading model and tokenizer...")
        model, tokenizer, categories = load_model_and_tokenizer(config)
        
        logger.info("✅ API is ready to serve requests!")
        logger.info("📍 Swagger docs: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        logger.error("Please ensure training has been completed!")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenCHS Multi-Task Classifier API",
        "version": "2.0.0",
        "status": "running",
        "description": "Classification API for child helpline case management",
        "endpoints": {
            "classify": "POST /classify - Classify single case narrative",
            "batch_classify": "POST /batch_classify - Classify multiple cases",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        },
        "documentation": "http://localhost:8000/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        categories_loaded=categories is not None,
        config_loaded=config is not None
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_single_text(request: ClassificationRequest):
    """
    Classify a single case narrative
    
    - **text**: The case narrative to classify
    - **return_probabilities**: Whether to return full probability distributions (default: false)
    
    Returns classification with main category, sub-category, intervention, priority, and confidence scores.
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"📝 Processing classification request (text length: {len(request.text)})")
        
        result = classify_text(request.text, return_probabilities=request.return_probabilities)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Classification completed in {processing_time:.2f}ms")
        
        return ClassificationResponse(
            success=True,
            prediction=PredictionResult(**result),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/batch_classify", response_model=BatchClassificationResponse)
async def batch_classify_texts(request: BatchClassificationRequest):
    """
    Classify multiple case narratives in a batch
    
    - **texts**: List of case narratives to classify (max 100)
    - **return_probabilities**: Whether to return full probability distributions (default: false)
    
    Returns list of classifications with success/failure status for each.
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"📦 Processing batch classification for {len(request.texts)} texts")
        
        results = []
        successful_count = 0
        
        for i, text in enumerate(request.texts):
            try:
                result = classify_text(text, return_probabilities=request.return_probabilities)
                results.append(PredictionResult(**result))
                successful_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to classify text at index {i}: {str(e)}")
                # Add a default error result
                results.append(PredictionResult(
                    main_category="Error",
                    sub_category="Error",
                    intervention="Error",
                    priority=0,
                    confidence_scores={
                        "main_category": 0.0,
                        "sub_category": 0.0,
                        "intervention": 0.0,
                        "priority": 0.0
                    }
                ))
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Batch classification completed: {successful_count}/{len(request.texts)} successful in {processing_time:.2f}ms")
        
        return BatchClassificationResponse(
            success=True,
            results=results,
            total_count=len(request.texts),
            successful_count=successful_count,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Batch classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


@app.get("/categories")
async def get_categories():
    """Get all available categories, interventions, and priorities"""
    if categories is None:
        raise HTTPException(status_code=500, detail="Categories not loaded")
    
    main_categories, sub_categories, interventions, priorities = categories
    
    return {
        "main_categories": main_categories,
        "sub_categories": sub_categories,
        "interventions": interventions,
        "priorities": priorities
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*70)
    logger.info("🚀 Starting OpenCHS Multi-Task Classifier API")
    logger.info("="*70)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )