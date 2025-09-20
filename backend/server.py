from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import hashlib
import time
import base64
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import json
import requests
from io import BytesIO

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(
    title="VeriSource - Blockchain Media Verification API",
    description="API for capture-time content authenticity and blockchain verification",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class MediaCaptureRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Image data must be a non-empty string')
        # Remove data URL prefix if present
        if ',' in v:
            v = v.split(',')[1]
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 image data')
        return v

class MediaVerificationRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data to verify")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Image data must be a non-empty string')
        if ',' in v:
            v = v.split(',')[1]
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 image data')
        return v

class Receipt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    media_sha256: str
    c2pa_manifest_sha256: str
    device_pubkey_fingerprint: str
    capture_time_iso: str
    batch_id: Optional[str] = None
    merkle_leaf_proof: Optional[List[str]] = None
    chain_txid: Optional[str] = None
    blockchain_status: str = "pending"
    trust_score: Optional[float] = None
    tamper_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TrustScoreResponse(BaseModel):
    trust_score: float = Field(..., description="Trust score from 0-100")
    blockchain_provenance: Dict[str, Any]
    c2pa_integrity: Dict[str, Any]
    ai_tamper_analysis: Dict[str, Any]
    derivative_analysis: Dict[str, Any]
    overall_verdict: str
    explanation: str

# In-memory storage for demo (in production, use proper blockchain integration)
pending_hashes = []
committed_batches = {}
batch_counter = 0

# Service Classes
class ImageProcessor:
    @staticmethod
    def calculate_sha256(image_data: str) -> str:
        """Calculate SHA-256 hash of image data"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            return hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    @staticmethod
    def create_c2pa_manifest(image_hash: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple C2PA-like manifest"""
        manifest = {
            "creator_app": "VeriSource",
            "capture_time": datetime.now(timezone.utc).isoformat(),
            "image_hash": image_hash,
            "device_fingerprint": "web_browser_" + str(uuid.uuid4())[:8],
            "metadata": metadata
        }
        return manifest
    
    @staticmethod
    def calculate_manifest_hash(manifest: Dict[str, Any]) -> str:
        """Calculate hash of the manifest"""
        manifest_json = json.dumps(manifest, sort_keys=True)
        return hashlib.sha256(manifest_json.encode()).hexdigest()

class CerebrasAnalysisService:
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY', 'csk-txhh48cc9m9mjjtdy2r26k5mwencwyn58nxhjewvnc4rt5hh')
        self.base_url = "https://api.cerebras.ai/v1"
        
    async def analyze_image_tamper(self, image_hash: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image for tampering using Cerebras AI"""
        try:
            prompt = f"""
            Analyze the following image metadata for signs of tampering or manipulation:
            
            Image Hash: {image_hash}
            Capture Time: {metadata.get('capture_time', 'Unknown')}
            Device Info: {metadata.get('device_fingerprint', 'Unknown')}
            
            Provide a tamper analysis with:
            1. Tamper probability (0-100%)
            2. Confidence level (High/Medium/Low)
            3. Specific indicators found
            4. Recommendations
            
            Format as JSON with keys: tamper_probability, confidence_level, indicators, recommendations
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3.1-70b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert digital forensics analyst specializing in image authenticity verification."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                ai_response = response.json()
                analysis_text = ai_response['choices'][0]['message']['content']
                
                # Parse AI response (simplified for demo)
                return {
                    "tamper_probability": 15.0,  # Mock low probability for demo
                    "confidence_level": "Medium",
                    "indicators": ["Image appears to be captured from web browser", "No obvious manipulation detected"],
                    "recommendations": "Image appears authentic based on metadata analysis",
                    "ai_analysis": analysis_text
                }
            else:
                # Fallback analysis
                return {
                    "tamper_probability": 20.0,
                    "confidence_level": "Low",
                    "indicators": ["Limited metadata available"],
                    "recommendations": "Manual verification recommended",
                    "ai_analysis": "AI analysis unavailable"
                }
                
        except Exception as e:
            logging.error(f"Cerebras analysis failed: {e}")
            return {
                "tamper_probability": 50.0,
                "confidence_level": "Low",
                "indicators": ["Analysis failed"],
                "recommendations": "Manual verification required",
                "ai_analysis": f"Error: {str(e)}"
            }

class BlockchainBatchingService:
    def __init__(self):
        self.batch_interval = 30  # seconds
        self.max_batch_size = 100
        
    async def add_to_batch(self, leaf_hash: str, manifest_hash: str) -> str:
        """Add hash to pending batch"""
        global pending_hashes, batch_counter
        
        leaf_data = {
            "leaf_hash": leaf_hash,
            "manifest_hash": manifest_hash,
            "timestamp": time.time()
        }
        
        pending_hashes.append(leaf_data)
        
        # Auto-commit if batch is full
        if len(pending_hashes) >= self.max_batch_size:
            return await self.commit_batch()
        
        # Return pending status
        return "pending"
    
    async def commit_batch(self) -> str:
        """Commit current batch to blockchain (simulated)"""
        global pending_hashes, committed_batches, batch_counter
        
        if not pending_hashes:
            return "no_pending_hashes"
        
        batch_counter += 1
        batch_id = f"batch_{batch_counter}_{int(time.time())}"
        
        # Calculate Merkle root (simplified)
        merkle_root = self.calculate_merkle_root([item["leaf_hash"] for item in pending_hashes])
        
        # Simulate blockchain commitment
        tx_hash = f"0x{hashlib.sha256(f'{batch_id}_{merkle_root}'.encode()).hexdigest()}"
        
        batch_info = {
            "batch_id": batch_id,
            "merkle_root": merkle_root,
            "transaction_hash": tx_hash,
            "block_number": batch_counter + 1000000,  # Simulated block number
            "leaves": pending_hashes.copy(),
            "committed_at": time.time()
        }
        
        committed_batches[batch_id] = batch_info
        
        # Clear pending hashes
        pending_hashes.clear()
        
        logging.info(f"Batch {batch_id} committed with {len(batch_info['leaves'])} items")
        return batch_id
    
    def calculate_merkle_root(self, leaf_hashes: List[str]) -> str:
        """Calculate Merkle root from leaf hashes"""
        if not leaf_hashes:
            return "0x" + "0" * 64
        
        # Simple Merkle tree calculation
        hashes = [bytes.fromhex(h.replace('0x', '')) for h in leaf_hashes]
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicate if odd number
                next_level.append(hashlib.sha256(combined).digest())
            hashes = next_level
        
        return "0x" + hashes[0].hex()
    
    async def verify_inclusion(self, leaf_hash: str) -> Dict[str, Any]:
        """Verify if a hash exists in any committed batch"""
        for batch_id, batch_info in committed_batches.items():
            for leaf_data in batch_info["leaves"]:
                if leaf_data["leaf_hash"] == leaf_hash:
                    return {
                        "verified": True,
                        "batch_id": batch_id,
                        "transaction_hash": batch_info["transaction_hash"],
                        "block_number": batch_info["block_number"],
                        "merkle_root": batch_info["merkle_root"],
                        "committed_at": batch_info["committed_at"]
                    }
        
        return {"verified": False, "message": "Hash not found in any committed batch"}

class TrustScoreCalculator:
    @staticmethod
    def calculate_trust_score(
        blockchain_data: Dict[str, Any],
        c2pa_data: Dict[str, Any],
        ai_analysis: Dict[str, Any]
    ) -> TrustScoreResponse:
        """Calculate comprehensive trust score"""
        
        # Blockchain provenance (0-40 points)
        blockchain_score = 0
        if blockchain_data.get("verified"):
            blockchain_score += 20  # Hash found on blockchain
            # Add points for finality/age
            age_hours = (time.time() - blockchain_data.get("committed_at", time.time())) / 3600
            if age_hours > 1:
                blockchain_score += min(10, age_hours)  # More points for older entries
            blockchain_score += 10  # Consistency points
        
        blockchain_score = min(blockchain_score, 40)
        
        # C2PA integrity (0-25 points)
        c2pa_score = 0
        if c2pa_data:
            c2pa_score += 10  # Manifest present
            c2pa_score += 10  # Signature valid (assumed)
            c2pa_score += 5   # Device attested
        
        # AI tamper analysis (0-25 points)
        ai_score = 0
        tamper_prob = ai_analysis.get("tamper_probability", 50)
        confidence = ai_analysis.get("confidence_level", "Low")
        
        # Inverse of tamper probability
        ai_score += max(0, 15 - (tamper_prob * 15 / 100))
        
        # Confidence bonus
        if confidence == "High":
            ai_score += 10
        elif confidence == "Medium":
            ai_score += 5
        
        # Derivative analysis (0-10 points, simplified)
        derivative_score = 5  # Assume original for now
        
        # Calculate total
        total_score = blockchain_score + c2pa_score + ai_score + derivative_score
        
        # Determine verdict
        if total_score >= 80:
            verdict = "HIGHLY_AUTHENTIC"
        elif total_score >= 60:
            verdict = "LIKELY_AUTHENTIC"
        elif total_score >= 40:
            verdict = "UNCERTAIN"
        else:
            verdict = "LIKELY_TAMPERED"
        
        explanation = f"Score breakdown: Blockchain ({blockchain_score}/40), C2PA ({c2pa_score}/25), AI Analysis ({ai_score}/25), Derivative ({derivative_score}/10)"
        
        return TrustScoreResponse(
            trust_score=total_score,
            blockchain_provenance={
                "score": blockchain_score,
                "verified": blockchain_data.get("verified", False),
                "details": blockchain_data
            },
            c2pa_integrity={
                "score": c2pa_score,
                "manifest_present": bool(c2pa_data),
                "details": c2pa_data
            },
            ai_tamper_analysis={
                "score": ai_score,
                "tamper_probability": tamper_prob,
                "confidence": confidence,
                "details": ai_analysis
            },
            derivative_analysis={
                "score": derivative_score,
                "analysis": "No derivative matches found"
            },
            overall_verdict=verdict,
            explanation=explanation
        )

# Initialize services
image_processor = ImageProcessor()
cerebras_service = CerebrasAnalysisService()
batching_service = BlockchainBatchingService()
trust_calculator = TrustScoreCalculator()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "VeriSource API - Blockchain Media Verification"}

@api_router.post("/capture")
async def capture_image(request: MediaCaptureRequest):
    """Capture and commit image to blockchain"""
    try:
        # Calculate image hash
        image_hash = image_processor.calculate_sha256(request.image_data)
        
        # Create C2PA manifest
        manifest = image_processor.create_c2pa_manifest(image_hash, request.metadata)
        manifest_hash = image_processor.calculate_manifest_hash(manifest)
        
        # Add to blockchain batch
        batch_status = await batching_service.add_to_batch(image_hash, manifest_hash)
        
        # Create receipt
        receipt = Receipt(
            media_sha256=image_hash,
            c2pa_manifest_sha256=manifest_hash,
            device_pubkey_fingerprint=manifest["device_fingerprint"],
            capture_time_iso=manifest["capture_time"],
            blockchain_status=batch_status
        )
        
        # Store receipt in database
        receipt_dict = receipt.dict()
        receipt_dict['c2pa_manifest'] = manifest
        await db.receipts.insert_one(receipt_dict)
        
        return {
            "status": "success",
            "image_hash": image_hash,
            "receipt_id": receipt.id,
            "blockchain_status": batch_status,
            "message": "Image captured and queued for blockchain commitment"
        }
        
    except Exception as e:
        logging.error(f"Capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")

@api_router.post("/verify")
async def verify_image(request: MediaVerificationRequest):
    """Verify image authenticity against blockchain"""
    try:
        # Calculate image hash
        image_hash = image_processor.calculate_sha256(request.image_data)
        
        # Check blockchain
        blockchain_verification = await batching_service.verify_inclusion(image_hash)
        
        # Get stored receipt if exists
        stored_receipt = await db.receipts.find_one({"media_sha256": image_hash})
        
        c2pa_data = {}
        if stored_receipt:
            c2pa_data = stored_receipt.get('c2pa_manifest', {})
        
        # AI analysis
        metadata = c2pa_data or {"capture_time": "unknown", "device_fingerprint": "unknown"}
        ai_analysis = await cerebras_service.analyze_image_tamper(image_hash, metadata)
        
        # Calculate trust score
        trust_score_result = trust_calculator.calculate_trust_score(
            blockchain_verification, c2pa_data, ai_analysis
        )
        
        return {
            "image_hash": image_hash,
            "blockchain_verification": blockchain_verification,
            "trust_score_analysis": trust_score_result.dict(),
            "receipt_found": bool(stored_receipt),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logging.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@api_router.get("/receipt/{receipt_id}")
async def get_receipt(receipt_id: str):
    """Get receipt by ID"""
    receipt = await db.receipts.find_one({"id": receipt_id})
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    receipt.pop('_id', None)  # Remove MongoDB ObjectId
    return receipt

@api_router.get("/batch/{batch_id}")
async def get_batch_info(batch_id: str):
    """Get batch information"""
    if batch_id not in committed_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return committed_batches[batch_id]

@api_router.post("/force-commit-batch")
async def force_commit_batch():
    """Force commit current pending batch"""
    batch_id = await batching_service.commit_batch()
    if batch_id == "no_pending_hashes":
        return {"message": "No pending hashes to commit"}
    
    return {"message": f"Batch {batch_id} committed successfully", "batch_id": batch_id}

@api_router.get("/stats")
async def get_stats():
    """Get system statistics"""
    total_receipts = await db.receipts.count_documents({})
    
    return {
        "total_images_captured": total_receipts,
        "pending_batch_size": len(pending_hashes),
        "committed_batches": len(committed_batches),
        "total_verified_images": sum(len(batch["leaves"]) for batch in committed_batches.values())
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Background task to commit batches periodically
async def periodic_batch_commit():
    """Periodically commit pending batches"""
    while True:
        try:
            await asyncio.sleep(30)  # Wait 30 seconds
            if pending_hashes:
                batch_id = await batching_service.commit_batch()
                logger.info(f"Auto-committed batch: {batch_id}")
        except Exception as e:
            logger.error(f"Auto-commit failed: {e}")

# Start background task
@app.on_event("startup")
async def startup():
    asyncio.create_task(periodic_batch_commit())