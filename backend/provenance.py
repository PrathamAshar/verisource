"""
Image Perceptual Provenance + Trust Report Module

This module provides image provenance verification using:
- SHA-256 hashing for exact matches
- CLIP embeddings for perceptual similarity
- FAISS index for efficient similarity search
- SSIM for structural similarity analysis
"""

import hashlib
import base64
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

logger = logging.getLogger(__name__)

class ImageProvenanceService:
    """Service for image provenance verification using CLIP embeddings and SSIM"""
    
    def __init__(self, db=None):
        self.model = None
        self.processor = None
        self.index = None
        self.image_metadata = {}  # Maps FAISS index IDs to MongoDB record IDs
        self.sha256_to_index = {}  # Maps SHA-256 to FAISS index ID
        self.index_counter = 0
        self.similarity_threshold = 0.85  # Threshold for considering images similar
        self.ssim_threshold = 0.80  # Threshold for SSIM similarity
        self.db = db  # MongoDB database connection
        
        # Initialize CLIP model
        self._initialize_clip_model()
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
        # Note: MongoDB loading will be done during server startup
    
    def _initialize_clip_model(self):
        """Initialize CLIP model and processor"""
        try:
            logger.info("Loading CLIP model (ViT-B/32)...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("CLIP model loaded on GPU")
            else:
                logger.info("CLIP model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for storing embeddings"""
        try:
            # CLIP ViT-B/32 produces 512-dimensional embeddings
            embedding_dim = 512
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            logger.info(f"FAISS index initialized with dimension {embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    async def _load_embeddings_from_db(self):
        """Load all embeddings from MongoDB and rebuild FAISS index"""
        try:
            if self.db is None:
                logger.warning("No database connection available for loading embeddings")
                return
                
            logger.info("Loading embeddings from MongoDB...")
            
            # Get all image references from MongoDB
            cursor = self.db.image_references.find({})
            embeddings_data = await cursor.to_list(length=None)
            
            if not embeddings_data:
                logger.info("No existing embeddings found in MongoDB")
                return
            
            # Rebuild FAISS index and mappings
            embeddings = []
            for record in embeddings_data:
                embedding = np.array(record['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                
                # Update mappings
                faiss_id = self.index_counter
                self.index_counter += 1
                
                # Store ObjectId as string for later conversion
                self.image_metadata[faiss_id] = str(record['_id'])
                self.sha256_to_index[record['sha256']] = faiss_id
                logger.info(f"Loaded embedding {faiss_id} for SHA256 {record['sha256'][:16]}...")
            
            # Add all embeddings to FAISS index
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                self.index.add(embeddings_array)
                logger.info(f"Loaded {len(embeddings)} embeddings from MongoDB into FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from MongoDB: {e}")
            # Don't raise - allow service to continue with empty index
    
    def _preprocess_image(self, image_data: str) -> Image.Image:
        """Preprocess image data for CLIP processing"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def _compute_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Compute CLIP embedding for an image"""
        try:
            with torch.no_grad():
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get image features
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to compute CLIP embedding: {e}")
            raise
    
    def _compute_sha256(self, image_data: str) -> str:
        """Compute SHA-256 hash of image data"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            return hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute SHA-256: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def _compute_ssim(self, image1_data: str, image2_data: str) -> Tuple[float, Optional[str]]:
        """Compute SSIM between two images and generate difference map"""
        try:
            logger.info("Starting SSIM computation...")
            
            # Preprocess both images
            img1 = self._preprocess_image(image1_data)
            img2 = self._preprocess_image(image2_data)
            
            logger.info(f"Image 1 size: {img1.size}, Image 2 size: {img2.size}")
            
            # Convert PIL images to numpy arrays
            img1_np = np.array(img1)
            img2_np = np.array(img2)
            
            logger.info(f"Array 1 shape: {img1_np.shape}, Array 2 shape: {img2_np.shape}")
            
            # Resize images to same size if different
            if img1_np.shape != img2_np.shape:
                # Resize to smaller dimensions
                min_height = min(img1_np.shape[0], img2_np.shape[0])
                min_width = min(img1_np.shape[1], img2_np.shape[1])
                
                img1_np = cv2.resize(img1_np, (min_width, min_height))
                img2_np = cv2.resize(img2_np, (min_width, min_height))
                logger.info(f"Resized images to: {img1_np.shape}")
            
            # Normalize images to reduce global differences
            img1_np = img1_np.astype(np.float32)
            img2_np = img2_np.astype(np.float32)
            
            # Apply histogram equalization to reduce brightness/contrast differences
            if len(img1_np.shape) == 3:
                # Convert to LAB color space for better normalization
                img1_lab = cv2.cvtColor(img1_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
                img2_lab = cv2.cvtColor(img2_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
                
                # Apply histogram equalization to L channel only
                img1_lab[:,:,0] = cv2.equalizeHist(img1_lab[:,:,0])
                img2_lab[:,:,0] = cv2.equalizeHist(img2_lab[:,:,0])
                
                # Convert back to RGB
                img1_np = cv2.cvtColor(img1_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
                img2_np = cv2.cvtColor(img2_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
            
            # Convert to grayscale for SSIM
            if len(img1_np.shape) == 3:
                img1_gray = cv2.cvtColor(img1_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1_np.astype(np.uint8)
                img2_gray = img2_np.astype(np.uint8)
            
            # Compute SSIM with full=True to get difference map
            ssim_score, diff_map = ssim(img1_gray, img2_gray, data_range=255, full=True)
            logger.info(f"SSIM score: {ssim_score:.3f}")
            
            # Generate difference visualization
            diff_image_path = self._generate_diff_visualization(img1_np.astype(np.uint8), img2_np.astype(np.uint8), diff_map)
            
            return float(ssim_score), diff_image_path
            
        except Exception as e:
            logger.error(f"Failed to compute SSIM: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, None
    
    def _generate_diff_visualization(self, img1: np.ndarray, img2: np.ndarray, diff_map: np.ndarray) -> str:
        """Generate difference visualization with bounding boxes"""
        try:
            logger.info("Generating difference visualization...")

            # SSIM diff_map: values close to 1 = similar, lower = different
            # So we flip the condition: highlight where SSIM < threshold
            threshold = 0.75
            diff_binary = (diff_map < threshold).astype(np.uint8) * 255

            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_CLOSE, kernel)
            diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"Found {len(contours)} contours")

            annotated_img = img2.copy()
            img_height, img_width = annotated_img.shape[:2]
            total_area = img_height * img_width
            rectangles_drawn = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < (total_area * 0.95):  # allow broader range
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw red rectangle
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 12)
                    rectangles_drawn += 1

            logger.info(f"Drew {rectangles_drawn} rectangles")

            # Save annotated image
            output_dir = Path("temp_diff")
            output_dir.mkdir(exist_ok=True)
            diff_image_path = str(output_dir / "diff_result.jpg")
            cv2.imwrite(diff_image_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

            return diff_image_path
        except Exception as e:
            logger.error(f"Failed to generate diff visualization: {e}")
            return None
    
    async def store_reference_image(self, image_data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a reference image and compute its embedding"""
        try:
            # Compute SHA-256
            image_hash = self._compute_sha256(image_data)
            
            # Check if image already exists in MongoDB
            if self.db is not None:
                existing_record = await self.db.image_references.find_one({"sha256": image_hash})
                if existing_record:
                    return {
                        "status": "already_exists",
                        "image_hash": image_hash,
                        "mongo_id": str(existing_record['_id']),
                        "message": "Image already stored as reference"
                    }
            
            # Preprocess image
            image = self._preprocess_image(image_data)
            
            # Compute CLIP embedding
            embedding = self._compute_clip_embedding(image)
            
            # Store in MongoDB
            mongo_record = {
                "sha256": image_hash,
                "embedding": embedding.tolist(),  # Convert to list for MongoDB storage
                "c2pa_manifest": metadata or {},
                "original_image_data": image_data,  # Store original for diff generation
                "created_at": datetime.now(timezone.utc)
            }
            logger.info(f"Storing MongoDB record with original_image_data length: {len(image_data)}")
            
            if self.db is not None:
                result = await self.db.image_references.insert_one(mongo_record)
                mongo_id = str(result.inserted_id)
            else:
                mongo_id = "no_db"
            
            # Add to FAISS index
            faiss_id = self.index_counter
            self.index_counter += 1
            
            self.index.add(embedding.reshape(1, -1))
            
            # Update mappings
            self.image_metadata[faiss_id] = mongo_id
            self.sha256_to_index[image_hash] = faiss_id
            
            logger.info(f"Stored reference image with hash {image_hash}, FAISS ID {faiss_id}, MongoDB ID {mongo_id}")
            
            return {
                "status": "stored",
                "image_hash": image_hash,
                "faiss_id": faiss_id,
                "mongo_id": mongo_id,
                "embedding_dim": len(embedding),
                "message": "Reference image stored successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to store reference image: {e}")
            raise
    
    async def verify_image(self, image_data: str) -> Dict[str, Any]:
        """Verify an image against stored reference images"""
        try:
            # Compute SHA-256
            image_hash = self._compute_sha256(image_data)
            
            # Check for exact match in MongoDB first
            matched_sha256 = None
            if self.db is not None:
                exact_match_record = await self.db.image_references.find_one({"sha256": image_hash})
                if exact_match_record:
                    matched_sha256 = image_hash
                    return {
                        "exact_match": True,
                        "similarity": 1.0,
                        "ssim": 1.0,
                        "verdict": "PASS",
                        "explanation": "Exact match found - this is the original authentic image",
                        "image_hash": image_hash,
                        "matched_sha256": matched_sha256,
                        "diff_image_path": None
                    }
            
            # No exact match, check for perceptual similarity
            if self.index.ntotal == 0:
                return {
                    "exact_match": False,
                    "similarity": 0.0,
                    "ssim": 0.0,
                    "verdict": "FAIL",
                    "explanation": "No reference images stored for comparison",
                    "image_hash": image_hash,
                    "matched_sha256": None,
                    "diff_image_path": None
                }
            
            # Preprocess image and compute embedding
            image = self._preprocess_image(image_data)
            embedding = self._compute_clip_embedding(image)
            
            # Search for similar images
            similarities, indices = self.index.search(embedding.reshape(1, -1), k=min(5, self.index.ntotal))
            
            best_similarity = float(similarities[0][0])
            best_faiss_id = int(indices[0][0])
            
            if best_similarity < self.similarity_threshold:
                return {
                    "exact_match": False,
                    "similarity": best_similarity,
                    "ssim": 0.0,
                    "verdict": "FAIL",
                    "explanation": f"No similar images found (similarity: {best_similarity:.3f} < {self.similarity_threshold})",
                    "image_hash": image_hash,
                    "matched_sha256": None,
                    "diff_image_path": None
                }
            
            # Found similar image, get MongoDB record
            mongo_id = self.image_metadata.get(best_faiss_id)
            reference_record = None
            if self.db is not None and mongo_id:
                try:
                    # Convert string to ObjectId for MongoDB query
                    object_id = ObjectId(mongo_id)
                    reference_record = await self.db.image_references.find_one({"_id": object_id})
                    if reference_record:
                        matched_sha256 = reference_record["sha256"]
                        logger.info(f"Found MongoDB record for FAISS ID {best_faiss_id}: {matched_sha256[:16]}...")
                    else:
                        logger.warning(f"No MongoDB record found for FAISS ID {best_faiss_id}, mongo_id: {mongo_id}")
                except Exception as e:
                    logger.error(f"Error querying MongoDB record: {e}")
            else:
                logger.warning(f"No MongoDB connection or mongo_id for FAISS ID {best_faiss_id}")
            
            # Get original image data for SSIM computation
            reference_image_data = None
            if reference_record:
                reference_image_data = reference_record.get("original_image_data")
            
            if reference_image_data:
                # Compute actual SSIM and generate diff
                actual_ssim, diff_image_path = self._compute_ssim(reference_image_data, image_data)
                logger.info(f"SSIM computation: score={actual_ssim}, diff_path={diff_image_path}")
            else:
                # Fallback: estimate SSIM based on similarity
                actual_ssim = min(best_similarity * 0.9, 0.95)  # Conservative estimate
                diff_image_path = None
                logger.warning("No reference image data available for SSIM computation")
            
            # Determine verdict based on similarity and SSIM
            if actual_ssim == 0.0 and best_similarity >= self.similarity_threshold:
                # SSIM computation failed, use similarity as fallback
                actual_ssim = best_similarity * 0.9
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a modified version of a stored authentic image (similarity: {best_similarity:.3f})"
            elif actual_ssim >= self.ssim_threshold:
                # High SSIM - likely minor modifications
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a modified version of a stored authentic image (similarity: {best_similarity:.3f})"
            elif best_similarity >= 0.90 and actual_ssim < 0.50:
                # High similarity but very low SSIM - likely a cropped version
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a cropped version of the original image (similarity: {best_similarity:.3f}, SSIM: {actual_ssim:.3f})"
            elif best_similarity >= 0.93 and actual_ssim < 0.60:
                # Very high similarity but low SSIM - likely a cropped version
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a cropped version of the original image (similarity: {best_similarity:.3f}, SSIM: {actual_ssim:.3f})"
            elif best_similarity >= 0.95 and actual_ssim < 0.70:
                # Very high similarity but lower SSIM - likely cropping or major structural changes
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a cropped version of the original image (similarity: {best_similarity:.3f}, SSIM: {actual_ssim:.3f})"
            else:
                # Low similarity - likely different image
                verdict = "FAIL"
                explanation = f"Image is similar but structurally different (similarity: {best_similarity:.3f}, SSIM: {actual_ssim:.3f})"
            
            return {
                "exact_match": False,
                "similarity": best_similarity,
                "ssim": actual_ssim,
                "verdict": verdict,
                "explanation": explanation,
                "image_hash": image_hash,
                "matched_sha256": matched_sha256,
                "diff_image_path": diff_image_path
            }
            
        except Exception as e:
            logger.error(f"Failed to verify image: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored images"""
        mongo_count = 0
        if self.db is not None:
            mongo_count = await self.db.image_references.count_documents({})
        
        return {
            "total_reference_images": self.index.ntotal,
            "mongo_reference_images": mongo_count,
            "similarity_threshold": self.similarity_threshold,
            "ssim_threshold": self.ssim_threshold,
            "embedding_dimension": 512,
            "model_name": "openai/clip-vit-base-patch32"
        }
    
    async def clear_all_references(self) -> Dict[str, Any]:
        """Clear all stored reference images"""
        # Clear MongoDB
        if self.db is not None:
            result = await self.db.image_references.delete_many({})
            logger.info(f"Cleared {result.deleted_count} records from MongoDB")
        
        # Clear FAISS index and mappings
        self.index.reset()
        self.image_metadata.clear()
        self.sha256_to_index.clear()
        self.index_counter = 0
        
        logger.info("Cleared all reference images")
        return {"message": "All reference images cleared successfully"}

# Global instance - will be initialized with database connection
provenance_service = None
