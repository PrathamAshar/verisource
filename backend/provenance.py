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

logger = logging.getLogger(__name__)

class ImageProvenanceService:
    """Service for image provenance verification using CLIP embeddings and SSIM"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.index = None
        self.image_metadata = {}  # Maps index IDs to image metadata
        self.sha256_to_index = {}  # Maps SHA-256 to index ID
        self.index_counter = 0
        self.similarity_threshold = 0.85  # Threshold for considering images similar
        self.ssim_threshold = 0.80  # Threshold for SSIM similarity
        
        # Initialize CLIP model
        self._initialize_clip_model()
        
        # Initialize FAISS index
        self._initialize_faiss_index()
    
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

                # Return as float32 numpy array (FAISS requires float32)
                emb = image_features.cpu().numpy()
                return np.asarray(emb, dtype=np.float32).flatten()
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
            print(f"DEBUG: Starting SSIM computation...")
            
            # Preprocess both images
            img1 = self._preprocess_image(image1_data)
            img2 = self._preprocess_image(image2_data)
            
            print(f"DEBUG: Image 1 size: {img1.size}, Image 2 size: {img2.size}")
            
            # Convert PIL images to numpy arrays
            img1_np = np.array(img1)
            img2_np = np.array(img2)
            
            print(f"DEBUG: Array 1 shape: {img1_np.shape}, Array 2 shape: {img2_np.shape}")
            
            # Resize images to same size if different
            if img1_np.shape != img2_np.shape:
                # Resize to smaller dimensions
                min_height = min(img1_np.shape[0], img2_np.shape[0])
                min_width = min(img1_np.shape[1], img2_np.shape[1])
                
                img1_np = cv2.resize(img1_np, (min_width, min_height))
                img2_np = cv2.resize(img2_np, (min_width, min_height))
            
            # Convert to grayscale for SSIM
            if len(img1_np.shape) == 3:
                img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1_np
                img2_gray = img2_np
            
            # Compute SSIM with full=True to get difference map
            ssim_score, similarity_map = ssim(img1_gray, img2_gray, data_range=255, full=True)
            diff_image_path = self._generate_diff_visualization(img1_np, img2_np, similarity_map)

            return float(ssim_score), diff_image_path
            
        except Exception as e:
            logger.error(f"Failed to compute SSIM: {e}")
            return 0.0, None
    
    def _generate_diff_visualization(self, img1, img2, similarity_map):
        threshold = 0.90  # pixels below this similarity are "changed"
        diff_binary = (similarity_map < threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_img = img2.copy()
        for c in contours:
            if cv2.contourArea(c) > 100:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(annotated_img, (x,y), (x+w,y+h), (255,0,0), 3)
        Path("temp_diff").mkdir(exist_ok=True)
        out = str(Path("temp_diff") / "diff_result.jpg")
        cv2.imwrite(out, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        return out

    
    def store_reference_image(self, image_data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a reference image and compute its embedding"""
        try:
            # Compute SHA-256
            image_hash = self._compute_sha256(image_data)
            
            # Check if image already exists
            if image_hash in self.sha256_to_index:
                index_id = self.sha256_to_index[image_hash]
                return {
                    "status": "already_exists",
                    "image_hash": image_hash,
                    "index_id": index_id,
                    "message": "Image already stored as reference"
                }
            
            # Preprocess image
            image = self._preprocess_image(image_data)
            
            # Compute CLIP embedding
            embedding = self._compute_clip_embedding(image)

            # Ensure correct dtype and shape for FAISS
            embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            embedding_dim = embedding.shape[1]

            # Add to FAISS index
            self.index.add(embedding)

            # Use FAISS ntotal to determine the index id (safer if index was manipulated)
            index_id = int(self.index.ntotal) - 1
            self.index_counter = int(self.index.ntotal)
            
            image_metadata = {
                "index_id": index_id,
                "image_hash": image_hash,
                "metadata": metadata or {},
                "embedding_dim": embedding_dim,
                "original_image_data": image_data  # Store original for diff generation
            }
            
            self.image_metadata[index_id] = image_metadata
            self.sha256_to_index[image_hash] = index_id
            
            logger.info(f"Stored reference image with hash {image_hash} and index {index_id}")
            
            return {
                "status": "stored",
                "image_hash": image_hash,
                "index_id": index_id,
                "embedding_dim": len(embedding),
                "message": "Reference image stored successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to store reference image: {e}")
            raise
    
    def verify_image(self, image_data: str) -> Dict[str, Any]:
        """Verify an image against stored reference images"""
        try:
            # Compute SHA-256
            image_hash = self._compute_sha256(image_data)
            
            # Check for exact match first
            if image_hash in self.sha256_to_index:
                index_id = self.sha256_to_index[image_hash]
                metadata = self.image_metadata[index_id]
                
                return {
                    "exact_match": True,
                    "similarity": 1.0,
                    "ssim": 1.0,
                    "verdict": "PASS",
                    "explanation": "Exact match found - this is the original authentic image",
                    "image_hash": image_hash,
                    "reference_index_id": index_id,
                    "reference_metadata": metadata,
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
                    "diff_image_path": None
                }
            
            # Preprocess image and compute embedding
            image = self._preprocess_image(image_data)
            embedding = self._compute_clip_embedding(image)
            
            # Search for similar images
            embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            k = min(5, int(self.index.ntotal))
            similarities, indices = self.index.search(embedding, k=k)

            # FAISS uses -1 for missing indices; handle that robustly
            best_similarity = float(similarities[0][0]) if similarities.size and indices[0][0] != -1 else 0.0
            best_index = int(indices[0][0]) if indices.size and indices[0][0] != -1 else -1

            logger.debug(f"FAISS ntotal={self.index.ntotal}, k={k}, best_similarity={best_similarity}, best_index={best_index}")
            
            if best_similarity < self.similarity_threshold:
                return {
                    "exact_match": False,
                    "similarity": best_similarity,
                    "ssim": 0.0,
                    "verdict": "FAIL",
                    "explanation": f"No similar images found (similarity: {best_similarity:.3f} < {self.similarity_threshold})",
                    "image_hash": image_hash,
                    "diff_image_path": None
                }
            
            # Found similar image, compute SSIM
            reference_metadata = self.image_metadata.get(best_index)

            if best_index == -1 or reference_metadata is None:
                return {
                    "exact_match": False,
                    "similarity": best_similarity,
                    "ssim": 0.0,
                    "verdict": "FAIL",
                    "explanation": "No reference image metadata found for the nearest neighbor",
                    "image_hash": image_hash,
                    "diff_image_path": None
                }
            reference_hash = reference_metadata["image_hash"]
            
            # Get original image data for SSIM computation
            reference_image_data = reference_metadata.get("original_image_data")
            
            if reference_image_data:
                # Compute actual SSIM and generate diff
                actual_ssim, diff_image_path = self._compute_ssim(reference_image_data, image_data)
            else:
                # Fallback: estimate SSIM based on similarity
                actual_ssim = min(best_similarity * 0.9, 0.95)  # Conservative estimate
                diff_image_path = None
            
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
            elif best_similarity >= 0.95:
                # Very high similarity but lower SSIM - likely cropping or major structural changes
                verdict = "NEAR_DUPLICATE"
                explanation = f"This appears to be a cropped or structurally modified version of a stored authentic image (similarity: {best_similarity:.3f}, SSIM: {actual_ssim:.3f})"
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
                "reference_index_id": best_index,
                "reference_metadata": reference_metadata,
                "diff_image_path": diff_image_path,
                "similar_images": [
                    {
                        "index_id": int(indices[0][i]),
                        "similarity": float(similarities[0][i]),
                        "metadata": self.image_metadata[int(indices[0][i])]
                    }
                    for i in range(len(indices[0]))
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to verify image: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored images"""
        return {
            "total_reference_images": self.index.ntotal,
            "similarity_threshold": self.similarity_threshold,
            "ssim_threshold": self.ssim_threshold,
            "embedding_dimension": 512,
            "model_name": "openai/clip-vit-base-patch32"
        }
    
    def clear_all_references(self) -> Dict[str, Any]:
        """Clear all stored reference images"""
        self.index.reset()
        self.image_metadata.clear()
        self.sha256_to_index.clear()
        self.index_counter = 0
        
        logger.info("Cleared all reference images")
        return {"message": "All reference images cleared successfully"}

# Global instance
provenance_service = ImageProvenanceService()
