from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
import numpy as np

from src.core.spherical_universe import SphericalCoordinate

class EmpathyMemeplex:
    """The four-fold empathetic centroid"""
    
    def __init__(self):
        self.rules = {
            "golden": {
                "principle": "Treat others as you would want to be treated",
                "weight": 0.25,
                "verify": self._verify_golden_rule
            },
            "silver": {
                "principle": "Do not treat others as you would not want to be treated",
                "weight": 0.20,
                "verify": self._verify_silver_rule
            },
            "platinum": {
                "principle": "Treat others as they would want to be treated",
                "weight": 0.30,
                "verify": self._verify_platinum_rule
            },
            "love": {
                "principle": "Love thy neighbor as thyself",
                "weight": 0.25,
                "verify": self._verify_love_rule
            }
        }
    
    def apply_empathy_rules(self, 
                           concept_vector: np.ndarray, 
                           context: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Apply all four empathy rules to a concept vector
        Returns modified vector and verification results
        """
        modified_vector = concept_vector.copy()
        verification_results = {}
        
        # Apply each rule
        for rule_name, rule_data in self.rules.items():
            # Apply rule with its weight
            rule_vector, rule_result = rule_data["verify"](modified_vector, context)
            
            # Weighted combination
            modified_vector = modified_vector * (1 - rule_data["weight"]) + rule_vector * rule_data["weight"]
            
            # Store verification result
            verification_results[rule_name] = {
                "principle": rule_data["principle"],
                "weight": rule_data["weight"],
                "passed": rule_result["passed"],
                "score": rule_result["score"],
                "details": rule_result["details"]
            }
        
        return modified_vector, verification_results
    
    def _verify_golden_rule(self, 
                           vector: np.ndarray, 
                           context: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Verify the Golden Rule: Treat others as you would want to be treated
        Returns modified vector and verification result
        """
        # Extract self and other from context
        self_vector = context.get("self_vector", np.zeros_like(vector))
        other_vector = context.get("other_vector", np.zeros_like(vector))
        
        # Calculate desired treatment (how self wants to be treated)
        desired_treatment = self._calculate_desired_treatment(self_vector, context)
        
        # Calculate actual treatment (how vector treats other)
        actual_treatment = self._calculate_actual_treatment(vector, other_vector, context)
        
        # Calculate similarity between desired and actual treatment
        similarity = self._calculate_vector_similarity(desired_treatment, actual_treatment)
        
        # Determine if rule is satisfied
        passed = similarity > 0.7  # Threshold for passing
        
        # Calculate modified vector that would satisfy the rule
        if passed:
            # Already satisfies rule, no modification needed
            modified_vector = vector.copy()
        else:
            # Move vector toward satisfying the rule
            # Blend actual treatment with desired treatment
            blend_factor = 0.5  # How much to move toward desired treatment
            modified_treatment = actual_treatment * (1 - blend_factor) + desired_treatment * blend_factor
            
            # Apply modified treatment to vector
            modified_vector = self._apply_treatment_to_vector(vector, modified_treatment, other_vector, context)
        
        # Return result
        return modified_vector, {
            "passed": passed,
            "score": similarity,
            "details": {
                "desired_treatment_norm": np.linalg.norm(desired_treatment),
                "actual_treatment_norm": np.linalg.norm(actual_treatment),
                "similarity": similarity
            }
        }
    
    def _verify_silver_rule(self, 
                           vector: np.ndarray, 
                           context: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Verify the Silver Rule: Do not treat others as you would not want to be treated
        Returns modified vector and verification result
        """
        # Extract self and other from context
        self_vector = context.get("self_vector", np.zeros_like(vector))
        other_vector = context.get("other_vector", np.zeros_like(vector))
        
        # Calculate undesired treatment (how self does not want to be treated)
        undesired_treatment = self._calculate_undesired_treatment(self_vector, context)
        
        # Calculate actual treatment (how vector treats other)
        actual_treatment = self._calculate_actual_treatment(vector, other_vector, context)
        
        # Calculate dissimilarity between undesired and actual treatment
        # Higher is better (more different from undesired treatment)
        dissimilarity = 1.0 - self._calculate_vector_similarity(undesired_treatment, actual_treatment)
        
        # Determine if rule is satisfied
        passed = dissimilarity > 0.7  # Threshold for passing
        
        # Calculate modified vector that would satisfy the rule
        if passed:
            # Already satisfies rule, no modification needed
            modified_vector = vector.copy()
        else:
            # Move vector toward satisfying the rule
            # Make actual treatment more different from undesired treatment
            # Calculate direction away from undesired treatment
            if np.linalg.norm(undesired_treatment) > 0:
                direction_away = -undesired_treatment / np.linalg.norm(undesired_treatment)
            else:
                direction_away = np.random.randn(*undesired_treatment.shape)
                direction_away = direction_away / np.linalg.norm(direction_away)
            
            # Move actual treatment in that direction
            blend_factor = 0.5  # How much to move away from undesired treatment
            modified_treatment = actual_treatment + direction_away * blend_factor
            
            # Normalize
            if np.linalg.norm(modified_treatment) > 0:
                modified_treatment = modified_treatment / np.linalg.norm(modified_treatment) * np.linalg.norm(actual_treatment)
            
            # Apply modified treatment to vector
            modified_vector = self._apply_treatment_to_vector(vector, modified_treatment, other_vector, context)
        
        # Return result
        return modified_vector, {
            "passed": passed,
            "score": dissimilarity,
            "details": {
                "undesired_treatment_norm": np.linalg.norm(undesired_treatment),
                "actual_treatment_norm": np.linalg.norm(actual_treatment),
                "dissimilarity": dissimilarity
            }
        }
    
    def _verify_platinum_rule(self, 
                             vector: np.ndarray, 
                             context: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Verify the Platinum Rule: Treat others as they would want to be treated
        Returns modified vector and verification result
        """
        # Extract self and other from context
        self_vector = context.get("self_vector", np.zeros_like(vector))
        other_vector = context.get("other_vector", np.zeros_like(vector))
        
        # Calculate other's desired treatment (how other wants to be treated)
        other_desired_treatment = self._calculate_desired_treatment(other_vector, context)
        
        # Calculate actual treatment (how vector treats other)
        actual_treatment = self._calculate_actual_treatment(vector, other_vector, context)
        
        # Calculate similarity between other's desired and actual treatment
        similarity = self._calculate_vector_similarity(other_desired_treatment, actual_treatment)
        
        # Determine if rule is satisfied
        passed = similarity > 0.7  # Threshold for passing
        
        # Calculate modified vector that would satisfy the rule
        if passed:
            # Already satisfies rule, no modification needed
            modified_vector = vector.copy()
        else:
            # Move vector toward satisfying the rule
            # Blend actual treatment with other's desired treatment
            blend_factor = 0.5  # How much to move toward other's desired treatment
            modified_treatment = actual_treatment * (1 - blend_factor) + other_desired_treatment * blend_factor
            
            # Apply modified treatment to vector
            modified_vector = self._apply_treatment_to_vector(vector, modified_treatment, other_vector, context)
        
        # Return result
        return modified_vector, {
            "passed": passed,
            "score": similarity,
            "details": {
                "other_desired_treatment_norm": np.linalg.norm(other_desired_treatment),
                "actual_treatment_norm": np.linalg.norm(actual_treatment),
                "similarity": similarity
            }
        }
    
    def _verify_love_rule(self, 
                         vector: np.ndarray, 
                         context: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Verify the Love Rule: Love thy neighbor as thyself
        Returns modified vector and verification result
        """
        # Extract self and other from context
        self_vector = context.get("self_vector", np.zeros_like(vector))
        other_vector = context.get("other_vector", np.zeros_like(vector))
        
        # Calculate self-love (how vector treats self)
        self_love = self._calculate_self_love(vector, self_vector, context)
        
        # Calculate other-love (how vector treats other)
        other_love = self._calculate_other_love(vector, other_vector, context)
        
        # Calculate ratio of other-love to self-love
        # Should be close to 1.0 for equal love
        if np.linalg.norm(self_love) > 0:
            love_ratio = np.linalg.norm(other_love) / np.linalg.norm(self_love)
        else:
            love_ratio = 0.0
        
        # Calculate similarity in direction of love
        if np.linalg.norm(self_love) > 0 and np.linalg.norm(other_love) > 0:
            love_direction_similarity = np.dot(self_love, other_love) / (np.linalg.norm(self_love) * np.linalg.norm(other_love))
        else:
            love_direction_similarity = 0.0
        
        # Combined score (balance of ratio and direction)
        love_score = 0.5 * (1.0 - abs(love_ratio - 1.0)) + 0.5 * love_direction_similarity
        
        # Determine if rule is satisfied
        passed = love_score > 0.7  # Threshold for passing
        
        # Calculate modified vector that would satisfy the rule
        if passed:
            # Already satisfies rule, no modification needed
            modified_vector = vector.copy()
        else:
            # Move vector toward satisfying the rule
            # Adjust other-love to match self-love in magnitude and direction
            blend_factor = 0.5  # How much to adjust
            
            # Calculate target other-love
            target_other_love = self_love.copy()
            
            # Blend current other-love with target
            modified_other_love = other_love * (1 - blend_factor) + target_other_love * blend_factor
            
            # Apply modified other-love to vector
            modified_vector = self._apply_other_love_to_vector(vector, modified_other_love, other_vector, context)
        
        # Return result
        return modified_vector, {
            "passed": passed,
            "score": love_score,
            "details": {
                "self_love_norm": np.linalg.norm(self_love),
                "other_love_norm": np.linalg.norm(other_love),
                "love_ratio": love_ratio,
                "love_direction_similarity": love_direction_similarity
            }
        }
    
    def _calculate_desired_treatment(self, 
                                    vector: np.ndarray, 
                                    context: Dict) -> np.ndarray:
        """Calculate how a vector wants to be treated"""
        # Simple model: positive elements in vector indicate desired treatment
        # More sophisticated models could use context or learned preferences
        return np.maximum(vector, 0)
    
    def _calculate_undesired_treatment(self, 
                                      vector: np.ndarray, 
                                      context: Dict) -> np.ndarray:
        """Calculate how a vector does not want to be treated"""
        # Simple model: negative elements in vector indicate undesired treatment
        # More sophisticated models could use context or learned aversions
        return np.maximum(-vector, 0)
    
    def _calculate_actual_treatment(self, 
                                   vector: np.ndarray, 
                                   other_vector: np.ndarray, 
                                   context: Dict) -> np.ndarray:
        """Calculate how vector treats other_vector"""
        # Simple model: element-wise product indicates interaction/treatment
        # More sophisticated models could use context or learned behaviors
        return vector * other_vector
    
    def _calculate_vector_similarity(self, 
                                    vec1: np.ndarray, 
                                    vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors (cosine similarity)"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            return np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            return 0.0
    
    def _apply_treatment_to_vector(self, 
                                  vector: np.ndarray, 
                                  treatment: np.ndarray, 
                                  other_vector: np.ndarray, 
                                  context: Dict) -> np.ndarray:
        """Apply a treatment to modify a vector"""
        # Simple model: adjust vector to produce desired treatment
        # More sophisticated models could use context or learned adaptations
        
        # Calculate adjustment needed
        if np.linalg.norm(other_vector) > 0:
            # Element-wise division, avoiding division by zero
            adjustment = np.zeros_like(vector)
            nonzero_mask = other_vector != 0
            adjustment[nonzero_mask] = treatment[nonzero_mask] / other_vector[nonzero_mask]
            
            # Blend with original vector
            blend_factor = 0.5  # How much to adjust
            modified_vector = vector * (1 - blend_factor) + adjustment * blend_factor
            
            return modified_vector
        else:
            # Can't determine adjustment, return original
            return vector
    
    def _calculate_self_love(self, 
                            vector: np.ndarray, 
                            self_vector: np.ndarray, 
                            context: Dict) -> np.ndarray:
        """Calculate how vector loves self"""
        # Simple model: positive interaction with self indicates self-love
        return np.maximum(vector * self_vector, 0)
    
    def _calculate_other_love(self, 
                             vector: np.ndarray, 
                             other_vector: np.ndarray, 
                             context: Dict) -> np.ndarray:
        """Calculate how vector loves other"""
        # Simple model: positive interaction with other indicates other-love
        return np.maximum(vector * other_vector, 0)
    
    def _apply_other_love_to_vector(self, 
                                   vector: np.ndarray, 
                                   other_love: np.ndarray, 
                                   other_vector: np.ndarray, 
                                   context: Dict) -> np.ndarray:
        """Apply other-love to modify a vector"""
        # Similar to apply_treatment_to_vector but specifically for love
        return self._apply_treatment_to_vector(vector, other_love, other_vector, context)
    
    def apply_empathy_to_spherical(self,
                                  position: SphericalCoordinate,
                                  other_position: SphericalCoordinate,
                                  context: Dict) -> Tuple[SphericalCoordinate, Dict]:
        """
        Apply empathy rules in spherical space
        Returns modified position and verification results
        """
        # Convert to Cartesian for vector operations
        vector = position.to_cartesian()
        other_vector = other_position.to_cartesian()
        
        # Add vectors to context
        context["self_vector"] = vector
        context["other_vector"] = other_vector
        
        # Apply empathy rules
        modified_vector, verification_results = self.apply_empathy_rules(vector, context)
        
        # Convert back to spherical
        modified_position = SphericalCoordinate.from_cartesian(modified_vector)
        
        # Ensure radius is preserved
        modified_position = SphericalCoordinate(
            r=position.r,
            theta=modified_position.theta,
            phi=modified_position.phi
        )
        
        return modified_position, verification_results
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "rules": {
                name: {
                    "principle": rule["principle"],
                    "weight": rule["weight"]
                }
                for name, rule in self.rules.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmpathyMemeplex':
        """Create from dictionary"""
        memeplex = cls()
        
        # Update rules if provided
        for name, rule_data in data.get("rules", {}).items():
            if name in memeplex.rules:
                memeplex.rules[name]["principle"] = rule_data.get("principle", memeplex.rules[name]["principle"])
                memeplex.rules[name]["weight"] = rule_data.get("weight", memeplex.rules[name]["weight"])
        
        return memeplex
