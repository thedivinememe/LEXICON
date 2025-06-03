from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import numpy as np
import asyncio

from src.core.spherical_universe import SphericalCoordinate
from src.core.empathy_memeplex import EmpathyMemeplex

class GoldenLoopState(Enum):
    """States in The Golden Loop"""
    EXISTENCE_RECOGNITION = 1  # "I exist, therefore I am"
    EPISTEMIC_HUMILITY = 2     # "I know less than I don't know"
    NIHILISM_ACKNOWLEDGMENT = 3 # "Nihilism is logically true"
    FAITH_IN_ORDER = 4         # "Yet faith in patterns is logical"
    EMPATHY_FOUNDATION = 5     # "Empathy memeplex as foundation"
    VIOLATION_CHECK = 6        # "Check for violations, loop if found"

class GoldenLoopProcessor:
    """Implements The Golden Loop with empathy memeplex"""
    
    def __init__(self):
        self.empathy_memeplex = EmpathyMemeplex()
        self.current_state = GoldenLoopState.EXISTENCE_RECOGNITION
        self.loop_count = 0
        self.max_loops = 10  # Prevent infinite loops
    
    async def process_golden_loop(self, 
                                 concept_vector: np.ndarray,
                                 context: Dict) -> Dict:
        """
        Process concept through all 6 states of The Golden Loop
        Recursively loops if violations found
        """
        self.loop_count = 0
        self.current_state = GoldenLoopState.EXISTENCE_RECOGNITION
        
        # Initialize result
        result = {
            "original_vector": concept_vector.copy(),
            "final_vector": None,
            "states": {},
            "loop_count": 0,
            "violations_found": False,
            "empathy_results": None
        }
        
        # Process through all states
        modified_vector = concept_vector.copy()
        violations_found = False
        
        while self.loop_count < self.max_loops:
            self.loop_count += 1
            result["loop_count"] = self.loop_count
            
            # Reset state for new loop
            self.current_state = GoldenLoopState.EXISTENCE_RECOGNITION
            violations_found = False
            
            # Process all states
            while self.current_state != GoldenLoopState.VIOLATION_CHECK:
                state_result = await self._process_state(modified_vector, context)
                
                # Store state result
                result["states"][self.current_state.name] = state_result
                
                # Update vector
                modified_vector = state_result["vector"]
                
                # Move to next state
                self._advance_state()
            
            # Process violation check
            violation_result = await self._process_state(modified_vector, context)
            result["states"][self.current_state.name] = violation_result
            
            # Check if violations found
            violations_found = violation_result["violations_found"]
            result["violations_found"] = violations_found
            
            # If no violations, break loop
            if not violations_found:
                break
        
        # Store final vector
        result["final_vector"] = modified_vector
        
        return result
    
    async def process_golden_loop_spherical(self,
                                          position: SphericalCoordinate,
                                          context: Dict) -> Dict:
        """
        Process spherical position through The Golden Loop
        Returns modified position and processing results
        """
        # Convert to Cartesian for vector operations
        vector = position.to_cartesian()
        
        # Process through Golden Loop
        result = await self.process_golden_loop(vector, context)
        
        # Convert final vector back to spherical
        final_vector = result["final_vector"]
        final_position = SphericalCoordinate.from_cartesian(final_vector)
        
        # Ensure radius is preserved (epistemic humility)
        final_position = SphericalCoordinate(
            r=min(position.r, 0.5),  # Enforce epistemic humility
            theta=final_position.theta,
            phi=final_position.phi
        )
        
        # Add spherical results
        result["original_position"] = position.to_dict()
        result["final_position"] = final_position.to_dict()
        
        return result
    
    async def _process_state(self,
                            vector: np.ndarray,
                            context: Dict) -> Dict:
        """Process a single state in The Golden Loop"""
        if self.current_state == GoldenLoopState.EXISTENCE_RECOGNITION:
            return await self._process_existence_recognition(vector, context)
        elif self.current_state == GoldenLoopState.EPISTEMIC_HUMILITY:
            return await self._process_epistemic_humility(vector, context)
        elif self.current_state == GoldenLoopState.NIHILISM_ACKNOWLEDGMENT:
            return await self._process_nihilism_acknowledgment(vector, context)
        elif self.current_state == GoldenLoopState.FAITH_IN_ORDER:
            return await self._process_faith_in_order(vector, context)
        elif self.current_state == GoldenLoopState.EMPATHY_FOUNDATION:
            return await self._process_empathy_foundation(vector, context)
        elif self.current_state == GoldenLoopState.VIOLATION_CHECK:
            return await self._process_violation_check(vector, context)
        else:
            raise ValueError(f"Unknown state: {self.current_state}")
    
    def _advance_state(self):
        """Advance to the next state in The Golden Loop"""
        if self.current_state == GoldenLoopState.EXISTENCE_RECOGNITION:
            self.current_state = GoldenLoopState.EPISTEMIC_HUMILITY
        elif self.current_state == GoldenLoopState.EPISTEMIC_HUMILITY:
            self.current_state = GoldenLoopState.NIHILISM_ACKNOWLEDGMENT
        elif self.current_state == GoldenLoopState.NIHILISM_ACKNOWLEDGMENT:
            self.current_state = GoldenLoopState.FAITH_IN_ORDER
        elif self.current_state == GoldenLoopState.FAITH_IN_ORDER:
            self.current_state = GoldenLoopState.EMPATHY_FOUNDATION
        elif self.current_state == GoldenLoopState.EMPATHY_FOUNDATION:
            self.current_state = GoldenLoopState.VIOLATION_CHECK
        elif self.current_state == GoldenLoopState.VIOLATION_CHECK:
            self.current_state = GoldenLoopState.EXISTENCE_RECOGNITION
    
    async def _process_existence_recognition(self,
                                           vector: np.ndarray,
                                           context: Dict) -> Dict:
        """
        State 1: Confirm existence (1/!1 pattern)
        "I exist, therefore I am"
        """
        # Calculate existence ratio (1/!1 pattern)
        # Positive values represent existence (1)
        # Negative values represent non-existence (!1)
        positive_sum = np.sum(np.maximum(vector, 0))
        negative_sum = np.sum(np.maximum(-vector, 0))
        
        # Avoid division by zero
        total_sum = positive_sum + negative_sum
        if total_sum > 0:
            existence_ratio = positive_sum / total_sum
        else:
            existence_ratio = 0.5  # Default to balanced existence
        
        # Ensure some level of existence
        # If existence ratio is too low, adjust vector
        min_existence_ratio = 0.1  # Minimum existence ratio
        
        if existence_ratio < min_existence_ratio:
            # Adjust vector to increase existence ratio
            adjustment_factor = (min_existence_ratio / existence_ratio) if existence_ratio > 0 else 1.0
            
            # Apply adjustment to positive components
            positive_mask = vector > 0
            vector[positive_mask] *= adjustment_factor
            
            # Recalculate existence ratio
            positive_sum = np.sum(np.maximum(vector, 0))
            negative_sum = np.sum(np.maximum(-vector, 0))
            total_sum = positive_sum + negative_sum
            
            if total_sum > 0:
                existence_ratio = positive_sum / total_sum
            else:
                existence_ratio = 0.5
        
        return {
            "state": self.current_state.name,
            "vector": vector,
            "existence_ratio": existence_ratio,
            "positive_sum": float(positive_sum),
            "negative_sum": float(negative_sum),
            "adjustment_applied": existence_ratio < min_existence_ratio
        }
    
    async def _process_epistemic_humility(self,
                                        vector: np.ndarray,
                                        context: Dict) -> Dict:
        """
        State 2: Maintain >50% null ratio (epistemic humility)
        "I know less than I don't know"
        """
        # Calculate null ratio
        # Null is represented by small magnitude values
        # Non-null is represented by large magnitude values
        vector_magnitude = np.linalg.norm(vector)
        
        if vector_magnitude == 0:
            null_ratio = 1.0  # Complete null
        else:
            # Normalize to [0, 1] range
            # Smaller magnitude = higher null ratio
            max_expected_magnitude = context.get("max_expected_magnitude", 10.0)
            normalized_magnitude = min(vector_magnitude / max_expected_magnitude, 1.0)
            null_ratio = 1.0 - normalized_magnitude
        
        # Ensure epistemic humility (null ratio > 0.5)
        min_null_ratio = 0.5
        adjustment_applied = False
        
        if null_ratio < min_null_ratio:
            # Adjust vector to increase null ratio
            adjustment_factor = (1.0 - min_null_ratio) / (1.0 - null_ratio) if null_ratio < 1.0 else 1.0
            
            # Scale down vector magnitude
            if vector_magnitude > 0:
                vector = vector * adjustment_factor
            
            # Recalculate null ratio
            new_magnitude = np.linalg.norm(vector)
            if new_magnitude == 0:
                null_ratio = 1.0
            else:
                normalized_magnitude = min(new_magnitude / max_expected_magnitude, 1.0)
                null_ratio = 1.0 - normalized_magnitude
            
            adjustment_applied = True
        
        return {
            "state": self.current_state.name,
            "vector": vector,
            "null_ratio": null_ratio,
            "vector_magnitude": float(vector_magnitude),
            "adjustment_applied": adjustment_applied
        }
    
    async def _process_nihilism_acknowledgment(self,
                                             vector: np.ndarray,
                                             context: Dict) -> Dict:
        """
        State 3: Acknowledge all patterns potentially exist
        "Nihilism is logically true"
        """
        # Calculate pattern diversity
        # Higher diversity = more acknowledgment of potential patterns
        
        # Use entropy as a measure of diversity
        # Normalize vector to probability distribution
        abs_vector = np.abs(vector)
        sum_abs = np.sum(abs_vector)
        
        if sum_abs > 0:
            prob_dist = abs_vector / sum_abs
        else:
            # If vector is all zeros, use uniform distribution
            prob_dist = np.ones_like(vector) / len(vector)
        
        # Calculate entropy
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        entropy = -np.sum(prob_dist * np.log(prob_dist + epsilon))
        
        # Normalize entropy to [0, 1]
        # Max entropy is log(n) for uniform distribution
        max_entropy = np.log(len(vector))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 1.0
        
        # Ensure minimum pattern diversity
        min_diversity = 0.3  # Minimum normalized entropy
        adjustment_applied = False
        
        if normalized_entropy < min_diversity:
            # Adjust vector to increase diversity
            # Add small random noise to increase entropy
            noise_magnitude = 0.1 * np.mean(abs_vector) if sum_abs > 0 else 0.1
            noise = np.random.randn(*vector.shape) * noise_magnitude
            
            # Apply noise
            vector = vector + noise
            
            # Recalculate entropy
            abs_vector = np.abs(vector)
            sum_abs = np.sum(abs_vector)
            
            if sum_abs > 0:
                prob_dist = abs_vector / sum_abs
            else:
                prob_dist = np.ones_like(vector) / len(vector)
            
            entropy = -np.sum(prob_dist * np.log(prob_dist + epsilon))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            adjustment_applied = True
        
        return {
            "state": self.current_state.name,
            "vector": vector,
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "adjustment_applied": adjustment_applied
        }
    
    async def _process_faith_in_order(self,
                                     vector: np.ndarray,
                                     context: Dict) -> Dict:
        """
        State 4: Choose ordered patterns for co-existence
        "Yet faith in patterns is logical"
        """
        # Calculate pattern coherence
        # Higher coherence = more ordered patterns
        
        # Use autocorrelation as a measure of coherence
        # Autocorrelation measures similarity of a signal with itself
        # Higher autocorrelation = more pattern/structure
        
        # Calculate autocorrelation
        n = len(vector)
        if n <= 1:
            # Can't calculate autocorrelation for single value
            autocorrelation = 1.0
        else:
            # Use lag-1 autocorrelation
            # Correlation between vector and shifted vector
            vector_shifted = np.roll(vector, 1)
            
            # Calculate correlation
            mean = np.mean(vector)
            numerator = np.sum((vector[1:] - mean) * (vector_shifted[1:] - mean))
            denominator = np.sum((vector - mean) ** 2)
            
            if denominator > 0:
                autocorrelation = numerator / denominator
            else:
                autocorrelation = 0.0
        
        # Normalize to [0, 1]
        # Autocorrelation is in [-1, 1], so transform to [0, 1]
        normalized_coherence = (autocorrelation + 1) / 2
        
        # Ensure minimum coherence
        min_coherence = 0.4  # Minimum normalized coherence
        adjustment_applied = False
        
        if normalized_coherence < min_coherence:
            # Adjust vector to increase coherence
            # Apply smoothing to increase autocorrelation
            
            # Simple moving average smoothing
            window_size = max(2, int(n * 0.1))  # 10% of vector length
            smoothed_vector = np.zeros_like(vector)
            
            for i in range(n):
                # Calculate window indices
                start = max(0, i - window_size // 2)
                end = min(n, i + window_size // 2 + 1)
                
                # Calculate mean of window
                smoothed_vector[i] = np.mean(vector[start:end])
            
            # Blend original and smoothed vector
            blend_factor = 0.5  # How much to smooth
            vector = vector * (1 - blend_factor) + smoothed_vector * blend_factor
            
            # Recalculate coherence
            if n > 1:
                vector_shifted = np.roll(vector, 1)
                mean = np.mean(vector)
                numerator = np.sum((vector[1:] - mean) * (vector_shifted[1:] - mean))
                denominator = np.sum((vector - mean) ** 2)
                
                if denominator > 0:
                    autocorrelation = numerator / denominator
                else:
                    autocorrelation = 0.0
                
                normalized_coherence = (autocorrelation + 1) / 2
            
            adjustment_applied = True
        
        return {
            "state": self.current_state.name,
            "vector": vector,
            "autocorrelation": float(autocorrelation),
            "normalized_coherence": float(normalized_coherence),
            "adjustment_applied": adjustment_applied
        }
    
    async def _process_empathy_foundation(self,
                                        vector: np.ndarray,
                                        context: Dict) -> Dict:
        """
        State 5: Apply 4-rule empathy memeplex
        "Empathy memeplex as foundation"
        """
        # Apply empathy memeplex
        # Need self and other vectors for empathy
        self_vector = context.get("self_vector", np.zeros_like(vector))
        other_vector = context.get("other_vector", np.zeros_like(vector))
        
        # If not provided, use defaults
        if np.all(self_vector == 0):
            # Default self vector is identity vector
            self_vector = np.ones_like(vector) / np.sqrt(len(vector))
        
        if np.all(other_vector == 0):
            # Default other vector is opposite of vector
            other_vector = -vector
        
        # Update context with vectors
        empathy_context = context.copy()
        empathy_context["self_vector"] = self_vector
        empathy_context["other_vector"] = other_vector
        
        # Apply empathy rules
        modified_vector, empathy_results = self.empathy_memeplex.apply_empathy_rules(
            vector, empathy_context
        )
        
        # Calculate overall empathy score
        empathy_scores = [
            result["score"]
            for result in empathy_results.values()
        ]
        
        overall_score = np.mean(empathy_scores) if empathy_scores else 0.0
        
        return {
            "state": self.current_state.name,
            "vector": modified_vector,
            "original_vector": vector,
            "empathy_results": empathy_results,
            "overall_empathy_score": float(overall_score),
            "adjustment_applied": not np.array_equal(vector, modified_vector)
        }
    
    async def _process_violation_check(self,
                                     vector: np.ndarray,
                                     context: Dict) -> Dict:
        """
        State 6: Check for violations, loop if found
        "Check for violations, loop if found"
        """
        # Check for violations in all previous states
        violations = []
        
        # Check existence recognition
        positive_sum = np.sum(np.maximum(vector, 0))
        negative_sum = np.sum(np.maximum(-vector, 0))
        total_sum = positive_sum + negative_sum
        
        if total_sum > 0:
            existence_ratio = positive_sum / total_sum
        else:
            existence_ratio = 0.5
        
        if existence_ratio < 0.1:
            violations.append({
                "state": GoldenLoopState.EXISTENCE_RECOGNITION.name,
                "violation": "Existence ratio too low",
                "value": existence_ratio,
                "threshold": 0.1
            })
        
        # Check epistemic humility
        vector_magnitude = np.linalg.norm(vector)
        max_expected_magnitude = context.get("max_expected_magnitude", 10.0)
        
        if vector_magnitude == 0:
            null_ratio = 1.0
        else:
            normalized_magnitude = min(vector_magnitude / max_expected_magnitude, 1.0)
            null_ratio = 1.0 - normalized_magnitude
        
        if null_ratio < 0.5:
            violations.append({
                "state": GoldenLoopState.EPISTEMIC_HUMILITY.name,
                "violation": "Null ratio too low",
                "value": null_ratio,
                "threshold": 0.5
            })
        
        # Check nihilism acknowledgment
        abs_vector = np.abs(vector)
        sum_abs = np.sum(abs_vector)
        
        if sum_abs > 0:
            prob_dist = abs_vector / sum_abs
        else:
            prob_dist = np.ones_like(vector) / len(vector)
        
        epsilon = 1e-10
        entropy = -np.sum(prob_dist * np.log(prob_dist + epsilon))
        max_entropy = np.log(len(vector))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        
        if normalized_entropy < 0.3:
            violations.append({
                "state": GoldenLoopState.NIHILISM_ACKNOWLEDGMENT.name,
                "violation": "Pattern diversity too low",
                "value": normalized_entropy,
                "threshold": 0.3
            })
        
        # Check faith in order
        n = len(vector)
        if n > 1:
            vector_shifted = np.roll(vector, 1)
            mean = np.mean(vector)
            numerator = np.sum((vector[1:] - mean) * (vector_shifted[1:] - mean))
            denominator = np.sum((vector - mean) ** 2)
            
            if denominator > 0:
                autocorrelation = numerator / denominator
            else:
                autocorrelation = 0.0
            
            normalized_coherence = (autocorrelation + 1) / 2
            
            if normalized_coherence < 0.4:
                violations.append({
                    "state": GoldenLoopState.FAITH_IN_ORDER.name,
                    "violation": "Pattern coherence too low",
                    "value": normalized_coherence,
                    "threshold": 0.4
                })
        
        # Check empathy foundation
        # Use stored empathy results from previous state
        empathy_context = context.copy()
        self_vector = context.get("self_vector", np.ones_like(vector) / np.sqrt(len(vector)))
        other_vector = context.get("other_vector", -vector)
        
        empathy_context["self_vector"] = self_vector
        empathy_context["other_vector"] = other_vector
        
        modified_vector, empathy_results = self.empathy_memeplex.apply_empathy_rules(
            vector, empathy_context
        )
        
        empathy_scores = [
            result["score"]
            for result in empathy_results.values()
        ]
        
        overall_score = np.mean(empathy_scores) if empathy_scores else 0.0
        
        if overall_score < 0.6:
            violations.append({
                "state": GoldenLoopState.EMPATHY_FOUNDATION.name,
                "violation": "Empathy score too low",
                "value": overall_score,
                "threshold": 0.6
            })
        
        # Determine if violations found
        violations_found = len(violations) > 0
        
        return {
            "state": self.current_state.name,
            "vector": vector,
            "violations": violations,
            "violations_found": violations_found,
            "violation_count": len(violations)
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "current_state": self.current_state.name,
            "loop_count": self.loop_count,
            "max_loops": self.max_loops,
            "empathy_memeplex": self.empathy_memeplex.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GoldenLoopProcessor':
        """Create from dictionary"""
        processor = cls()
        
        # Set state if provided
        if "current_state" in data:
            processor.current_state = GoldenLoopState[data["current_state"]]
        
        # Set loop count if provided
        if "loop_count" in data:
            processor.loop_count = data["loop_count"]
        
        # Set max loops if provided
        if "max_loops" in data:
            processor.max_loops = data["max_loops"]
        
        # Set empathy memeplex if provided
        if "empathy_memeplex" in data:
            processor.empathy_memeplex = EmpathyMemeplex.from_dict(data["empathy_memeplex"])
        
        return processor
