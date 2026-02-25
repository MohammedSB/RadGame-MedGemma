import json
import csv
import os
import sys
import re

from CRIMSON.prompt_parts import build_prompt as _build_evaluation_prompt_fn

from transformers import pipeline
import torch


class CRIMSONScore:
    """
    Calculate CRIMSON scores using a HuggingFace model to evaluate
    radiology report generation quality.
    """

    def __init__(
        self,
        model_name=None,
        device=None,
        torch_dtype=None,
    ):
        """
        Initialize CRIMSON scorer.

        Args:
            model_name: HuggingFace model name or local path (required).
            device: Device for inference. Defaults to "cuda".
            torch_dtype: PyTorch dtype for model weights (e.g., torch.float16, torch.bfloat16).
                        If None, uses torch.float16.
        """
        if not model_name:
            raise ValueError("model_name must be provided (HuggingFace model ID or local path).")
        self.model_name = model_name
        self.device = device or "cuda"
        self.torch_dtype = torch_dtype or torch.float16

        print(f"Loading HuggingFace model: {self.model_name}")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        print("Model loaded.")
    
    
    def _chat_completion(self, prompt):
        """
        Run inference via the HuggingFace text-generation pipeline.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        messages = [
            {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
            {"role": "user", "content": prompt + "\nPlease respond with valid JSON only."},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=2048,
            temperature=0,
        )
        return outputs[0]["generated_text"][-1]["content"]
    
    def _build_evaluation_prompt(
        self, 
        reference_findings, 
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Build the evaluation prompt for CRIMSON scoring.
        Delegates to prompt_parts.build_prompt for composable prompt assembly.
        
        Args:
            reference_findings: Ground truth findings (string or list of strings)
            predicted_findings: Predicted findings (string or list of strings)
            patient_context: Optional patient context (age, indication, etc.)
            include_guidelines: If False, disables significance examples,
                attribute severity guidelines, and context guidelines.
            
        Returns:
            Formatted prompt string
        """
        return _build_evaluation_prompt_fn(
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )
    
    def evaluate(
        self,
        reference_findings,
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Evaluate findings and calculate CRIMSON score.
        
        Args:
            reference_findings: Ground truth findings
            predicted_findings: Model-generated findings  
            patient_context: Optional patient context
            include_guidelines: If False, disables guidelines in the prompt
            
        Returns:
            Dictionary containing:
                - raw_evaluation: Raw model output with errors by category
                - error_counts: Counts of each error type
                - metrics: Individual metric components
                - crimson_score: Final CRIMSON score
        """
        # Get evaluation from model
        prompt = self._build_evaluation_prompt(
            reference_findings, 
            predicted_findings, 
            patient_context,
            include_guidelines=include_guidelines,
        )
        
        response = self._chat_completion(prompt)
        
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {response}")
        
        # Calculate CRIMSON score
        crimson_result = self._calculate_crimson(evaluation)
        
        return crimson_result
    
    def _calculate_crimson(
        self,
        evaluation
    ):
        """
        Calculate CRIMSON score from evaluation results.
        
        Args:
            evaluation: Parsed evaluation with matched findings and errors
            
        Returns:
            Dictionary with scores and metrics
        """
        # Extract error counts with clinical significance weighting
        errors = evaluation.get("errors", {})
        matched = evaluation.get("matched_findings", [])
        reference_findings_list = evaluation.get("reference_findings", [])
        predicted_findings_list = evaluation.get("predicted_findings", [])
        
        # Weight mapping for clinical significance (findings)
        significance_weights = {
            "urgent": 1.0,
            "actionable_not_urgent": 0.5,
            "not_actionable_not_urgent": 0.25,
            "benign_expected": 0.0,
        }
        
        # Separate weight mapping for attribute error severity
        attribute_severity_weights = {
            "significant": 0.5,
            "negligible": 0.0,
        }
        
        def calculate_weighted_count(error_list, weights=significance_weights, key="clinical_significance"):
            """Calculate weighted count based on significance/severity."""
            return sum(weights[error[key]] for error in error_list)
        
        # Build ID-based weight maps for direct lookup
        ref_weight_by_id = {
            ref["id"]: significance_weights[ref["clinical_significance"]]
            for ref in reference_findings_list
        }
        pred_weight_by_id = {
            pred["id"]: significance_weights[pred["clinical_significance"]]
            for pred in predicted_findings_list
        }
        
        # E_false: weight derived from predicted_findings by ID
        E_false = sum(pred_weight_by_id.get(f_id, 0.0) for f_id in errors.get("false_findings", []))
        
        # E_miss: weight derived from reference_findings by ID
        E_miss = sum(ref_weight_by_id.get(m_id, 0.0) for m_id in errors.get("missing_findings", []))
        
        # Count individual error types within attribute_errors for analysis
        attr_errors = errors.get("attribute_errors", [])
        n_location = sum(1 for e in attr_errors if "location" in e.get("error_types", []))
        n_severity = sum(1 for e in attr_errors if "severity" in e.get("error_types", []))
        n_descriptor = sum(1 for e in attr_errors if "descriptor" in e.get("error_types", []))
        n_measurement = sum(1 for e in attr_errors if "measurement" in e.get("error_types", []))
        n_certainty = sum(1 for e in attr_errors if "certainty" in e.get("error_types", []))
        n_unspecific = sum(1 for e in attr_errors if "unspecific" in e.get("error_types", []))
        n_overinterpretation = sum(1 for e in attr_errors if "overinterpretation" in e.get("error_types", []))
        n_temporal = sum(1 for e in attr_errors if "temporal" in e.get("error_types", []))
        
        # Group attribute errors by ref_id for partial credit calculation
        attr_errors_by_ref_id = {}
        for err in attr_errors:
            ref_id = err["ref_id"]
            if ref_id not in attr_errors_by_ref_id:
                attr_errors_by_ref_id[ref_id] = []
            attr_errors_by_ref_id[ref_id].append(err)
        
        # N_G: weighted total of positive findings in reference
        N_G = calculate_weighted_count(reference_findings_list)
        if N_G == 0 and not reference_findings_list:
            N_G = len(matched) + E_miss
        
        # Calculate penalty and correct counts
        # Attribute errors reduce correct credit rather than adding to penalty
        E_penalty = E_false
        
        # Calculate correct as weighted sum of matched findings with partial credit for attribute errors
        # Deduplicate by ref_id: each reference finding contributes weight only once,
        # even if it matches multiple predicted findings (one-to-many).
        matched_ref_ids = set()
        correct = 0.0
        for m in matched:
            ref_id = m["ref_id"]
            if ref_id in matched_ref_ids:
                continue  # Already counted this reference finding
            matched_ref_ids.add(ref_id)
            base_weight = ref_weight_by_id.get(ref_id, 0.0)
            
            # Check if this finding has attribute errors
            finding_attr_errors = attr_errors_by_ref_id.get(ref_id, [])
            
            if not finding_attr_errors:
                # Perfect match, full credit
                correct += base_weight
            else:
                # Partial credit: credit_factor = base_weight / (base_weight + sum_of_error_weights)
                sum_error_weights = sum(
                    attribute_severity_weights[err["severity"]] for err in finding_attr_errors
                )
                credit_factor = base_weight / (base_weight + sum_error_weights) if (base_weight + sum_error_weights) > 0 else 0.0
                correct += base_weight * credit_factor
        
        errors_more_than_correct = E_penalty - correct
        
        # Calculate score S
        if N_G == 0:
            # Normal GT: perfect if no errors, else penalize
            S = 1.0 if E_penalty == 0 and E_miss == 0 else -(E_penalty + E_miss + 1)
        else:
            S = (correct - E_penalty) / N_G
        
        # Calculate CRIMSON score
        if S >= 0:
            crimson = S
        else:
            if errors_more_than_correct > 0:
                crimson = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
            else:
                crimson = 0
        
        return {
            "raw_evaluation": evaluation,
            "error_counts": {
                "false_findings": len(errors.get("false_findings", [])),
                "missing_findings": len(errors.get("missing_findings", [])),
                "attribute_errors": len(attr_errors),
                "location_errors": n_location,
                "severity_errors": n_severity,
                "descriptor_errors": n_descriptor,
                "measurement_errors": n_measurement,
                "certainty_errors": n_certainty,
                "unspecific_errors": n_unspecific,
                "overinterpretation_errors": n_overinterpretation,
                "temporal_errors": n_temporal
            },
            "weighted_error_counts": {
                "false_findings": E_false,
                "missing_findings": E_miss,
                "attribute_errors": calculate_weighted_count(attr_errors, attribute_severity_weights, "severity")
            },
            "metrics": {
                "N_G": N_G,  # Number of ground truth findings
                "E_penalty": E_penalty,
                "correct": correct,
                "errors_more_than_correct": errors_more_than_correct,
                "S": S
            },
            "crimson_score": round(crimson, 4)
        }