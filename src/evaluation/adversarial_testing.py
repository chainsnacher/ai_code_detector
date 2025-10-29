"""
Adversarial testing framework for robustness evaluation.
Tests model robustness against various types of attacks and perturbations.
"""

import numpy as np
import pandas as pd
import re
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from collections import defaultdict

logger = logging.getLogger("ai_code_detector")

class AdversarialTester:
    """Tests model robustness against adversarial attacks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adversarial tester."""
        # Accept either a direct adversarial config or a parent config containing
        # an 'adversarial_testing' section (e.g., config.get_section('evaluation'))
        defaults = {
            'attack_types': ['substitution', 'insertion', 'deletion', 'reordering', 'formatting'],
            'perturbation_ratio': 0.1,
            'max_perturbations': 10,
            'random_seed': 42
        }

        # If user passed a larger config (e.g., evaluation section), allow nested key
        user_cfg = None
        if config is None:
            user_cfg = {}
        elif isinstance(config, dict) and 'adversarial_testing' in config:
            user_cfg = config.get('adversarial_testing') or {}
        else:
            user_cfg = config

        self.config = {**defaults, **(user_cfg or {})}

        # Seed RNGs using safe get
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        
        self.attack_results = {}
        self.robustness_metrics = {}
        # Optional callable to transform text samples into feature arrays for model prediction
        # e.g., feature_extractor(texts: List[str]) -> np.ndarray
        self.feature_extractor = None
        if isinstance(user_cfg, dict) and 'feature_extractor' in user_cfg:
            self.feature_extractor = user_cfg.get('feature_extractor')
    
    def test_model_robustness(self, model, X: np.ndarray, y: np.ndarray, 
                            attack_types: List[str] = None) -> Dict[str, Any]:
        """Test model robustness against various attacks."""
        if attack_types is None:
            attack_types = self.config['attack_types']
        
        logger.info("Starting adversarial testing")
        
        # Get original predictions
        original_predictions = model.predict(X)
        original_accuracy = np.mean(original_predictions == y)
        
        results = {
            'original_accuracy': original_accuracy,
            'attack_results': {},
            'robustness_summary': {}
        }
        
        # Test each attack type
        for attack_type in attack_types:
            logger.info(f"Testing {attack_type} attack")
            
            try:
                attack_results = self._test_attack_type(
                    model, X, y, attack_type, original_predictions
                )
                results['attack_results'][attack_type] = attack_results
                
            except Exception as e:
                logger.error(f"Error testing {attack_type} attack: {e}")
                results['attack_results'][attack_type] = {'error': str(e)}
        
        # Calculate overall robustness metrics
        results['robustness_summary'] = self._calculate_robustness_summary(results)
        
        return results
    
    def _test_attack_type(self, model, X: np.ndarray, y: np.ndarray, 
                         attack_type: str, original_predictions: np.ndarray) -> Dict[str, Any]:
        """Test a specific attack type."""
        attack_results = {
            'attack_type': attack_type,
            'perturbed_samples': [],
            'accuracy_drop': 0,
            'robustness_score': 0,
            'attack_success_rate': 0
        }
        
        # Generate perturbed samples
        perturbed_samples = []
        successful_attacks = 0
        
        for i, sample in enumerate(X):
            if isinstance(sample, str):
                # Text-based perturbation
                perturbed = self._perturb_text(sample, attack_type)
            else:
                # Feature-based perturbation
                perturbed = self._perturb_features(sample, attack_type)
            
            if perturbed is not None:
                perturbed_samples.append(perturbed)
                
                # Check if attack was successful
                original_pred = original_predictions[i]
                try:
                    if isinstance(perturbed, str):
                        # For text, we need to extract features first. Use provided feature_extractor if available.
                        if self.feature_extractor is not None:
                            try:
                                features = self.feature_extractor([perturbed])
                                perturbed_pred = model.predict(features)[0]
                            except Exception as e:
                                logger.warning(f"Feature extraction failed for perturbed text: {e}")
                                continue
                        else:
                            # No extractor available; skip prediction for text samples
                            logger.warning("No feature_extractor provided; skipping text-based perturbed prediction")
                            continue
                    else:
                        perturbed_pred = model.predict([perturbed])[0]

                    if perturbed_pred != original_pred:
                        successful_attacks += 1

                except Exception as e:
                    logger.warning(f"Error predicting perturbed sample {i}: {e}")
                    continue
            else:
                perturbed_samples.append(sample)  # Keep original if perturbation failed
        
        # Calculate metrics
        if perturbed_samples:
            try:
                # If perturbed samples are text, convert them to features first if extractor provided
                if isinstance(perturbed_samples[0], str):
                    if self.feature_extractor is None:
                        logger.warning("No feature_extractor provided; skipping perturbed text evaluation")
                        attack_results['error'] = 'no_feature_extractor_for_text'
                        return attack_results

                    try:
                        perturbed_features = self.feature_extractor(perturbed_samples)
                    except Exception as e:
                        logger.error(f"Feature extraction failed for perturbed samples: {e}")
                        attack_results['error'] = str(e)
                        return attack_results

                    if len(perturbed_features) == 0:
                        logger.error("Feature extractor returned empty array for perturbed samples")
                        attack_results['error'] = 'empty_feature_array'
                        return attack_results

                    perturbed_predictions = model.predict(perturbed_features)
                else:
                    # Ensure perturbed_samples is a proper array shape (list of numeric arrays)
                    try:
                        perturbed_array = np.array(perturbed_samples)
                        if perturbed_array.size == 0:
                            raise ValueError('empty perturbed array')
                    except Exception as e:
                        logger.error(f"Error preparing perturbed samples for prediction: {e}")
                        attack_results['error'] = str(e)
                        return attack_results

                    perturbed_predictions = model.predict(perturbed_array)

                perturbed_accuracy = np.mean(perturbed_predictions == y)
                accuracy_drop = original_predictions.mean() - perturbed_accuracy
                robustness_score = perturbed_accuracy / original_predictions.mean() if original_predictions.mean() > 0 else 0
                attack_success_rate = successful_attacks / len(perturbed_samples)
                
                attack_results.update({
                    'perturbed_samples': perturbed_samples[:5],  # Store first 5 for inspection
                    'accuracy_drop': accuracy_drop,
                    'robustness_score': robustness_score,
                    'attack_success_rate': attack_success_rate,
                    'perturbed_accuracy': perturbed_accuracy
                })
                
            except Exception as e:
                logger.error(f"Error evaluating perturbed samples: {e}")
                attack_results['error'] = str(e)
        
        return attack_results
    
    def _perturb_text(self, text: str, attack_type: str) -> Optional[str]:
        """Apply text-based perturbations."""
        try:
            if attack_type == 'substitution':
                return self._substitution_attack(text)
            elif attack_type == 'insertion':
                return self._insertion_attack(text)
            elif attack_type == 'deletion':
                return self._deletion_attack(text)
            elif attack_type == 'reordering':
                return self._reordering_attack(text)
            elif attack_type == 'formatting':
                return self._formatting_attack(text)
            else:
                return text
        except Exception as e:
            logger.warning(f"Error applying {attack_type} attack: {e}")
            return None
    
    def _substitution_attack(self, text: str) -> str:
        """Substitute characters or words in text."""
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        
        # Character-level substitution
        if random.random() < 0.5:
            # Substitute random characters
            chars = list(text)
            n_substitutions = max(1, int(len(chars) * perturbation_ratio))
            
            for _ in range(n_substitutions):
                if chars:
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            
            return ''.join(chars)
        else:
            # Substitute keywords
            keywords = ['def', 'class', 'import', 'if', 'for', 'while', 'return']
            words = text.split()
            
            for i, word in enumerate(words):
                if word in keywords and random.random() < perturbation_ratio:
                    words[i] = random.choice(keywords)
            
            return ' '.join(words)
    
    def _insertion_attack(self, text: str) -> str:
        """Insert random characters or words."""
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        
        # Character-level insertion
        if random.random() < 0.5:
            chars = list(text)
            n_insertions = max(1, int(len(chars) * perturbation_ratio))
            
            for _ in range(n_insertions):
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
            
            return ''.join(chars)
        else:
            # Word-level insertion
            words = text.split()
            n_insertions = max(1, int(len(words) * perturbation_ratio))
            
            for _ in range(n_insertions):
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(['# comment', 'pass', 'None', 'True', 'False']))
            
            return ' '.join(words)
    
    def _deletion_attack(self, text: str) -> str:
        """Delete random characters or words."""
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        
        # Character-level deletion
        if random.random() < 0.5:
            chars = list(text)
            n_deletions = max(1, int(len(chars) * perturbation_ratio))
            
            for _ in range(min(n_deletions, len(chars))):
                if chars:
                    idx = random.randint(0, len(chars) - 1)
                    chars.pop(idx)
            
            return ''.join(chars)
        else:
            # Word-level deletion
            words = text.split()
            n_deletions = max(1, int(len(words) * perturbation_ratio))
            
            for _ in range(min(n_deletions, len(words))):
                if words:
                    idx = random.randint(0, len(words) - 1)
                    words.pop(idx)
            
            return ' '.join(words)
    
    def _reordering_attack(self, text: str) -> str:
        """Reorder lines or words in text."""
        lines = text.split('\n')
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        
        if len(lines) > 1:
            # Reorder lines
            n_swaps = max(1, int(len(lines) * perturbation_ratio))
            
            for _ in range(n_swaps):
                if len(lines) > 1:
                    i, j = random.sample(range(len(lines)), 2)
                    lines[i], lines[j] = lines[j], lines[i]
        else:
            # Reorder words within line
            words = text.split()
            if len(words) > 1:
                n_swaps = max(1, int(len(words) * self.config['perturbation_ratio']))
                
                for _ in range(n_swaps):
                    if len(words) > 1:
                        i, j = random.sample(range(len(words)), 2)
                        words[i], words[j] = words[j], words[i]
                
                return ' '.join(words)
        
        return '\n'.join(lines)
    
    def _formatting_attack(self, text: str) -> str:
        """Modify formatting of text."""
        # Change indentation
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        if random.random() < 0.5:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip():  # Non-empty line
                    if random.random() < perturbation_ratio:
                        # Add or remove spaces
                        if random.random() < 0.5:
                            lines[i] = '    ' + line  # Add indentation
                        else:
                            lines[i] = line.lstrip()  # Remove indentation
            return '\n'.join(lines)
        else:
            # Change line breaks
            return text.replace('\n', ' ').replace('  ', ' ')
    
    def _perturb_features(self, features: np.ndarray, attack_type: str) -> np.ndarray:
        """Apply feature-based perturbations."""
        perturbation_ratio = self.config.get('perturbation_ratio', 0.1)
        perturbed = features.copy()
        
        if attack_type == 'substitution':
            # Add noise to features
            noise = np.random.normal(0, 0.1, features.shape)
            mask = np.random.random(features.shape) < perturbation_ratio
            perturbed[mask] += noise[mask]
            
        elif attack_type == 'insertion':
            # Add random features
            n_new_features = max(1, int(len(features) * perturbation_ratio))
            new_features = np.random.normal(0, 1, n_new_features)
            perturbed = np.concatenate([perturbed, new_features])
            
        elif attack_type == 'deletion':
            # Remove random features
            n_remove = max(1, int(len(features) * perturbation_ratio))
            indices_to_remove = np.random.choice(len(features), n_remove, replace=False)
            perturbed = np.delete(perturbed, indices_to_remove)
            
        elif attack_type == 'reordering':
            # Shuffle features
            n_swaps = max(1, int(len(features) * perturbation_ratio))
            for _ in range(n_swaps):
                i, j = np.random.choice(len(features), 2, replace=False)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        return perturbed
    
    def _calculate_robustness_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall robustness summary."""
        summary = {
            'original_accuracy': results['original_accuracy'],
            'average_robustness_score': 0,
            'worst_attack': None,
            'best_attack': None,
            'overall_robustness': 0
        }
        
        robustness_scores = []
        attack_types = []
        
        for attack_type, attack_results in results['attack_results'].items():
            if 'error' not in attack_results and 'robustness_score' in attack_results:
                robustness_scores.append(attack_results['robustness_score'])
                attack_types.append(attack_type)
        
        if robustness_scores:
            summary['average_robustness_score'] = np.mean(robustness_scores)
            summary['overall_robustness'] = np.mean(robustness_scores)
            
            # Find worst and best attacks
            worst_idx = np.argmin(robustness_scores)
            best_idx = np.argmax(robustness_scores)
            
            summary['worst_attack'] = {
                'type': attack_types[worst_idx],
                'score': robustness_scores[worst_idx]
            }
            summary['best_attack'] = {
                'type': attack_types[best_idx],
                'score': robustness_scores[best_idx]
            }
        
        return summary
    
    def test_ensemble_robustness(self, ensemble_model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Test ensemble model robustness."""
        logger.info("Testing ensemble model robustness")
        
        # Test individual models in ensemble
        individual_results = {}
        if hasattr(ensemble_model, 'base_models'):
            for model_name, model in ensemble_model.base_models.items():
                try:
                    model_results = self.test_model_robustness(model, X, y)
                    individual_results[model_name] = model_results
                except Exception as e:
                    logger.error(f"Error testing {model_name}: {e}")
                    individual_results[model_name] = {'error': str(e)}
        
        # Test ensemble as a whole
        ensemble_results = self.test_model_robustness(ensemble_model, X, y)
        
        return {
            'individual_results': individual_results,
            'ensemble_results': ensemble_results,
            'ensemble_advantage': self._calculate_ensemble_advantage(individual_results, ensemble_results)
        }
    
    def _calculate_ensemble_advantage(self, individual_results: Dict[str, Any], 
                                    ensemble_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble advantage over individual models."""
        advantage = {}
        
        if 'robustness_summary' in ensemble_results:
            ensemble_robustness = ensemble_results['robustness_summary']['overall_robustness']
            
            individual_robustness = []
            for model_name, results in individual_results.items():
                if 'error' not in results and 'robustness_summary' in results:
                    individual_robustness.append(results['robustness_summary']['overall_robustness'])
            
            if individual_robustness:
                advantage['vs_best_individual'] = ensemble_robustness - max(individual_robustness)
                advantage['vs_average_individual'] = ensemble_robustness - np.mean(individual_robustness)
                advantage['vs_worst_individual'] = ensemble_robustness - min(individual_robustness)
        
        return advantage
    
    def generate_robustness_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable robustness report."""
        report = []
        report.append("=" * 60)
        report.append("ADVERSARIAL ROBUSTNESS TESTING REPORT")
        report.append("=" * 60)
        
        # Original performance
        if 'original_accuracy' in results:
            report.append(f"Original Accuracy: {results['original_accuracy']:.4f}")
        
        # Attack results
        report.append("\nATTACK RESULTS:")
        report.append("-" * 30)
        
        for attack_type, attack_results in results.get('attack_results', {}).items():
            if 'error' not in attack_results:
                report.append(f"\n{attack_type.upper()} Attack:")
                report.append(f"  Robustness Score: {attack_results.get('robustness_score', 0):.4f}")
                report.append(f"  Accuracy Drop: {attack_results.get('accuracy_drop', 0):.4f}")
                report.append(f"  Attack Success Rate: {attack_results.get('attack_success_rate', 0):.4f}")
            else:
                report.append(f"\n{attack_type.upper()} Attack: ERROR - {attack_results['error']}")
        
        # Overall robustness
        if 'robustness_summary' in results:
            summary = results['robustness_summary']
            report.append(f"\nOVERALL ROBUSTNESS:")
            report.append("-" * 20)
            report.append(f"Average Robustness Score: {summary.get('average_robustness_score', 0):.4f}")
            report.append(f"Overall Robustness: {summary.get('overall_robustness', 0):.4f}")
            
            if summary.get('worst_attack'):
                worst = summary['worst_attack']
                report.append(f"Worst Attack: {worst['type']} (Score: {worst['score']:.4f})")
            
            if summary.get('best_attack'):
                best = summary['best_attack']
                report.append(f"Best Attack: {best['type']} (Score: {best['score']:.4f})")
        
        report.append("\n" + "=" * 60)
        
        return '\n'.join(report)
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save adversarial testing results."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save human-readable report
        report_path = save_path.with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(self.generate_robustness_report(results))
        
        logger.info(f"Adversarial testing results saved to {save_path}")
        logger.info(f"Robustness report saved to {report_path}")
