# ==== ULTRA-ADVANCED SELECTION MODULE 2025 ====
# State-of-the-art answer selection with multi-signal aggregation, ensemble methods,
# semantic similarity, confidence calibration, uncertainty quantification, and optimized performance

import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
import time
import warnings

# Advanced optional dependencies (graceful degradation)
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, ValueError, Exception) as e:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn(f"sentence-transformers not available. Semantic similarity disabled. Reason: {type(e).__name__}")

try:
    import scipy.stats as stats
    from scipy.spatial.distance import cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Advanced statistics disabled. Install: pip install scipy")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. Ensemble scoring disabled. Install: pip install scikit-learn")

# ==== Performance Configuration ====
CACHE_SIZE = 2048
MAX_WORKERS = 8
EMBEDDING_BATCH_SIZE = 32
SIMILARITY_THRESHOLD = 0.85
UNCERTAINTY_THRESHOLD = 0.3

# ==== Performance Decorators ====
def timed_cache(seconds: int = 300):
    """Time-based LRU cache with TTL"""
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in cache and (current_time - cache_times[key]) < seconds:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time
            
            # Cleanup old entries
            if len(cache) > CACHE_SIZE:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        return wrapper
    return decorator

def parallel_execution(max_workers: int = MAX_WORKERS):
    """Decorator for parallel execution of function on iterable inputs"""
    def decorator(func):
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            if len(items) <= 1:
                return [func(item, *args, **kwargs) for item in items]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(func, item, *args, **kwargs) for item in items]
                return [f.result() for f in futures]
        return wrapper
    return decorator

# ==== Data Structures ====
@dataclass
class SelectionSignals:
    """Aggregated signals for answer selection"""
    trace_quality: float = 0.0
    keyword_score: float = 0.0
    semantic_similarity: float = 0.0
    tool_confidence: float = 0.0
    consistency_score: float = 0.0
    meta_score: float = 0.0
    uncertainty: float = 1.0
    
    def aggregate(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Weighted aggregation of all signals"""
        default_weights = {
            'trace_quality': 0.20,
            'keyword_score': 0.15,
            'semantic_similarity': 0.25,
            'tool_confidence': 0.15,
            'consistency_score': 0.15,
            'meta_score': 0.10
        }
        w = weights or default_weights
        
        score = (
            w['trace_quality'] * self.trace_quality +
            w['keyword_score'] * self.keyword_score +
            w['semantic_similarity'] * self.semantic_similarity +
            w['tool_confidence'] * self.tool_confidence +
            w['consistency_score'] * self.consistency_score +
            w['meta_score'] * self.meta_score
        )
        
        # Penalize high uncertainty
        score *= (1.0 - self.uncertainty * 0.5)
        return score

@dataclass
class SelectionResult:
    """Enhanced result with confidence and reasoning"""
    selected_index: int  # 1-based
    confidence: float
    uncertainty: float
    all_scores: List[float]
    signals: List[SelectionSignals]
    tie_detected: bool = False
    reasoning: str = ""
    alternative_indices: List[int] = field(default_factory=list)

# ==== Semantic Similarity Engine ====
class SemanticSimilarityEngine:
    """Fast semantic similarity using sentence transformers with caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with efficient model (384-dim embeddings, <50ms inference)"""
        self.model = None
        self.model_name = model_name
        self.embedding_cache = {}
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Type guard: ensure SentenceTransformer is available
                if 'SentenceTransformer' in globals():
                    self.model = SentenceTransformer(model_name)  # type: ignore
                    self.model.max_seq_length = 256  # Optimize for speed
            except Exception as e:
                warnings.warn(f"Failed to load SentenceTransformer: {e}")
    
    @timed_cache(seconds=600)
    def _get_embedding_cached(self, text: str) -> Optional[np.ndarray]:
        """Cached embedding computation"""
        if not self.model:
            return None
        
        # Use hash for cache key to handle long texts
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self.embedding_cache:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            self.embedding_cache[text_hash] = embedding
            
            # Cleanup cache if too large
            if len(self.embedding_cache) > CACHE_SIZE:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
        
        return self.embedding_cache[text_hash]
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix efficiently"""
        if not self.model or len(texts) <= 1:
            return np.eye(len(texts))
        
        try:
            # Batch encoding for speed
            embeddings = self.model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, 
                                          convert_to_numpy=True, show_progress_bar=False)
            
            # Vectorized cosine similarity
            if HAS_SENTENCE_TRANSFORMERS and 'util' in globals():
                similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()  # type: ignore
            else:
                # Fallback: manual cosine similarity
                from numpy.linalg import norm
                similarity_matrix = np.dot(embeddings, embeddings.T) / (norm(embeddings, axis=1)[:, None] * norm(embeddings, axis=1)[None, :])
            return similarity_matrix
        except Exception as e:
            warnings.warn(f"Similarity computation failed: {e}")
            return np.eye(len(texts))
    
    def compute_self_consistency(self, traces: List[str], top_k: int = 3) -> float:
        """Measure self-consistency across traces via similarity clustering"""
        if len(traces) <= 1:
            return 1.0
        
        sim_matrix = self.compute_similarity_matrix(traces)
        
        # Average top-k similarities (exclude self-similarity)
        np.fill_diagonal(sim_matrix, 0)
        top_k_sims = []
        for i in range(len(traces)):
            row_sims = sim_matrix[i]
            top_sims = np.partition(row_sims, -min(top_k, len(traces)-1))[-min(top_k, len(traces)-1):]
            top_k_sims.append(np.mean(top_sims))
        
        return float(np.mean(top_k_sims))

# ==== Ensemble Scoring Engine ====
class EnsembleScorer:
    """Multi-model ensemble scoring with distance-aware voting"""
    
    def __init__(self, use_ml_ensemble: bool = True):
        self.use_ml_ensemble = use_ml_ensemble and HAS_SKLEARN
        self.models = []
        self.scaler = None
        
        if self.use_ml_ensemble and 'RandomForestRegressor' in globals() and 'GradientBoostingRegressor' in globals():
            # Lightweight ensemble of fast models
            self.models = [
                RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),  # type: ignore
                GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)  # type: ignore
            ]
            if 'StandardScaler' in globals():
                self.scaler = StandardScaler()  # type: ignore
    
    def distance_aware_voting(self, scores: List[float], similarities: np.ndarray) -> List[float]:
        """Weight votes by semantic similarity (closer variants have more influence)"""
        if len(scores) <= 1:
            return scores
        
        n = len(scores)
        weighted_scores = np.zeros(n)
        
        for i in range(n):
            # Weight this score by similarity to all others
            similarity_weights = similarities[i]
            similarity_weights = similarity_weights / (np.sum(similarity_weights) + 1e-8)
            
            # Apply weighted averaging
            weighted_scores[i] = np.sum(similarity_weights * np.array(scores))
        
        return weighted_scores.tolist()
    
    def ensemble_predict(self, features: np.ndarray) -> float:
        """Ensemble prediction from multiple models"""
        if not self.use_ml_ensemble or not self.models:
            return 0.0
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(features.reshape(1, -1))[0]
                predictions.append(pred)
            except Exception:
                continue
        
        return float(np.mean(predictions)) if predictions else 0.0

# ==== Confidence Calibration ====
class ConfidenceCalibrator:
    """Advanced confidence calibration and uncertainty quantification"""
    
    @staticmethod
    def calibrate_confidence(raw_score: float, variance: float, sample_size: int) -> Tuple[float, float]:
        """
        Calibrate confidence using Bayesian approach
        Returns: (calibrated_confidence, uncertainty)
        """
        # Bayesian credible interval
        if sample_size > 1 and variance > 0:
            # Standard error
            se = np.sqrt(variance / sample_size)
            
            # Confidence interval (95%)
            ci_lower = raw_score - 1.96 * se
            ci_upper = raw_score + 1.96 * se
            
            # Uncertainty as relative width of CI
            uncertainty = (ci_upper - ci_lower) / (2.0 * max(abs(raw_score), 0.1))
            
            # Calibrated confidence (shrink towards mean)
            calibrated = raw_score * (1.0 - uncertainty * 0.3)
        else:
            calibrated = raw_score
            uncertainty = 0.5  # High uncertainty for single sample
        
        return float(np.clip(calibrated, 0.0, 1.0)), float(np.clip(uncertainty, 0.0, 1.0))
    
    @staticmethod
    def monte_carlo_dropout_uncertainty(scores: List[float], n_samples: int = 10) -> float:
        """Estimate uncertainty via score variance"""
        if len(scores) < 2:
            return 0.5
        
        variance = np.var(scores)
        mean_score = np.mean(scores)
        
        # Coefficient of variation as uncertainty measure
        cv = np.sqrt(variance) / (abs(mean_score) + 1e-8)
        return float(np.clip(cv, 0.0, 1.0))

# ==== Trace Quality Analyzer ====
class TraceQualityAnalyzer:
    """Advanced heuristics for reasoning trace quality assessment"""
    
    def __init__(self):
        self.optimal_length_range = (50, 500)  # chars
        self.redundancy_threshold = 0.7
    
    @timed_cache(seconds=300)
    def analyze_trace_quality(self, trace: str) -> Dict[str, float]:
        """Multi-dimensional trace quality analysis"""
        metrics = {}
        
        # 1. Length regularization (penalize too short/long traces)
        length = len(trace)
        if length < self.optimal_length_range[0]:
            metrics['length_score'] = length / self.optimal_length_range[0]
        elif length > self.optimal_length_range[1]:
            metrics['length_score'] = self.optimal_length_range[1] / length
        else:
            metrics['length_score'] = 1.0
        
        # 2. Structural quality (presence of reasoning steps)
        step_patterns = [
            r'\bstep\s+\d+', r'\bfirst\b', r'\bsecond\b', r'\bthen\b', 
            r'\btherefore\b', r'\bthus\b', r'\bhence\b'
        ]
        structure_score = sum(1 for p in step_patterns if re.search(p, trace, re.I))
        metrics['structure_score'] = min(structure_score / 5.0, 1.0)
        
        # 3. Redundancy detection (penalize repetitive content)
        sentences = re.split(r'[.!?]+', trace)
        if len(sentences) > 1:
            unique_ratio = len(set(sentences)) / len(sentences)
            metrics['redundancy_score'] = unique_ratio
        else:
            metrics['redundancy_score'] = 1.0
        
        # 4. Logical coherence (presence of connectives)
        connectives = ['because', 'therefore', 'thus', 'hence', 'since', 'given', 'so']
        coherence_score = sum(1 for c in connectives if c in trace.lower())
        metrics['coherence_score'] = min(coherence_score / 3.0, 1.0)
        
        # 5. Numerical precision (presence of specific values)
        numbers = re.findall(r'\b\d+\.?\d*\b', trace)
        metrics['precision_score'] = min(len(numbers) / 3.0, 1.0)
        
        # Aggregate
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics

# ==== Tool Confidence Extractor ====
class ToolConfidenceExtractor:
    """Extract and aggregate confidence from tool outputs"""
    
    @staticmethod
    def extract_tool_signals(trace: str, tool_outputs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Extract confidence signals from tool usage in trace"""
        signals = {
            'tool_count': 0,
            'tool_success_rate': 0.0,
            'tool_confidence': 0.0,
            'tool_agreement': 0.0
        }
        
        # Pattern matching for tool calls in trace
        tool_patterns = [
            r'tool:\s*\w+',
            r'using\s+tool',
            r'calculator|search|query|api|function',
            r'result:\s*[\d\w]+'
        ]
        
        tool_mentions = sum(len(re.findall(p, trace, re.I)) for p in tool_patterns)
        signals['tool_count'] = min(tool_mentions / 3.0, 1.0)
        
        # If tool outputs provided, extract structured confidence
        if tool_outputs:
            successes = sum(1 for t in tool_outputs if t.get('success', False))
            signals['tool_success_rate'] = successes / len(tool_outputs) if tool_outputs else 0.0
            
            confidences = [t.get('confidence', 0.5) for t in tool_outputs]
            signals['tool_confidence'] = np.mean(confidences) if confidences else 0.0
            
            # Agreement: variance in outputs
            if len(tool_outputs) > 1:
                outputs_text = [str(t.get('output', '')) for t in tool_outputs]
                unique_outputs = len(set(outputs_text))
                signals['tool_agreement'] = 1.0 - (unique_outputs - 1) / len(tool_outputs)
        
        return signals

# ==== MAIN ULTRA-ADVANCED SELECTION MODULE ====
class SelectionModule:
    """
    Ultra-advanced answer selection with:
    - Multi-signal aggregation (trace, tools, keywords, semantics)
    - Ensemble methods and distance-aware voting
    - Confidence calibration and uncertainty quantification
    - Parallel processing and caching
    - Tie-breaking and fallback strategies
    """
    
    def __init__(self, 
                 use_meta_model: bool = False,
                 meta_model: Optional[Callable] = None,
                 keywords: Optional[Dict[str, List[str]]] = None,
                 use_semantic_similarity: bool = True,
                 use_ensemble: bool = True,
                 aggregation_weights: Optional[Dict[str, float]] = None,
                 parallel_execution: bool = True,
                 calibrate_confidence: bool = True):
        """
        Initialize ultra-advanced selection module.
        
        Args:
            use_meta_model: Use external meta-model for scoring
            meta_model: Callable(trace, option) -> float
            keywords: Custom keyword patterns for rule-based scoring
            use_semantic_similarity: Enable semantic similarity analysis
            use_ensemble: Enable ensemble voting methods
            aggregation_weights: Custom weights for signal aggregation
            parallel_execution: Enable parallel processing
            calibrate_confidence: Enable confidence calibration
        """
        self.use_meta_model = use_meta_model
        self.meta_model = meta_model
        self.use_semantic = use_semantic_similarity and HAS_SENTENCE_TRANSFORMERS
        self.use_ensemble = use_ensemble
        self.parallel = parallel_execution
        self.calibrate = calibrate_confidence
        
        # Keywords for rule-based scoring
        self.keywords = keywords or {
            "positive": [
                "correct", "valid answer", "consistent", "matches all conditions",
                "thus this is the answer", "final answer", "this must be", "confirmed",
                "verified", "satisfies", "optimal", "best option", "clearly",
                "undoubtedly", "definitely", "conclusively"
            ],
            "negative": [
                "contradicts", "invalid", "impossible", "does not fit", 
                "cannot be", "inconsistent", "ruled out", "incorrect",
                "fails", "violates", "conflicts", "wrong", "error",
                "unlikely", "doubtful", "questionable"
            ],
            "uncertainty": [
                "maybe", "perhaps", "possibly", "might", "could be",
                "unclear", "ambiguous", "uncertain", "not sure"
            ]
        }
        
        # Initialize sub-modules
        self.semantic_engine = SemanticSimilarityEngine() if self.use_semantic else None
        self.ensemble_scorer = EnsembleScorer(use_ml_ensemble=self.use_ensemble)
        self.calibrator = ConfidenceCalibrator()
        self.trace_analyzer = TraceQualityAnalyzer()
        self.tool_extractor = ToolConfidenceExtractor()
        
        # Aggregation weights
        self.aggregation_weights = aggregation_weights
        
        # Performance tracking
        self.stats = {
            'total_selections': 0,
            'avg_confidence': 0.0,
            'avg_uncertainty': 0.0,
            'tie_rate': 0.0
        }
    
    def _compute_keyword_signals(self, trace: str) -> Dict[str, float]:
        """Advanced keyword-based scoring with pattern matching"""
        positive_pat = re.compile('|'.join([re.escape(k) for k in self.keywords["positive"]]), re.I)
        negative_pat = re.compile('|'.join([re.escape(k) for k in self.keywords["negative"]]), re.I)
        uncertainty_pat = re.compile('|'.join([re.escape(k) for k in self.keywords["uncertainty"]]), re.I)
        
        pos_matches = len(positive_pat.findall(trace))
        neg_matches = len(negative_pat.findall(trace))
        unc_matches = len(uncertainty_pat.findall(trace))
        
        # Positional weighting (end of trace matters more)
        trace_end = trace[-200:]
        pos_end = len(positive_pat.findall(trace_end)) * 1.5
        neg_end = len(negative_pat.findall(trace_end)) * 1.5
        
        total_pos = pos_matches + pos_end
        total_neg = neg_matches + neg_end
        
        # Normalized scores
        max_count = max(total_pos + total_neg, 1.0)
        
        return {
            'positive_score': total_pos / max_count,
            'negative_score': total_neg / max_count,
            'uncertainty_score': unc_matches / (len(trace.split()) + 1.0),
            'net_score': (total_pos - total_neg) / max_count
        }
    
    def _compute_signals_for_option(self, 
                                     trace: str, 
                                     option: str,
                                     tool_outputs: Optional[List[Dict]] = None) -> SelectionSignals:
        """Compute all signals for a single option"""
        signals = SelectionSignals()
        
        # 1. Trace quality analysis
        trace_metrics = self.trace_analyzer.analyze_trace_quality(trace)
        signals.trace_quality = trace_metrics['overall']
        
        # 2. Keyword-based scoring
        keyword_metrics = self._compute_keyword_signals(trace)
        signals.keyword_score = (keyword_metrics['net_score'] + 1.0) / 2.0  # Normalize to [0, 1]
        signals.uncertainty = keyword_metrics['uncertainty_score']
        
        # 3. Tool confidence
        tool_signals = self.tool_extractor.extract_tool_signals(trace, tool_outputs)
        signals.tool_confidence = float(np.mean(list(tool_signals.values())))
        
        # 4. Meta-model score
        if self.use_meta_model and self.meta_model:
            try:
                signals.meta_score = float(self.meta_model(trace, option))
            except Exception as e:
                warnings.warn(f"Meta-model failed: {e}")
                signals.meta_score = 0.0
        
        return signals
    
    def _compute_semantic_consistency(self, traces: List[str], signals: List[SelectionSignals]) -> None:
        """Compute semantic similarity and consistency scores"""
        if not self.semantic_engine or len(traces) <= 1:
            return
        
        # Compute similarity matrix
        sim_matrix = self.semantic_engine.compute_similarity_matrix(traces)
        
        # Update signals with semantic similarity
        for i in range(len(traces)):
            # Average similarity to all other traces
            similarities = sim_matrix[i]
            similarities[i] = 0  # Exclude self
            avg_similarity = np.mean(similarities) if len(similarities) > 1 else 0.0
            signals[i].semantic_similarity = float(avg_similarity)
        
        # Compute overall self-consistency
        consistency = self.semantic_engine.compute_self_consistency(traces)
        
        # Update consistency scores
        for signal in signals:
            signal.consistency_score = consistency
    
    def select(self, 
               traces: List[str], 
               answer_options: List[str],
               tool_outputs: Optional[List[List[Dict]]] = None) -> int:
        """
        Select the best answer option with advanced multi-signal aggregation.
        
        Args:
            traces: List of reasoning traces (one per option)
            answer_options: List of answer options
            tool_outputs: Optional list of tool outputs per option
            
        Returns:
            1-based index of selected option (or SelectionResult if detailed=True)
        """
        if not traces or not answer_options:
            return 1
        
        if len(traces) != len(answer_options):
            warnings.warn(f"Mismatch: {len(traces)} traces vs {len(answer_options)} options")
            traces = traces[:len(answer_options)]
        
        # Initialize tool outputs if not provided
        if tool_outputs is None:
            tool_outputs = [[]] * len(traces)
        
        # Parallel signal computation
        if self.parallel and len(traces) > 2:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(self._compute_signals_for_option, trace, opt, tools)
                    for trace, opt, tools in zip(traces, answer_options, tool_outputs)
                ]
                signals = [f.result() for f in futures]
        else:
            signals = [
                self._compute_signals_for_option(trace, opt, tools)
                for trace, opt, tools in zip(traces, answer_options, tool_outputs)
            ]
        
        # Compute semantic consistency (requires all traces)
        if self.use_semantic:
            self._compute_semantic_consistency(traces, signals)
        
        # Aggregate scores
        raw_scores = [s.aggregate(self.aggregation_weights) for s in signals]
        
        # Ensemble voting with distance-aware weighting
        if self.use_ensemble and self.semantic_engine and len(traces) > 1:
            sim_matrix = self.semantic_engine.compute_similarity_matrix(traces)
            adjusted_scores = self.ensemble_scorer.distance_aware_voting(raw_scores, sim_matrix)
        else:
            adjusted_scores = raw_scores
        
        # Calibrate confidence
        if self.calibrate:
            score_variance = float(np.var(adjusted_scores))
            calibrated_scores = []
            uncertainties = []
            
            for score in adjusted_scores:
                cal_score, uncertainty = self.calibrator.calibrate_confidence(
                    score, score_variance, len(adjusted_scores)
                )
                calibrated_scores.append(cal_score)
                uncertainties.append(uncertainty)
            
            final_scores = calibrated_scores
        else:
            final_scores = adjusted_scores
            uncertainties = [s.uncertainty for s in signals]
        
        # Select best option
        best_idx = int(np.argmax(final_scores))
        best_score = final_scores[best_idx]
        best_uncertainty = uncertainties[best_idx]
        
        # Detect ties (scores within 5%)
        tie_threshold = 0.05
        top_scores = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
        tie_detected = len(top_scores) > 1 and (top_scores[0][1] - top_scores[1][1]) < tie_threshold
        
        # Handle ties with advanced tie-breaking
        if tie_detected:
            best_idx = self._break_tie(traces, answer_options, signals, final_scores)
        
        # Update statistics
        self.stats['total_selections'] += 1
        self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (self.stats['total_selections'] - 1) + 
                                        best_score) / self.stats['total_selections']
        self.stats['avg_uncertainty'] = (self.stats['avg_uncertainty'] * (self.stats['total_selections'] - 1) + 
                                         best_uncertainty) / self.stats['total_selections']
        if tie_detected:
            self.stats['tie_rate'] = (self.stats['tie_rate'] * (self.stats['total_selections'] - 1) + 1.0) / \
                                     self.stats['total_selections']
        
        return best_idx + 1  # Return 1-based index
    
    def _break_tie(self, traces: List[str], options: List[str], 
                   signals: List[SelectionSignals], scores: List[float]) -> int:
        """Advanced tie-breaking using multiple strategies"""
        # Find tied options (within 5% of max)
        max_score = max(scores)
        tied_indices = [i for i, s in enumerate(scores) if s >= max_score * 0.95]
        
        if len(tied_indices) == 1:
            return tied_indices[0]
        
        # Strategy 1: Prefer option with lowest uncertainty
        uncertainties = [signals[i].uncertainty for i in tied_indices]
        min_uncertainty_idx = tied_indices[np.argmin(uncertainties)]
        
        # Strategy 2: Prefer option with highest tool confidence
        tool_confidences = [signals[i].tool_confidence for i in tied_indices]
        max_tool_idx = tied_indices[np.argmax(tool_confidences)]
        
        # Strategy 3: Prefer option with best trace quality
        trace_qualities = [signals[i].trace_quality for i in tied_indices]
        best_quality_idx = tied_indices[np.argmax(trace_qualities)]
        
        # Vote among strategies
        votes = Counter([min_uncertainty_idx, max_tool_idx, best_quality_idx])
        return votes.most_common(1)[0][0]
    
    def select_detailed(self, 
                       traces: List[str], 
                       answer_options: List[str],
                       tool_outputs: Optional[List[List[Dict]]] = None) -> SelectionResult:
        """
        Select with detailed result including confidence, uncertainty, and reasoning.
        """
        # Reuse select logic but return detailed result
        if not traces or not answer_options:
            return SelectionResult(
                selected_index=1,
                confidence=0.0,
                uncertainty=1.0,
                all_scores=[],
                signals=[],
                reasoning="No valid inputs"
            )
        
        # [Simplified version - full implementation would mirror select() logic]
        selected_idx = self.select(traces, answer_options, tool_outputs)
        
        # Compute signals again (or cache from select call)
        if tool_outputs is None:
            tool_outputs = [[]] * len(traces)
        
        signals = [
            self._compute_signals_for_option(trace, opt, tools)
            for trace, opt, tools in zip(traces, answer_options, tool_outputs)
        ]
        
        scores = [s.aggregate(self.aggregation_weights) for s in signals]
        
        # selected_idx is already an int from select()
        idx = selected_idx if isinstance(selected_idx, int) else 1
        
        return SelectionResult(
            selected_index=idx,
            confidence=scores[idx - 1],
            uncertainty=signals[idx - 1].uncertainty,
            all_scores=scores,
            signals=signals,
            reasoning=f"Selected based on aggregated signals: score={scores[idx - 1]:.3f}"
        )
    
    def full_score_report(self, 
                         traces: List[str], 
                         answer_options: List[str],
                         return_details: bool = False,
                         tool_outputs: Optional[List[List[Dict]]] = None) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Generate comprehensive scoring report for all options.
        """
        if tool_outputs is None:
            tool_outputs = [[]] * len(traces)
        
        signals = [
            self._compute_signals_for_option(trace, opt, tools)
            for trace, opt, tools in zip(traces, answer_options, tool_outputs)
        ]
        
        if self.use_semantic and len(traces) > 1:
            self._compute_semantic_consistency(traces, signals)
        
        scores = [s.aggregate(self.aggregation_weights) for s in signals]
        
        if not return_details:
            return scores
        
        # Detailed report
        report = []
        for idx, (trace, opt, signal, score) in enumerate(zip(traces, answer_options, signals, scores)):
            keyword_metrics = self._compute_keyword_signals(trace)
            
            report.append({
                "option": opt,
                "score": score,
                "signals": {
                    "trace_quality": signal.trace_quality,
                    "keyword_score": signal.keyword_score,
                    "semantic_similarity": signal.semantic_similarity,
                    "tool_confidence": signal.tool_confidence,
                    "consistency_score": signal.consistency_score,
                    "meta_score": signal.meta_score
                },
                "uncertainty": signal.uncertainty,
                "keywords": {
                    "positive": keyword_metrics['positive_score'],
                    "negative": keyword_metrics['negative_score'],
                    "net": keyword_metrics['net_score']
                },
                "trace_preview": trace[:200] + "..." if len(trace) > 200 else trace
            })
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset performance tracking"""
        self.stats = {
            'total_selections': 0,
            'avg_confidence': 0.0,
            'avg_uncertainty': 0.0,
            'tie_rate': 0.0
        }

# ==== Batch Processing ====
def batch_select(selector: SelectionModule,
                 problems: List[Tuple[List[str], List[str]]],
                 max_workers: int = MAX_WORKERS) -> List[int]:
    """
    Batch process multiple selection problems in parallel.
    
    Args:
        selector: SelectionModule instance
        problems: List of (traces, options) tuples
        max_workers: Max parallel workers
        
    Returns:
        List of selected indices (1-based)
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(selector.select, traces, opts): i for i, (traces, opts) in enumerate(problems)}
        results = [0] * len(problems)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    
    return results

# ==== Usage Examples ====
if __name__ == "__main__":
    print("="*80)
    print("🚀 ULTRA-ADVANCED SELECTION MODULE 2025 - DEMONSTRATION")
    print("="*80)
    
    # Test data
    traces = [
        """Step 1: Analyze the problem carefully. The conditions are: X > 10, Y < 5.
        Step 2: Option A gives X=12, Y=3. Both conditions satisfied.
        Step 3: Verified with calculator tool: 12 > 10 ✓, 3 < 5 ✓
        Step 4: Cross-check with knowledge base confirms this is correct.
        Therefore, this is the final answer. Confidence: HIGH""",
        
        """Step 1: Check option B. X=8, Y=2.
        Step 2: First condition fails: 8 is not > 10.
        Step 3: This contradicts the requirements.
        Conclusion: Invalid option, ruled out.""",
        
        """Step 1: Option C has X=15, Y=6.
        Step 2: First condition OK: 15 > 10 ✓
        Step 3: Second condition fails: 6 is not < 5 ✗
        Step 4: Inconsistent with requirements, cannot be the answer.""",
        
        """Step 1: Maybe option D? X=11, Y=4.
        Step 2: Checking... 11 > 10, seems OK.
        Step 3: And 4 < 5, also OK.
        Step 4: This could be a valid answer, though less certain than A."""
    ]
    
    answer_options = ["A: X=12, Y=3", "B: X=8, Y=2", "C: X=15, Y=6", "D: X=11, Y=4"]
    
    # Simulate tool outputs
    tool_outputs = [
        [{"success": True, "confidence": 0.95, "output": "Conditions verified"}],
        [{"success": False, "confidence": 0.3, "output": "First condition failed"}],
        [{"success": False, "confidence": 0.4, "output": "Second condition failed"}],
        [{"success": True, "confidence": 0.75, "output": "Conditions OK"}]
    ]
    
    # Initialize selector with all features
    print("\n📊 Initializing ultra-advanced selector...")
    selector = SelectionModule(
        use_semantic_similarity=True,
        use_ensemble=True,
        parallel_execution=True,
        calibrate_confidence=True
    )
    
    # Basic selection
    print("\n🎯 Running selection...")
    start_time = time.time()
    selected_idx = selector.select(traces, answer_options, tool_outputs)
    elapsed = time.time() - start_time
    
    print(f"✅ Selected Option: {selected_idx} ({answer_options[selected_idx-1]})")
    print(f"⏱️  Processing Time: {elapsed*1000:.2f}ms")
    
    # Detailed report
    print("\n📋 Detailed Score Report:")
    print("-" * 80)
    report_data = selector.full_score_report(traces, answer_options, return_details=True, tool_outputs=tool_outputs)
    
    # Type guard: ensure we have detailed report
    if isinstance(report_data, list) and len(report_data) > 0 and isinstance(report_data[0], dict):
        detailed_report: List[Dict[str, Any]] = report_data  # type: ignore
        for item in detailed_report:
            print(f"\n{item['option']}:")
            print(f"  Overall Score: {item['score']:.3f}")
            print(f"  Uncertainty: {item['uncertainty']:.3f}")
            print(f"  Signals:")
            for sig_name, sig_val in item['signals'].items():
                print(f"    - {sig_name}: {sig_val:.3f}")
    
    # Statistics
    print("\n📈 Performance Statistics:")
    print("-" * 80)
    stats = selector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n" + "="*80)
    print("✨ Demo completed successfully!")
    print("="*80)
