"""
Problem Classifier for MCP-Enhanced Reasoning System
This module classifies reasoning problems into different types to enable
specialized handling and domain-specific prompts.
"""

import re
from typing import Dict, List, Tuple
from enum import Enum

class ProblemType(Enum):
    """Problem type categories for specialized handling"""
    SPATIAL_GEOMETRY = "spatial_geometry"
    OPTIMIZATION = "optimization"
    PATTERN_RECOGNITION = "pattern_recognition"
    LOGIC_PUZZLE = "logic_puzzle"
    MATHEMATICAL = "mathematical"
    SCHEDULING = "scheduling"
    PROBABILITY = "probability"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

class ProblemClassifier:
    """
    Classifies reasoning problems into types for specialized handling.
    Uses keyword matching and pattern recognition.
    """
    
    def __init__(self):
        """Initialize classifier with keyword patterns"""
        self.patterns = {
            ProblemType.SPATIAL_GEOMETRY: [
                r'\bcube\b', r'\bsphere\b', r'\bcylinder\b', r'\bpyramid\b',
                r'\bpaint(ed)?\b.*\b(sides?|faces?|edges?|corners?)\b',
                r'\b(width|height|length|depth|dimension)\b',
                r'\b(area|volume|perimeter|surface)\b',
                r'\b(adjacent|opposite|parallel|perpendicular)\b',
                r'\b\d+x\d+x\d+\b',  # Pattern like 10x10x10
                r'\bgeometric\b', r'\bshape\b',
            ],
            ProblemType.OPTIMIZATION: [
                r'\b(maximize|minimize|optimal|optimize|best way|most efficient)\b',
                r'\bminimum (time|cost|distance|effort)\b',
                r'\bmaximum (items|output|profit|value)\b',
                r'\b(shortest path|quickest route|fastest method)\b',
                r'\bhow (many|much).*in.*time\b',
            ],
            ProblemType.SCHEDULING: [
                r'\b(schedule|order|sequence|arrange)\b',
                r'\bmachine[s]?\b.*\b(process|produce|manufacture)\b',
                r'\b(queue|waiting|processing time)\b',
                r'\b(first|second|third|last).*then\b',
                r'\bin what order\b',
                r'\b(A->B->C|A→B→C)\b',  # Sequence notation
            ],
            ProblemType.PATTERN_RECOGNITION: [
                r'\bpattern\b', r'\bsequence\b', r'\bseries\b',
                r'\bnext (number|term|element)\b',
                r'\b\d+,\s*\d+,\s*\d+,\s*[?,]\b',  # Number sequences
                r'\bcomplete the (pattern|sequence)\b',
                r'\bwhat comes next\b',
            ],
            ProblemType.LOGIC_PUZZLE: [
                r'\bif you (are|were)\b.*\bthen\b',
                r'\brace\b.*\bovertake\b',
                r'\b(riddle|puzzle)\b',
                r'\bwhat position\b',
                r'\ball.*except\b',
                r'\b(true|false).*statement\b',
                r'\bwho is\b.*\bif\b',
            ],
            ProblemType.MATHEMATICAL: [
                r'\bcalculate\b', r'\bcompute\b', r'\bsolve for\b',
                r'\b(sum|product|quotient|difference)\b',
                r'\b(equation|expression|formula)\b',
                r'\b\d+\s*[\+\-\*\/\^]\s*\d+\b',  # Math operations
                r'\bwhat is \d+\b',
            ],
            ProblemType.PROBABILITY: [
                r'\bprobability\b', r'\bchance\b', r'\blikely\b',
                r'\b(odds|likelihood)\b',
                r'\bhow many ways\b',
                r'\brandom(ly)?\b',
                r'\b(coin|dice|card)\b',
            ],
            ProblemType.COMPARISON: [
                r'\b(compare|contrast|difference between)\b',
                r'\b(more|less|greater|smaller|larger) than\b',
                r'\bwhich (is|has|does)\b.*\b(most|least)\b',
                r'\b(better|worse|faster|slower)\b',
            ],
        }
        
        # Confidence weights for each pattern type
        self.weights = {
            ProblemType.SPATIAL_GEOMETRY: 1.5,  # Higher weight for spatial
            ProblemType.OPTIMIZATION: 1.3,
            ProblemType.SCHEDULING: 1.3,
            ProblemType.PATTERN_RECOGNITION: 1.0,
            ProblemType.LOGIC_PUZZLE: 1.0,
            ProblemType.MATHEMATICAL: 0.8,  # Lower weight (often combined with others)
            ProblemType.PROBABILITY: 1.0,
            ProblemType.COMPARISON: 0.7,
        }
    
    def classify(self, problem_statement: str) -> Tuple[ProblemType, float, List[ProblemType]]:
        """
        Classify a problem statement into a type.
        
        Args:
            problem_statement: The problem text to classify
            
        Returns:
            Tuple of (primary_type, confidence, all_matching_types)
        """
        problem_lower = problem_statement.lower()
        
        # Count matches for each type
        type_scores = {}
        for problem_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, problem_lower, re.IGNORECASE))
                score += matches
            
            if score > 0:
                # Apply weight
                weighted_score = score * self.weights.get(problem_type, 1.0)
                type_scores[problem_type] = weighted_score
        
        if not type_scores:
            return ProblemType.UNKNOWN, 0.0, []
        
        # Sort by score
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary type is highest scoring
        primary_type = sorted_types[0][0]
        primary_score = sorted_types[0][1]
        
        # Confidence based on score and gap to second place
        if len(sorted_types) == 1:
            confidence = min(primary_score / 3.0, 1.0)  # Max at 3+ matches
        else:
            second_score = sorted_types[1][1]
            score_gap = primary_score - second_score
            confidence = min((primary_score / 3.0) * (1 + score_gap / 5.0), 1.0)
        
        # All types with significant scores (>0.5 of primary)
        threshold = primary_score * 0.5
        all_types = [t for t, s in sorted_types if s >= threshold]
        
        return primary_type, confidence, all_types
    
    def get_complexity_estimate(self, problem_type: ProblemType) -> str:
        """
        Estimate complexity level for a problem type.
        
        Returns:
            'high', 'medium', or 'low'
        """
        high_complexity = {
            ProblemType.SPATIAL_GEOMETRY,
            ProblemType.OPTIMIZATION,
            ProblemType.SCHEDULING,
        }
        
        medium_complexity = {
            ProblemType.PATTERN_RECOGNITION,
            ProblemType.PROBABILITY,
            ProblemType.MATHEMATICAL,
        }
        
        if problem_type in high_complexity:
            return 'high'
        elif problem_type in medium_complexity:
            return 'medium'
        else:
            return 'low'
    
    def get_recommended_approach(self, problem_type: ProblemType) -> Dict:
        """
        Get recommended approach for solving this problem type.
        
        Returns:
            Dictionary with strategy recommendations
        """
        approaches = {
            ProblemType.SPATIAL_GEOMETRY: {
                'strategy': 'formula_based',
                'tools': ['calculator', 'sequential_thinking'],
                'decomposition_focus': 'identify geometric formulas, break into dimensions',
                'verification': 'check formula application and calculation steps',
            },
            ProblemType.OPTIMIZATION: {
                'strategy': 'constraint_analysis',
                'tools': ['calculator', 'sequential_thinking', 'memory'],
                'decomposition_focus': 'identify constraints, evaluate each option',
                'verification': 'validate against all constraints',
            },
            ProblemType.SCHEDULING: {
                'strategy': 'sequential_evaluation',
                'tools': ['calculator', 'sequential_thinking', 'memory'],
                'decomposition_focus': 'model time dependencies, evaluate sequences',
                'verification': 'check total time for each sequence',
            },
            ProblemType.PATTERN_RECOGNITION: {
                'strategy': 'pattern_matching',
                'tools': ['sequential_thinking', 'search'],
                'decomposition_focus': 'identify pattern type, find rule',
                'verification': 'apply rule to all elements',
            },
            ProblemType.LOGIC_PUZZLE: {
                'strategy': 'step_by_step_logic',
                'tools': ['sequential_thinking', 'search'],
                'decomposition_focus': 'trace logical steps, apply rules',
                'verification': 'verify consistency with problem statement',
            },
            ProblemType.MATHEMATICAL: {
                'strategy': 'calculation',
                'tools': ['calculator', 'sequential_thinking'],
                'decomposition_focus': 'identify operations, compute step by step',
                'verification': 'double-check calculations',
            },
            ProblemType.PROBABILITY: {
                'strategy': 'counting_principles',
                'tools': ['calculator', 'sequential_thinking'],
                'decomposition_focus': 'identify sample space, count outcomes',
                'verification': 'verify probability ranges 0-1',
            },
            ProblemType.COMPARISON: {
                'strategy': 'systematic_comparison',
                'tools': ['search', 'sequential_thinking'],
                'decomposition_focus': 'list criteria, compare each option',
                'verification': 'ensure all criteria considered',
            },
        }
        
        return approaches.get(problem_type, {
            'strategy': 'general',
            'tools': ['sequential_thinking', 'search'],
            'decomposition_focus': 'break into steps, analyze systematically',
            'verification': 'review each step',
        })


# Example usage and testing
if __name__ == "__main__":
    classifier = ProblemClassifier()
    
    test_problems = [
        "A 10x10x10 cube is painted red on all sides. How many smaller cubes have exactly two sides painted?",
        "Three machines A(3min), B(4min), C(2min). What order maximizes items in 60 min?",
        "If you are in a race and overtake the second person, what position are you in?",
        "What is the next number in the sequence: 2, 4, 8, 16, ?",
    ]
    
    print("Problem Classification Test")
    print("=" * 80)
    
    for problem in test_problems:
        ptype, conf, all_types = classifier.classify(problem)
        complexity = classifier.get_complexity_estimate(ptype)
        approach = classifier.get_recommended_approach(ptype)
        
        print(f"\nProblem: {problem[:60]}...")
        print(f"Type: {ptype.value} (confidence: {conf:.2f}, complexity: {complexity})")
        print(f"All matching types: {[t.value for t in all_types]}")
        print(f"Strategy: {approach['strategy']}")
        print(f"Recommended tools: {', '.join(approach['tools'])}")
