"""
Advanced Regex Engine Tool with LangChain Integration

This module provides a comprehensive regex engine with advanced features including:
- Fuzzy matching (approximate pattern matching)
- Named capture groups
- Lookahead/lookbehind assertions
- Multiple pattern testing
- Batch processing
- Performance optimization
- LangChain tool integration
- Comprehensive error handling and validation
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

# Try to import the advanced regex module, fall back to re if unavailable
try:
    import regex
    HAS_REGEX_MODULE = True
except ImportError:
    import re as regex
    HAS_REGEX_MODULE = False
    logging.warning("Advanced 'regex' module not found. Install with: pip install regex")

# LangChain imports
try:
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain_core.runnables import RunnableLambda
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
    HAS_LANGCHAIN = True
    BaseTool = LangChainBaseTool
    BaseModel = PydanticBaseModel
    Field = PydanticField
except ImportError:
    HAS_LANGCHAIN = False
    BaseTool = object  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda **kwargs: None  # type: ignore
    logging.warning("LangChain not found. Install with: pip install langchain langchain-core")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegexOperation(Enum):
    """Enum for different regex operations"""
    SEARCH = "search"
    FINDALL = "findall"
    MATCH = "match"
    SPLIT = "split"
    SUB = "sub"
    FINDITER = "finditer"
    FULLMATCH = "fullmatch"


class RegexFlags(Enum):
    """Common regex flags"""
    IGNORECASE = re.IGNORECASE
    MULTILINE = re.MULTILINE
    DOTALL = re.DOTALL
    VERBOSE = re.VERBOSE
    ASCII = re.ASCII
    UNICODE = re.UNICODE


@dataclass
class RegexMatch:
    """Data class to represent a regex match with metadata"""
    text: str
    start: int
    end: int
    groups: List[str] = field(default_factory=list)
    groupdict: Dict[str, str] = field(default_factory=dict)
    pattern: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "groups": self.groups,
            "groupdict": self.groupdict,
            "pattern": self.pattern
        }


@dataclass
class RegexResult:
    """Data class to represent regex operation results"""
    success: bool
    operation: str
    pattern: str
    matches: List[Union[str, RegexMatch]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        matches_data = []
        for match in self.matches:
            if isinstance(match, RegexMatch):
                matches_data.append(match.to_dict())
            else:
                matches_data.append(match)
        
        return {
            "success": self.success,
            "operation": self.operation,
            "pattern": self.pattern,
            "matches": matches_data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AdvancedRegexEngine:
    """
    Advanced Regex Engine with comprehensive pattern matching capabilities
    
    Features:
    - Pattern validation and compilation
    - Fuzzy matching (with regex module)
    - Named capture groups
    - Lookahead/lookbehind assertions
    - Multiple pattern testing
    - Batch processing
    - Performance optimization through pattern caching
    """
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize the regex engine
        
        Args:
            cache_size: Maximum number of compiled patterns to cache
        """
        self._pattern_cache: Dict[Tuple[str, int], Any] = {}
        self._cache_size = cache_size
        self.stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
    
    def validate_pattern(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a regex pattern
        
        Args:
            pattern: Regex pattern to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            regex.compile(pattern)
            return True, None
        except (regex.error if HAS_REGEX_MODULE else re.error) as e:
            return False, f"Invalid regex pattern: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _get_compiled_pattern(self, pattern: str, flags: int = 0):
        """
        Get compiled pattern from cache or compile and cache it
        
        Args:
            pattern: Regex pattern
            flags: Regex flags
            
        Returns:
            Compiled regex pattern
        """
        cache_key = (pattern, flags)
        
        if cache_key in self._pattern_cache:
            self.stats["cache_hits"] += 1
            return self._pattern_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        try:
            compiled = regex.compile(pattern, flags)
            
            # Manage cache size
            if len(self._pattern_cache) >= self._cache_size:
                # Remove oldest entry
                self._pattern_cache.pop(next(iter(self._pattern_cache)))
            
            self._pattern_cache[cache_key] = compiled
            return compiled
            
        except Exception as e:
            raise ValueError(f"Failed to compile pattern: {str(e)}")
    
    def search(
        self,
        pattern: str,
        text: str,
        flags: int = 0,
        fuzzy_errors: Optional[int] = None
    ) -> RegexResult:
        """
        Search for pattern in text
        
        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            fuzzy_errors: Number of errors allowed for fuzzy matching (requires regex module)
            
        Returns:
            RegexResult object
        """
        self.stats["total_operations"] += 1
        
        try:
            # Validate pattern
            is_valid, error = self.validate_pattern(pattern)
            if not is_valid:
                self.stats["errors"] += 1
                return RegexResult(
                    success=False,
                    operation="search",
                    pattern=pattern,
                    error=error
                )
            
            # Apply fuzzy matching if requested
            search_pattern = pattern
            if fuzzy_errors is not None and HAS_REGEX_MODULE:
                search_pattern = f'(?:{pattern}){{e<={fuzzy_errors}}}'
            
            compiled = self._get_compiled_pattern(search_pattern, flags)
            match = compiled.search(text)
            
            if match:
                regex_match = RegexMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    groups=list(match.groups()),
                    groupdict=match.groupdict(),
                    pattern=pattern
                )
                
                return RegexResult(
                    success=True,
                    operation="search",
                    pattern=pattern,
                    matches=[regex_match],
                    metadata={"fuzzy_errors": fuzzy_errors}
                )
            else:
                return RegexResult(
                    success=True,
                    operation="search",
                    pattern=pattern,
                    matches=[],
                    metadata={"fuzzy_errors": fuzzy_errors}
                )
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in search operation: {str(e)}")
            return RegexResult(
                success=False,
                operation="search",
                pattern=pattern,
                error=str(e)
            )
    
    def findall(
        self,
        pattern: str,
        text: str,
        flags: int = 0,
        fuzzy_errors: Optional[int] = None,
        return_matches: bool = False
    ) -> RegexResult:
        """
        Find all occurrences of pattern in text
        
        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            fuzzy_errors: Number of errors allowed for fuzzy matching
            return_matches: Return RegexMatch objects instead of strings
            
        Returns:
            RegexResult object
        """
        self.stats["total_operations"] += 1
        
        try:
            is_valid, error = self.validate_pattern(pattern)
            if not is_valid:
                self.stats["errors"] += 1
                return RegexResult(
                    success=False,
                    operation="findall",
                    pattern=pattern,
                    error=error
                )
            
            search_pattern = pattern
            if fuzzy_errors is not None and HAS_REGEX_MODULE:
                search_pattern = f'(?:{pattern}){{e<={fuzzy_errors}}}'
            
            compiled = self._get_compiled_pattern(search_pattern, flags)
            
            if return_matches:
                matches = []
                for match in compiled.finditer(text):
                    regex_match = RegexMatch(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        groups=list(match.groups()),
                        groupdict=match.groupdict(),
                        pattern=pattern
                    )
                    matches.append(regex_match)
            else:
                matches = compiled.findall(text)
            
            return RegexResult(
                success=True,
                operation="findall",
                pattern=pattern,
                matches=matches,
                metadata={
                    "count": len(matches),
                    "fuzzy_errors": fuzzy_errors
                }
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in findall operation: {str(e)}")
            return RegexResult(
                success=False,
                operation="findall",
                pattern=pattern,
                error=str(e)
            )
    
    def substitute(
        self,
        pattern: str,
        replacement: str,
        text: str,
        count: int = 0,
        flags: int = 0
    ) -> RegexResult:
        """
        Replace pattern occurrences with replacement string
        
        Args:
            pattern: Regex pattern
            replacement: Replacement string
            text: Text to process
            count: Maximum number of replacements (0 = all)
            flags: Regex flags
            
        Returns:
            RegexResult object with replaced text in metadata
        """
        self.stats["total_operations"] += 1
        
        try:
            is_valid, error = self.validate_pattern(pattern)
            if not is_valid:
                self.stats["errors"] += 1
                return RegexResult(
                    success=False,
                    operation="substitute",
                    pattern=pattern,
                    error=error
                )
            
            compiled = self._get_compiled_pattern(pattern, flags)
            result_text = compiled.sub(replacement, text, count=count)
            
            # Count actual replacements
            original_matches = len(compiled.findall(text))
            remaining_matches = len(compiled.findall(result_text))
            replacements_made = original_matches - remaining_matches
            
            return RegexResult(
                success=True,
                operation="substitute",
                pattern=pattern,
                matches=[],
                metadata={
                    "original_text": text,
                    "result_text": result_text,
                    "replacement": replacement,
                    "replacements_made": replacements_made
                }
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in substitute operation: {str(e)}")
            return RegexResult(
                success=False,
                operation="substitute",
                pattern=pattern,
                error=str(e)
            )
    
    def split(
        self,
        pattern: str,
        text: str,
        maxsplit: int = 0,
        flags: int = 0
    ) -> RegexResult:
        """
        Split text by pattern
        
        Args:
            pattern: Regex pattern
            text: Text to split
            maxsplit: Maximum number of splits
            flags: Regex flags
            
        Returns:
            RegexResult object
        """
        self.stats["total_operations"] += 1
        
        try:
            is_valid, error = self.validate_pattern(pattern)
            if not is_valid:
                self.stats["errors"] += 1
                return RegexResult(
                    success=False,
                    operation="split",
                    pattern=pattern,
                    error=error
                )
            
            compiled = self._get_compiled_pattern(pattern, flags)
            parts = compiled.split(text, maxsplit=maxsplit)
            
            return RegexResult(
                success=True,
                operation="split",
                pattern=pattern,
                matches=parts,
                metadata={"parts_count": len(parts)}
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in split operation: {str(e)}")
            return RegexResult(
                success=False,
                operation="split",
                pattern=pattern,
                error=str(e)
            )
    
    def test_multiple_patterns(
        self,
        patterns: List[str],
        text: str,
        flags: int = 0
    ) -> Dict[str, RegexResult]:
        """
        Test multiple patterns against text
        
        Args:
            patterns: List of regex patterns
            text: Text to test
            flags: Regex flags
            
        Returns:
            Dictionary mapping patterns to results
        """
        results = {}
        
        for pattern in patterns:
            result = self.findall(pattern, text, flags)
            results[pattern] = result
        
        return results
    
    def batch_search(
        self,
        pattern: str,
        texts: List[str],
        flags: int = 0,
        fuzzy_errors: Optional[int] = None
    ) -> List[RegexResult]:
        """
        Search pattern across multiple texts
        
        Args:
            pattern: Regex pattern
            texts: List of texts to search
            flags: Regex flags
            fuzzy_errors: Number of errors for fuzzy matching
            
        Returns:
            List of RegexResult objects
        """
        results = []
        
        for text in texts:
            result = self.findall(pattern, text, flags, fuzzy_errors, return_matches=True)
            results.append(result)
        
        return results
    
    def extract_with_groups(
        self,
        pattern: str,
        text: str,
        flags: int = 0
    ) -> RegexResult:
        """
        Extract text with named capture groups
        
        Args:
            pattern: Regex pattern with named groups
            text: Text to extract from
            flags: Regex flags
            
        Returns:
            RegexResult with captured groups
        """
        self.stats["total_operations"] += 1
        
        try:
            is_valid, error = self.validate_pattern(pattern)
            if not is_valid:
                self.stats["errors"] += 1
                return RegexResult(
                    success=False,
                    operation="extract_groups",
                    pattern=pattern,
                    error=error
                )
            
            compiled = self._get_compiled_pattern(pattern, flags)
            matches = []
            
            for match in compiled.finditer(text):
                regex_match = RegexMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    groups=list(match.groups()),
                    groupdict=match.groupdict(),
                    pattern=pattern
                )
                matches.append(regex_match)
            
            return RegexResult(
                success=True,
                operation="extract_groups",
                pattern=pattern,
                matches=matches,
                metadata={"matches_count": len(matches)}
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in extract_groups operation: {str(e)}")
            return RegexResult(
                success=False,
                operation="extract_groups",
                pattern=pattern,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        cache_hit_rate = 0.0
        if self.stats["cache_hits"] + self.stats["cache_misses"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / (
                self.stats["cache_hits"] + self.stats["cache_misses"]
            )
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "cache_size": len(self._pattern_cache),
            "max_cache_size": self._cache_size
        }
    
    def clear_cache(self):
        """Clear pattern cache"""
        self._pattern_cache.clear()
        logger.info("Pattern cache cleared")


# LangChain Integration
if HAS_LANGCHAIN:
    
    class RegexToolInput(BaseModel):
        """Input schema for regex tool"""
        pattern: str = Field(description="Regular expression pattern")
        text: str = Field(description="Text to search/process")
        operation: str = Field(
            default="findall",
            description="Operation: search, findall, substitute, split, extract_groups"
        )
        replacement: Optional[str] = Field(
            default=None,
            description="Replacement string for substitute operation"
        )
        flags: Optional[int] = Field(
            default=0,
            description="Regex flags (use 0 for default)"
        )
        fuzzy_errors: Optional[int] = Field(
            default=None,
            description="Number of fuzzy matching errors allowed"
        )
    
    
    class RegexEngineTool(BaseTool):
        """LangChain tool wrapper for regex engine"""
        
        name: str = "regex_engine"
        description: str = """
        Advanced regex pattern matching tool with support for:
        - Pattern searching and matching
        - Text extraction with named capture groups
        - Text replacement
        - Text splitting
        - Fuzzy matching (approximate matching)
        - Batch processing
        
        Operations:
        - search: Find first match
        - findall: Find all matches
        - substitute: Replace matches
        - split: Split text by pattern
        - extract_groups: Extract with named groups
        """
        args_schema: type[BaseModel] = RegexToolInput
        
        engine: AdvancedRegexEngine = AdvancedRegexEngine()
        
        def _run(
            self,
            pattern: str,
            text: str,
            operation: str = "findall",
            replacement: Optional[str] = None,
            flags: int = 0,
            fuzzy_errors: Optional[int] = None
        ) -> str:
            """Execute regex operation"""
            
            if operation == "search":
                result = self.engine.search(pattern, text, flags, fuzzy_errors)
            elif operation == "findall":
                result = self.engine.findall(pattern, text, flags, fuzzy_errors, return_matches=True)
            elif operation == "substitute":
                if replacement is None:
                    return json.dumps({
                        "success": False,
                        "error": "Replacement string required for substitute operation"
                    })
                result = self.engine.substitute(pattern, replacement, text, flags=flags)
            elif operation == "split":
                result = self.engine.split(pattern, text, flags=flags)
            elif operation == "extract_groups":
                result = self.engine.extract_with_groups(pattern, text, flags)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                })
            
            return result.to_json()
        
        async def _arun(self, *args, **kwargs):
            """Async version (falls back to sync)"""
            return self._run(*args, **kwargs)


def create_regex_tool() -> Union[BaseTool, None]:
    """
    Factory function to create regex tool
    
    Returns:
        RegexEngineTool if LangChain is available, None otherwise
    """
    if HAS_LANGCHAIN:
        return RegexEngineTool()
    else:
        logger.warning("LangChain not available. Cannot create tool.")
        return None


# Common regex patterns library
class RegexPatterns:
    """Library of common regex patterns"""
    
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    URL = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    PHONE_US = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    IP_ADDRESS = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    DATE_ISO = r'\b\d{4}-\d{2}-\d{2}\b'
    TIME_24H = r'\b([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?\b'
    HASHTAG = r'#\w+'
    MENTION = r'@\w+'
    HEX_COLOR = r'#[0-9A-Fa-f]{6}\b'
    CREDIT_CARD = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
    SSN = r'\b\d{3}-\d{2}-\d{4}\b'
    ZIP_CODE = r'\b\d{5}(?:-\d{4})?\b'
    MAC_ADDRESS = r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
    UUID = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
    
    @classmethod
    def get_all_patterns(cls) -> Dict[str, str]:
        """Get all predefined patterns"""
        return {
            name: value for name, value in vars(cls).items()
            if not name.startswith('_') and isinstance(value, str)
        }


# Example usage and testing
def demo_regex_engine():
    """Demonstrate regex engine capabilities"""
    
    print("=" * 80)
    print("Advanced Regex Engine Demo")
    print("=" * 80)
    
    engine = AdvancedRegexEngine()
    
    # Example 1: Basic pattern matching
    print("\n1. Basic Pattern Matching")
    print("-" * 40)
    result = engine.findall(r'\b\w+@\w+\.\w+\b', "Contact: user@example.com or admin@test.org")
    print(f"Emails found: {result.matches}")
    
    # Example 2: Named capture groups
    print("\n2. Named Capture Groups")
    print("-" * 40)
    pattern = r'(?P<name>\w+)\s+(?P<age>\d+)'
    text = "Alice 30, Bob 25, Charlie 35"
    result = engine.extract_with_groups(pattern, text)
    for match in result.matches:
        print(f"Match: {match.groupdict}")
    
    # Example 3: Fuzzy matching (if regex module available)
    if HAS_REGEX_MODULE:
        print("\n3. Fuzzy Matching (Approximate)")
        print("-" * 40)
        result = engine.findall(r'hello', "helo, hello, hallo, hllo", fuzzy_errors=1)
        print(f"Fuzzy matches: {result.matches}")
    
    # Example 4: Text replacement
    print("\n4. Text Replacement")
    print("-" * 40)
    result = engine.substitute(
        r'\b\d{3}-\d{2}-\d{4}\b',
        '[REDACTED]',
        "SSNs: 123-45-6789 and 987-65-4321"
    )
    print(f"Original: {result.metadata['original_text']}")
    print(f"Redacted: {result.metadata['result_text']}")
    
    # Example 5: Multiple patterns
    print("\n5. Multiple Pattern Testing")
    print("-" * 40)
    patterns = [RegexPatterns.EMAIL, RegexPatterns.URL, RegexPatterns.PHONE_US]
    text = "Contact: user@example.com, call 555-123-4567, visit https://example.com"
    results = engine.test_multiple_patterns(patterns, text)
    for pattern, result in results.items():
        print(f"Pattern: {pattern[:30]}... -> Found {len(result.matches)} matches")
    
    # Statistics
    print("\n6. Engine Statistics")
    print("-" * 40)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_regex_engine()
    
    # Test LangChain integration if available
    if HAS_LANGCHAIN:
        print("\nTesting LangChain Integration...")
        tool = create_regex_tool()
        if tool:
            result = tool._run(
                pattern=r'\b\w+@\w+\.\w+\b',
                text="Emails: alice@example.com, bob@test.org",
                operation="findall"
            )
            print(f"Tool result:\n{result}")
