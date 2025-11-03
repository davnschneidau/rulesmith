"""Hot path compilation for predicates and rules."""

import ast
import hashlib
from typing import Any, Callable, Dict, Optional


class PredicateCompiler:
    """Compiles predicates for hot path execution."""
    
    def __init__(self, enable_jit: bool = True):
        """
        Initialize predicate compiler.
        
        Args:
            enable_jit: Whether to enable JIT compilation (Numba/numexpr)
        """
        self.enable_jit = enable_jit
        self._compiled_cache: Dict[str, Callable] = {}
        self._numba_available = False
        self._numexpr_available = False
        
        # Check for Numba
        try:
            import numba
            self._numba_available = True
        except ImportError:
            pass
        
        # Check for numexpr
        try:
            import numexpr
            self._numexpr_available = True
        except ImportError:
            pass
    
    def compile_predicate(
        self,
        predicate: str,
        use_numba: bool = False,
        use_numexpr: bool = False,
    ) -> Callable:
        """
        Compile a predicate expression for fast evaluation.
        
        Args:
            predicate: Predicate expression (e.g., "age >= 18 and income > 30000")
            use_numba: Use Numba JIT (if available)
            use_numexpr: Use numexpr for numeric expressions (if available)
        
        Returns:
            Compiled function
        """
        # Check cache
        cache_key = hashlib.sha256(predicate.encode("utf-8")).hexdigest()
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        
        # Try numexpr for numeric expressions
        if use_numexpr and self._numexpr_available:
            try:
                compiled = self._compile_numexpr(predicate)
                self._compiled_cache[cache_key] = compiled
                return compiled
            except Exception:
                pass  # Fall through to Python
        
        # Try Numba for general predicates
        if use_numba and self._numba_available:
            try:
                compiled = self._compile_numba(predicate)
                self._compiled_cache[cache_key] = compiled
                return compiled
            except Exception:
                pass  # Fall through to Python
        
        # Fallback to compiled Python bytecode
        compiled = self._compile_python(predicate)
        self._compiled_cache[cache_key] = compiled
        return compiled
    
    def _compile_numexpr(self, predicate: str) -> Callable:
        """Compile using numexpr for numeric expressions."""
        import numexpr
        
        # numexpr is great for numeric expressions
        # e.g., "age >= 18 and income > 30000"
        
        def evaluate(variables: Dict[str, Any]) -> bool:
            try:
                result = numexpr.evaluate(predicate, local_dict=variables)
                # numexpr returns numpy array, convert to bool
                if hasattr(result, 'item'):
                    return bool(result.item())
                return bool(result)
            except Exception:
                # If numexpr fails, return False
                return False
        
        return evaluate
    
    def _compile_numba(self, predicate: str) -> Callable:
        """Compile using Numba JIT."""
        import numba
        
        # Parse predicate and create optimized function
        # This is a simplified implementation
        
        # Create a function that evaluates the predicate
        def evaluate_impl(variables: Dict[str, Any]) -> bool:
            try:
                # Use asteval for safe evaluation
                from asteval import Interpreter
                aeval = Interpreter()
                for key, value in variables.items():
                    aeval.symtable[key] = value
                result = aeval.eval(predicate)
                return bool(result)
            except Exception:
                return False
        
        # JIT compile if numba available
        if self.enable_jit:
            try:
                return numba.jit(nopython=False)(evaluate_impl)
            except Exception:
                return evaluate_impl
        
        return evaluate_impl
    
    def _compile_python(self, predicate: str) -> Callable:
        """Compile to Python bytecode (fastest safe fallback)."""
        # Parse and compile to bytecode
        try:
            # Use asteval for safe evaluation
            from asteval import Interpreter
            
            # Pre-compile the expression
            aeval = Interpreter()
            aeval.parse(predicate)
            
            def evaluate(variables: Dict[str, Any]) -> bool:
                try:
                    # Create new interpreter with variables
                    local_aeval = Interpreter()
                    for key, value in variables.items():
                        local_aeval.symtable[key] = value
                    result = local_aeval.eval(predicate)
                    return bool(result)
                except Exception:
                    return False
            
            return evaluate
        except Exception:
            # Ultimate fallback: eval with restricted globals
            def evaluate(variables: Dict[str, Any]) -> bool:
                try:
                    # Safe evaluation with only variables
                    result = eval(predicate, {"__builtins__": {}}, variables)
                    return bool(result)
                except Exception:
                    return False
            
            return evaluate
    
    def compile_rule_function(
        self,
        rule_func: Callable,
        use_numba: bool = False,
    ) -> Callable:
        """
        Compile a rule function for hot path execution.
        
        Args:
            rule_func: Rule function to compile
            use_numba: Use Numba JIT (if available)
        
        Returns:
            Compiled function
        """
        # Check cache
        func_hash = hashlib.sha256(str(rule_func.__code__.co_code).encode()).hexdigest()
        cache_key = f"rule_{func_hash}"
        
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        
        # Try Numba compilation
        if use_numba and self._numba_available:
            try:
                import numba
                compiled = numba.jit(nopython=False)(rule_func)
                self._compiled_cache[cache_key] = compiled
                return compiled
            except Exception:
                pass  # Fall through
        
        # Return original function (already compiled to bytecode)
        self._compiled_cache[cache_key] = rule_func
        return rule_func


# Global compiler
predicate_compiler = PredicateCompiler()

