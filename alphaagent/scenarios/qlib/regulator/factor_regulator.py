import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from alphaagent.core.evaluation import Evaluator
from alphaagent.log import logger
from alphaagent.core.scenario import Scenario
from alphaagent.components.coder.factor_coder.factor_ast import match_alphazoo, count_free_args, count_unique_vars
from alphaagent.components.coder.factor_coder.expr_parser import parse_expression

class FactorRegulator(Evaluator):
    """
    FactorRegulator class to evaluate expressions for duplication and manage the factor zoo database.
    This class provides functionality to detect duplicated subtrees in factor expressions
    and ensure new factors maintain appropriate originality.
    """
    
    def __init__(self, factor_zoo_path: str = "factor_zoo/alpha101.csv", duplication_threshold: int = 5):
        """
        Initialize the FactorRegulator with a reference to the factor zoo.
        
        Args:
            factor_zoo_path (str): Path to the CSV file containing the factor zoo database.
            duplication_threshold (int): Threshold for duplication detection.
        """
        super().__init__(None)
        self.factor_zoo_path = factor_zoo_path
        self.alphazoo = pd.read_csv(factor_zoo_path, index_col=None)
        self.duplication_threshold = duplication_threshold
        self.new_factors = []
        
    
        
    def is_parsable(self, expression: str) -> bool:
        """
        Checks if an expression can be successfully parsed.
        
        Args:
            expression (str): The factor expression to check.
            
        Returns:
            bool: True if the expression can be parsed, False otherwise.
        """
        try:
            parse_expression(expression)
            return True
        except Exception as e:
            logger.error(f"Failed to parse expression: {expression}. Error: {str(e)}")
            return False
        
    def evaluate(self, expression: str) -> Tuple[int, str, Optional[str]]:
        """
        Evaluates an expression for duplication with existing factors in the factor zoo.
        
        Args:
            expression (str): The factor expression to evaluate.
            
        Returns:
            Tuple containing:
                - duplicated_subtree_size (int): Size of the duplicated subtree
                - duplicated_subtree (str): The duplicated subtree expression
                - matched_alpha (str or None): Name of the matched alpha if available
        """
        try:
            # Check for duplication
            duplicated_subtree_size, duplicated_subtree, matched_alpha = match_alphazoo(
                expression, self.alphazoo
            )
            
            num_free_args = count_free_args(expression)
            num_unique_vars = count_unique_vars(expression)
            
            logger.info(f"""
                        Evaluated expr: {expression}
                        Duplicated Size: {duplicated_subtree_size}
                        Duplicated Subtree: {duplicated_subtree}
                        # Free Args: {num_free_args}
                        # Unique Vars: {num_unique_vars}
                        """)
            
            eval_dict = {
                "expr": expression,
                "duplicated_subtree_size": duplicated_subtree_size, 
                "duplicated_subtree": duplicated_subtree,
                "matched_alpha": matched_alpha,
                "num_free_args": num_free_args,
                "num_unique_vars": num_unique_vars
                }
            
            return True, eval_dict
            
        except Exception as e:
            logger.error(f"Failed to evaluate expression: {expression}. Error: {str(e)}")
            return False, None
    
    
    def is_expression_acceptable(self, eval_dict) -> bool:
        """
        Determines if an expression is acceptable based on the duplication threshold.
        """
        import pdb; pdb.set_trace()
        return eval_dict['duplicated_subtree_size'] <= self.duplication_threshold
    
            
    def add_factor(self, factor_name: str, factor_expression: str) -> bool:
        """
        Adds a new factor to the in-memory factor zoo if it passes the duplication check.
        
        Args:
            factor_name (str): Name of the new factor.
            factor_expression (str): Expression of the new factor.
            
        Returns:
            bool: True if the factor was added, False otherwise.
        """
        new_factor = pd.DataFrame({
                'factor_name': factor_name,
                'factor_expression': factor_expression
                })
            
        self.alphazoo = pd.concat([self.alphazoo, new_factor])
        self.new_factors.append((factor_name, factor_expression))
        logger.info(f"Added new factor: {factor_name} with expression: {factor_expression}")
            
    def save_factor_zoo(self, output_path: Optional[str] = None) -> None:
        """
        Saves the updated factor zoo to a CSV file.
        
        Args:
            output_path (str, optional): Path to save the updated factor zoo.
                                         If None, updates the original file.
        """
        save_path = output_path if output_path else self.factor_zoo_path
        self.alphazoo.to_csv(save_path, index=False)
        logger.info(f"Saved updated factor zoo to {save_path}")
        
    def get_new_factors(self) -> List[Tuple[str, str]]:
        """
        Returns the list of new factors added during this session.
        
        Returns:
            List[Tuple[str, str]]: List of (factor_name, factor_expression) tuples.
        """
        return self.new_factors