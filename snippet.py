"""
Databricks Dynamic Pivot Module - Production Ready
Uses your existing fire_query module for execution
"""

import pandas as pd
import time
from typing import Dict, List, Any, Optional
import logging

# Import your existing Databricks query module
try:
    from dbx import fire_query
except ImportError:
    def fire_query(sql_query: str) -> pd.DataFrame:
        raise ImportError("Please ensure 'from dbx import fire_query' is available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksPivotQueryBuilder:
    """Dynamic query builder for Databricks PIVOT operations using base queries"""
    
    def __init__(self, base_query_builder):
        self.base_query_builder = base_query_builder
    
    def build_two_step_pivot_query(self, filter_column: str, filter_value: str, 
                                 pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Build query in two steps using base query"""
        try:
            config = self._extract_and_validate_config(pivot_config)
            base_query = self.base_query_builder(filter_column, filter_value, column_filters)
            pivot_column = config['columns_cols'][0]
            
            values_query = f"""
            WITH base_data AS (
                {base_query}
            )
            SELECT DISTINCT {pivot_column}
            FROM base_data
            WHERE {pivot_column} IS NOT NULL
            ORDER BY {pivot_column}
            LIMIT 500
            """
            
            return {
                'step1_get_values': values_query,
                'config': config,
                'base_query': base_query,
                'pivot_column': pivot_column
            }
            
        except Exception as e:
            raise Exception(f"Query construction failed: {str(e)}")
    
    def build_final_pivot_query(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build final pivot query with actual pivot values"""
        try:
            pivot_column = config['columns_cols'][0]
            
            cleaned_values = []
            for val in pivot_values:
                if val is not None:
                    escaped_val = str(val).replace("'", "''")
                    cleaned_values.append(f"'{escaped_val}'")
            
            if not cleaned_values:
                raise Exception("No valid pivot values found")
            
            pivot_in_clause = ', '.join(cleaned_values)
            aggregations = [f"SUM({col}) as sum_{col}" for col in config['value_cols']]
            
            query = f"""
            WITH base_data AS (
                {base_query}
            ),
            filtered_base AS (
                SELECT 
                    {', '.join(config['index_cols'])},
                    {pivot_column},
                    {', '.join(config['value_cols'])}
                FROM base_data
                WHERE {pivot_column} IS NOT NULL
            )
            SELECT * FROM filtered_base
            PIVOT (
                {', '.join(aggregations)}
                FOR {pivot_column} IN ({pivot_in_clause})
            )
            ORDER BY {', '.join(config['index_cols'])}
            """
            
            return query
            
        except Exception as e:
            raise Exception(f"Final pivot query construction failed: {str(e)}")
    
    def build_case_when_pivot_query(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build pivot using CASE WHEN statements for maximum compatibility"""
        try:
            pivot_column = config['columns_cols'][0]
            
            case_statements = []
            for value in pivot_values:
                if value is not None:
                    safe_column_name = str(value).replace(' ', '_').replace('-', '_').replace('.', '_').replace("'", "")
                    safe_column_name = ''.join(c for c in safe_column_name if c.isalnum() or c == '_')
                    
                    for value_col in config['value_cols']:
                        escaped_value = str(value).replace("'", "''")
                        case_statements.append(
                            f"SUM(CASE WHEN {pivot_column} = '{escaped_value}' THEN {value_col} ELSE 0 END) as {safe_column_name}_{value_col}"
                        )
            
            if not case_statements:
                raise Exception("No valid case statements could be built")
            
            query = f"""
            WITH base_data AS (
                {base_query}
            ),
            filtered_base AS (
                SELECT 
                    {', '.join(config['index_cols'])},
                    {pivot_column},
                    {', '.join(config['value_cols'])}
                FROM base_data
                WHERE {pivot_column} IS NOT NULL
            )
            SELECT 
                {', '.join(config['index_cols'])},
                {', '.join(case_statements)}
            FROM filtered_base
            GROUP BY {', '.join(config['index_cols'])}
            ORDER BY {', '.join(config['index_cols'])}
            """
            
            return query
            
        except Exception as e:
            raise Exception(f"Case-when pivot construction failed: {str(e)}")
    
    def _extract_and_validate_config(self, pivot_config: Dict) -> Dict[str, List]:
        """Extract and normalize pivot configuration"""
        index_cols = pivot_config.get('index_cols', [])
        value_cols = pivot_config.get('value_cols', pivot_config.get('value_col', ['Sales']))
        columns_cols = pivot_config.get('columns_cols', pivot_config.get('columns_col', ['Month']))
        
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        if isinstance(columns_cols, str):
            columns_cols = [columns_cols]
        if isinstance(index_cols, str):
            index_cols = [index_cols]
        
        if not index_cols:
            raise ValueError("At least one index column is required")
        if not value_cols:
            raise ValueError("At least one value column is required")
        if not columns_cols:
            raise ValueError("At least one pivot column is required")
        
        return {
            'index_cols': index_cols,
            'value_cols': value_cols,
            'columns_cols': columns_cols
        }


class BaseQueryBuilder:
    """Class to build your custom base queries with multiple joins"""
    
    def __init__(self, base_query_template: str):
        self.base_query_template = base_query_template
    
    def build_base_query(self, filter_column: str, filter_value: str, 
                        column_filters: Optional[Dict] = None) -> str:
        """Build the complete base query with all joins and filters applied"""
        try:
            query = self.base_query_template
            where_conditions = []
            
            # Primary filter
            escaped_filter_value = str(filter_value).replace("'", "''")
            where_conditions.append(f"{filter_column} = '{escaped_filter_value}'")
            
            # Additional column filters
            if column_filters:
                for col, included_values in column_filters.items():
                    if included_values and len(included_values) > 0:
                        escaped_values = []
                        for val in included_values:
                            escaped_val = str(val).replace("'", "''")
                            escaped_values.append(f"'{escaped_val}'")
                        
                        if escaped_values:
                            values_str = ', '.join(escaped_values)
                            where_conditions.append(f"{col} IN ({values_str})")
            
            # Apply WHERE conditions
            if where_conditions:
                where_clause = ' AND '.join(where_conditions)
                if 'WHERE' in query.upper():
                    query += f" AND ({where_clause})"
                else:
                    query += f" WHERE {where_clause}"
            
            return query
            
        except Exception as e:
            raise Exception(f"Base query building failed: {str(e)}")


class DatabricksPivotSystem:
    """Main class for handling Databricks pivot operations using fire_query"""
    
    def __init__(self, base_query_builder: BaseQueryBuilder):
        self.base_query_builder = base_query_builder
        self.query_builder = DatabricksPivotQueryBuilder(
            base_query_builder.build_base_query
        )
    
    def execute_databricks_query(self, query: str) -> pd.DataFrame:
        """Execute query using your external fire_query module"""
        try:
            logger.info(f"Executing query via fire_query: {query[:200]}...")
            start_time = time.time()
            df = fire_query(query)
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f} seconds, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def filter_and_pivot(self, filter_column: str, filter_value: str, 
                        pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Main method to create dynamic pivot using custom base query and fire_query"""
        try:
            logger.info(f"Starting dynamic pivot for {filter_column} = {filter_value}")
            
            # Step 1: Build queries
            step_queries = self.query_builder.build_two_step_pivot_query(
                filter_column, filter_value, pivot_config, column_filters
            )
            
            # Step 2: Get unique pivot values
            values_df = self.execute_databricks_query(step_queries['step1_get_values'])
            if values_df.empty:
                return {'error': f'No pivot values found for {filter_column} = {filter_value}'}
            
            pivot_values = values_df.iloc[:, 0].tolist()
            logger.info(f"Found {len(pivot_values)} unique pivot values")
            
            # Step 3: Choose pivot method
            if len(pivot_values) <= 100:
                try:
                    final_query = self.query_builder.build_final_pivot_query(
                        pivot_values, step_queries['config'], step_queries['base_query']
                    )
                    logger.info("Using native PIVOT method")
                except Exception as e:
                    logger.warning(f"Native PIVOT failed, using CASE WHEN: {e}")
                    final_query = self.query_builder.build_case_when_pivot_query(
                        pivot_values, step_queries['config'], step_queries['base_query']
                    )
            else:
                final_query = self.query_builder.build_case_when_pivot_query(
                    pivot_values, step_queries['config'], step_queries['base_query']
                )
                logger.info("Using CASE WHEN method for large dataset")
            
            # Step 4: Execute pivot query
            pivot_df = self.execute_databricks_query(final_query)
            if pivot_df.empty:
                return {'error': f'No data found after pivot for {filter_column} = {filter_value}'}
            
            logger.info(f"Pivot completed successfully: {len(pivot_df)} result rows")
            
            # Step 5: Convert to frontend format
            return self._create_excel_style_pivot_from_databricks(
                pivot_df, 
                step_queries['config']['index_cols'],
                step_queries['config']['value_cols'],
                step_queries['config']['columns_cols'],
                pivot_values
            )
            
        except Exception as e:
            logger.error(f"Databricks pivot error: {e}")
            return {'error': f'Databricks pivot generation failed: {str(e)}'}
    
    def _create_excel_style_pivot_from_databricks(self, pivot_df: pd.DataFrame, 
                                                index_cols: List[str], value_cols: List[str], 
                                                columns_cols: List[str], pivot_values: List) -> Dict[str, Any]:
        """Convert Databricks pivot result to Excel-style structure for frontend"""
        try:
            row_hierarchy = self._build_row_hierarchy_from_pivot_result(pivot_df, index_cols)
            column_headers = self._build_column_headers_from_pivot_result(pivot_df, index_cols, value_cols, pivot_values)
            pivot_data = self._build_pivot_data_from_result(pivot_df, row_hierarchy, index_cols)
            
            if len(pivot_df) > 0:
                grand_totals = self._calculate_grand_totals(pivot_df, index_cols)
                pivot_data['Grand_Total'] = grand_totals
            
            return {
                'pivot_data': pivot_data,
                'column_headers': column_headers,
                'row_hierarchy': row_hierarchy,
                'hierarchy_levels': index_cols,
                'value_columns': value_cols,
                'column_dimensions': columns_cols,
                'excel_style': True,
                'config_used': {
                    'index_cols': index_cols,
                    'value_cols': value_cols,
                    'columns_cols': columns_cols
                },
                'databricks_optimized': True,
                'uses_fire_query': True,
                'row_count': len(pivot_df),
                'pivot_values_count': len(pivot_values)
            }
            
        except Exception as e:
            logger.error(f"Excel-style conversion failed: {e}")
            raise
    
    def _build_row_hierarchy_from_pivot_result(self, pivot_df: pd.DataFrame, index_cols: List[str]) -> Dict[str, Any]:
        """Build hierarchical structure from pivot result"""
        hierarchy = {}
        for _, row in pivot_df.iterrows():
            current = hierarchy
            path = []
            for i, col in enumerate(index_cols):
                level_value = str(row[col]) if pd.notna(row[col]) else 'NULL'
                path.append(level_value)
                if level_value not in current:
                    current[level_value] = {
                        'level': i,
                        'type': col,
                        'children': {},
                        'is_leaf': i == len(index_cols) - 1,
                        'path': path.copy(),
                        'row_data': row.to_dict() if i == len(index_cols) - 1 else None
                    }
                if i < len(index_cols) - 1:
                    current = current[level_value]['children']
        return hierarchy
    
    def _build_column_headers_from_pivot_result(self, pivot_df: pd.DataFrame, 
                                              index_cols: List[str], value_cols: List[str], 
                                              pivot_values: List) -> Dict[str, Any]:
        """Build column headers from pivot result"""
        data_columns = [col for col in pivot_df.columns if col not in index_cols]
        return {
            'levels': ['Pivoted Values'],
            'structure': {
                'main_groups': data_columns,
                'sub_groups': {},
                'has_subtotals': False
            },
            'data_keys': data_columns
        }
    
    def _build_pivot_data_from_result(self, pivot_df: pd.DataFrame, 
                                    row_hierarchy: Dict, index_cols: List[str]) -> Dict[str, Any]:
        """Build pivot data structure for frontend"""
        pivot_data = {}
        def process_hierarchy(hierarchy, level=0):
            for name, node in hierarchy.items():
                if node['is_leaf'] and node['row_data']:
                    row_key = self._format_row_key_for_frontend(node['path'])
                    data_values = {}
                    for k, v in node['row_data'].items():
                        if k not in index_cols:
                            data_values[k] = 0 if pd.isna(v) else v
                    pivot_data[row_key] = data_values
                if node['children']:
                    process_hierarchy(node['children'], level + 1)
        process_hierarchy(row_hierarchy)
        return pivot_data
    
    def _calculate_grand_totals(self, pivot_df: pd.DataFrame, index_cols: List[str]) -> Dict[str, Any]:
        """Calculate grand totals for numeric columns"""
        grand_totals = {}
        for col in pivot_df.columns:
            if col not in index_cols and pd.api.types.is_numeric_dtype(pivot_df[col]):
                total = pivot_df[col].sum()
                grand_totals[col] = total if not pd.isna(total) else 0
        return grand_totals
    
    def _format_row_key_for_frontend(self, path: List[str]) -> str:
        """Format row key to match frontend expectations"""
        if not path:
            return ''
        return "('" + "', '".join(path) + "')"


# =============================================================================
# INTEGRATION TEMPLATE - REPLACE WITH YOUR ACTUAL VALUES
# =============================================================================

# TODO: Replace this with your actual base query
YOUR_BASE_QUERY = """
SELECT 
    -- TODO: Add your actual columns here
    f.business_unit,
    f.region,
    f.product_name,
    f.month_name,
    f.sales_amount,
    f.profit_amount,
    d.manager_name,
    p.category_name
FROM your_catalog.your_schema.fact_table f
INNER JOIN your_catalog.your_schema.dim_business d ON f.business_unit_id = d.unit_id
INNER JOIN your_catalog.your_schema.dim_products p ON f.product_id = p.product_id
-- TODO: Add your additional joins here
"""

def create_pivot_system():
    """Factory function to create your pivot system"""
    base_query_builder = BaseQueryBuilder(YOUR_BASE_QUERY)
    return DatabricksPivotSystem(base_query_builder)


# =============================================================================
# TEST CONFIGURATION - REPLACE WITH YOUR ACTUAL CONFIG
# =============================================================================

def test_your_pivot():
    """Test function with placeholder config - replace with your actual values"""
    
    # Initialize system
    pivot_system = create_pivot_system()
    
    # TODO: Replace with your actual test configuration
    test_config = {
        'filter_column': 'business_unit',  # TODO: Your filter column
        'filter_value': 'Electronics',     # TODO: Your filter value
        'pivot_config': {
            'index_cols': ['region', 'manager_name'],     # TODO: Your row hierarchy
            'value_cols': ['sales_amount', 'profit_amount'], # TODO: Your metrics
            'columns_cols': ['month_name']                 # TODO: Your pivot column
        },
        'column_filters': {
            'category_name': ['Technology', 'Electronics'], # TODO: Your filters
            'region': ['North', 'South']                    # TODO: Your filters
        }
    }
    
    print("🧪 Testing with your configuration...")
    
    try:
        result = pivot_system.filter_and_pivot(**test_config)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success!")
            print(f"📊 Generated {len(result['pivot_data'])} data points")
            print(f"🌳 Hierarchy levels: {result['hierarchy_levels']}")
            print(f"📈 Row count: {result['row_count']}")
            print(f"🔧 Uses fire_query: {result['uses_fire_query']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("🚀 Databricks Dynamic Pivot System - Ready for Testing")
    print("="*60)
    print("📝 TODO: Update YOUR_BASE_QUERY with your actual query")
    print("📝 TODO: Update test_config with your actual configuration")
    print("📝 TODO: Ensure 'from dbx import fire_query' works")
    print()
    
    # Uncomment to test with your configuration
    # test_your_pivot()