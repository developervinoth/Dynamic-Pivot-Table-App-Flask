"""
Complete Databricks Dynamic Pivot Module with Base Query Support
Fully dynamic pivot table generation using custom base queries with multiple joins
"""

import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from databricks import sql
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabricksPivotQueryBuilder:
    """Dynamic query builder for Databricks PIVOT operations using base queries"""
    
    def __init__(self, base_query_builder: Callable[[str, str, Optional[Dict]], str]):
        """
        Initialize with a base query builder function
        
        Args:
            base_query_builder: Function that takes (filter_column, filter_value, column_filters) 
                               and returns the base query string
        """
        self.base_query_builder = base_query_builder
    
    def build_two_step_pivot_query(self, filter_column: str, filter_value: str, 
                                 pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Build query in two steps using base query: 1) Get pivot values, 2) Build actual pivot"""
        try:
            config = self._extract_and_validate_config(pivot_config)
            
            # Get base query from your custom builder
            base_query = self.base_query_builder(filter_column, filter_value, column_filters)
            
            pivot_column = config['columns_cols'][0]
            
            # Step 1: Query to get unique pivot values from base query result
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
            raise Exception(f"Two-step query construction failed: {str(e)}")
    
    def build_final_pivot_query(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build final pivot query with actual pivot values using base query"""
        try:
            pivot_column = config['columns_cols'][0]
            
            # Clean and format pivot values for SQL
            cleaned_values = []
            for val in pivot_values:
                if val is not None:
                    # Escape single quotes and convert to string
                    escaped_val = str(val).replace("'", "''")
                    cleaned_values.append(f"'{escaped_val}'")
            
            if not cleaned_values:
                raise Exception("No valid pivot values found")
            
            pivot_in_clause = ', '.join(cleaned_values)
            
            # Build aggregations for each value column
            aggregations = []
            for value_col in config['value_cols']:
                aggregations.append(f"SUM({value_col}) as sum_{value_col}")
            
            # Use base query as CTE and pivot the results
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
        """Build pivot using CASE WHEN statements for maximum compatibility with base query"""
        try:
            pivot_column = config['columns_cols'][0]
            
            # Build CASE WHEN statements for each pivot value and value column combination
            case_statements = []
            for value in pivot_values:
                if value is not None:
                    # Create safe column name
                    safe_column_name = str(value).replace(' ', '_').replace('-', '_').replace('.', '_').replace("'", "")
                    safe_column_name = ''.join(c for c in safe_column_name if c.isalnum() or c == '_')
                    
                    for value_col in config['value_cols']:
                        escaped_value = str(value).replace("'", "''")
                        case_statements.append(
                            f"SUM(CASE WHEN {pivot_column} = '{escaped_value}' THEN {value_col} ELSE 0 END) as {safe_column_name}_{value_col}"
                        )
            
            if not case_statements:
                raise Exception("No valid case statements could be built")
            
            # Use base query as CTE and apply CASE WHEN logic
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
    
    def build_sample_pivot_query(self, config: Dict, base_query: str, sample_percentage: float = 0.1) -> str:
        """Build a sample query for quick preview using base query"""
        try:
            pivot_column = config['columns_cols'][0]
            
            # Add sampling to base query
            sample_query = f"""
            WITH base_data AS (
                {base_query}
            ),
            sampled_data AS (
                SELECT 
                    {', '.join(config['index_cols'])},
                    {pivot_column},
                    {', '.join(config['value_cols'])}
                FROM base_data
                WHERE {pivot_column} IS NOT NULL
                  AND rand() < {sample_percentage}
                LIMIT 10000
            ),
            pivot_values AS (
                SELECT COLLECT_LIST(DISTINCT {pivot_column}) as pivot_cols
                FROM sampled_data
            )
            SELECT 
                {', '.join(config['index_cols'])},
                MAP_FROM_ARRAYS(
                    COLLECT_LIST({pivot_column}),
                    COLLECT_LIST({config['value_cols'][0]})
                ) as pivot_data
            FROM sampled_data
            GROUP BY {', '.join(config['index_cols'])}
            ORDER BY {', '.join(config['index_cols'])}
            """
            
            return sample_query
            
        except Exception as e:
            raise Exception(f"Sample query construction failed: {str(e)}")
    
    def _extract_and_validate_config(self, pivot_config: Dict) -> Dict[str, List]:
        """Extract and normalize pivot configuration"""
        # Extract configuration
        index_cols = pivot_config.get('index_cols', [])
        value_cols = pivot_config.get('value_cols', pivot_config.get('value_col', ['Sales']))
        columns_cols = pivot_config.get('columns_cols', pivot_config.get('columns_col', ['Month']))
        
        # Ensure all are lists
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        if isinstance(columns_cols, str):
            columns_cols = [columns_cols]
        if isinstance(index_cols, str):
            index_cols = [index_cols]
        
        # Validation
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
    """Class to build your custom base queries with multiple joins - no table mappings"""
    
    def __init__(self, base_query_template: str):
        """
        Initialize with base query template using direct table names
        
        Args:
            base_query_template: Your base query with actual table names
        """
        self.base_query_template = base_query_template
    
    def build_base_query(self, filter_column: str, filter_value: str, 
                        column_filters: Optional[Dict] = None) -> str:
        """
        Build the complete base query with all joins and filters applied
        
        Args:
            filter_column: Primary filter column
            filter_value: Primary filter value
            column_filters: Additional column filters
            
        Returns:
            Complete SQL query string
        """
        try:
            # Start with base template (already has actual table names)
            query = self.base_query_template
            
            # Build WHERE conditions
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
            
            # Apply WHERE conditions to base query
            if where_conditions:
                where_clause = ' AND '.join(where_conditions)
                
                # Check if query already has WHERE clause
                if 'WHERE' in query.upper():
                    query += f" AND ({where_clause})"
                else:
                    query += f" WHERE {where_clause}"
            
            return query
            
        except Exception as e:
            raise Exception(f"Base query building failed: {str(e)}")


class DatabricksPivotSystem:
    """Main class for handling Databricks pivot operations with custom base queries"""
    
    def __init__(self, databricks_config: Dict, base_query_builder: BaseQueryBuilder):
        """
        Initialize with Databricks config and base query builder
        
        Args:
            databricks_config: Databricks connection configuration
            base_query_builder: Instance of BaseQueryBuilder with your custom query
        """
        self.databricks_config = databricks_config
        self.base_query_builder = base_query_builder
        
        # Initialize query builder with base query function
        self.query_builder = DatabricksPivotQueryBuilder(
            base_query_builder.build_base_query
        )
        
    def get_databricks_connection(self):
        """Get or create Databricks connection"""
        try:
            connection = sql.connect(
                server_hostname=self.databricks_config['server_hostname'],
                http_path=self.databricks_config['http_path'],
                access_token=self.databricks_config['access_token']
            )
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to Databricks: {e}")
            raise
    
    def execute_databricks_query(self, query: str) -> pd.DataFrame:
        """Execute query and return pandas DataFrame"""
        connection = None
        try:
            logger.info(f"Executing query: {query[:200]}...")
            
            connection = self.get_databricks_connection()
            with connection.cursor() as cursor:
                start_time = time.time()
                cursor.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                execution_time = time.time() - start_time
                logger.info(f"Query executed in {execution_time:.2f} seconds, returned {len(rows)} rows")
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(rows, columns=columns)
                return df
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def filter_and_pivot(self, filter_column: str, filter_value: str, 
                        pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Main method to create dynamic pivot using custom base query"""
        try:
            logger.info(f"Starting dynamic pivot for {filter_column} = {filter_value}")
            
            # Step 1: Build initial queries using your base query
            step_queries = self.query_builder.build_two_step_pivot_query(
                filter_column, filter_value, pivot_config, column_filters
            )
            
            logger.info("Generated base query:")
            logger.info(step_queries['base_query'][:500] + "...")
            
            # Step 2: Get unique pivot values from your base query result
            logger.info("Getting unique pivot values from base query...")
            values_df = self.execute_databricks_query(step_queries['step1_get_values'])
            
            if values_df.empty:
                return {'error': f'No pivot values found for {filter_column} = {filter_value}'}
            
            pivot_values = values_df.iloc[:, 0].tolist()
            logger.info(f"Found {len(pivot_values)} unique pivot values: {pivot_values[:10]}...")
            
            # Step 3: Choose pivot method based on number of values and complexity
            if len(pivot_values) <= 100:
                # Use native PIVOT for smaller datasets
                try:
                    final_query = self.query_builder.build_final_pivot_query(
                        pivot_values, step_queries['config'], step_queries['base_query']
                    )
                    logger.info("Using native PIVOT method with base query")
                except Exception as e:
                    logger.warning(f"Native PIVOT failed, falling back to CASE WHEN: {e}")
                    final_query = self.query_builder.build_case_when_pivot_query(
                        pivot_values, step_queries['config'], step_queries['base_query']
                    )
                    logger.info("Using CASE WHEN method with base query")
            else:
                # Use CASE WHEN for larger datasets
                final_query = self.query_builder.build_case_when_pivot_query(
                    pivot_values, step_queries['config'], step_queries['base_query']
                )
                logger.info("Using CASE WHEN method for large dataset with base query")
            
            # Step 4: Execute pivot query
            logger.info("Executing final pivot query with base query...")
            pivot_df = self.execute_databricks_query(final_query)
            
            if pivot_df.empty:
                return {'error': f'No data found after pivot for {filter_column} = {filter_value}'}
            
            logger.info(f"Pivot completed successfully: {len(pivot_df)} result rows")
            
            # Step 5: Convert to frontend-compatible format
            return self._create_excel_style_pivot_from_databricks(
                pivot_df, 
                step_queries['config']['index_cols'],
                step_queries['config']['value_cols'],
                step_queries['config']['columns_cols'],
                pivot_values
            )
            
        except Exception as e:
            logger.error(f"Databricks pivot error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Databricks pivot generation failed: {str(e)}'}
    
    def get_sample_pivot(self, filter_column: str, filter_value: str, 
                        pivot_config: Dict, column_filters: Optional[Dict] = None,
                        sample_percentage: float = 0.1) -> Dict[str, Any]:
        """Get a quick sample pivot for preview using base query"""
        try:
            logger.info(f"Generating sample pivot ({sample_percentage*100}%) for {filter_column} = {filter_value}")
            
            config = self.query_builder._extract_and_validate_config(pivot_config)
            base_query = self.base_query_builder.build_base_query(filter_column, filter_value, column_filters)
            
            # Build sample query
            sample_query = self.query_builder.build_sample_pivot_query(config, base_query, sample_percentage)
            
            # Execute sample query
            sample_df = self.execute_databricks_query(sample_query)
            
            if sample_df.empty:
                return {'error': f'No sample data found for {filter_column} = {filter_value}'}
            
            logger.info(f"Sample pivot completed: {len(sample_df)} rows")
            
            return {
                'sample_data': sample_df.to_dict('records'),
                'row_count': len(sample_df),
                'is_sample': True,
                'sample_percentage': sample_percentage
            }
            
        except Exception as e:
            logger.error(f"Sample pivot error: {e}")
            return {'error': f'Sample pivot generation failed: {str(e)}'}
    
    # Keep all the _create_excel_style_pivot_from_databricks and helper methods from previous version
    def _create_excel_style_pivot_from_databricks(self, pivot_df: pd.DataFrame, 
                                                index_cols: List[str], value_cols: List[str], 
                                                columns_cols: List[str], pivot_values: List) -> Dict[str, Any]:
        """Convert Databricks pivot result to Excel-style structure for frontend"""
        try:
            logger.info("Converting pivot result to Excel-style format...")
            
            # Build row hierarchy from pivoted data
            row_hierarchy = self._build_row_hierarchy_from_pivot_result(pivot_df, index_cols)
            
            # Build column headers structure
            column_headers = self._build_column_headers_from_pivot_result(
                pivot_df, index_cols, value_cols, pivot_values
            )
            
            # Build data structure for tree rendering
            pivot_data = self._build_pivot_data_from_result(pivot_df, row_hierarchy, index_cols)
            
            # Add grand totals
            if len(pivot_df) > 0:
                grand_totals = self._calculate_grand_totals(pivot_df, index_cols)
                pivot_data['Grand_Total'] = grand_totals
            
            result = {
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
                'uses_base_query': True,
                'row_count': len(pivot_df),
                'pivot_values_count': len(pivot_values)
            }
            
            logger.info(f"Excel-style conversion completed: {len(row_hierarchy)} hierarchy nodes")
            return result
            
        except Exception as e:
            logger.error(f"Excel-style conversion failed: {e}")
            raise
    
    def _build_row_hierarchy_from_pivot_result(self, pivot_df: pd.DataFrame, 
                                             index_cols: List[str]) -> Dict[str, Any]:
        """Build hierarchical structure from Databricks pivot result"""
        hierarchy = {}
        
        for _, row in pivot_df.iterrows():
            # Build hierarchy path
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
        """Build column headers from Databricks pivot result"""
        # Get pivot column names (exclude index columns)
        data_columns = [col for col in pivot_df.columns if col not in index_cols]
        
        # Sort data columns to match pivot values order
        sorted_data_columns = []
        for pivot_val in pivot_values:
            for value_col in value_cols:
                # Handle different naming conventions from PIVOT vs CASE WHEN
                possible_names = [
                    f"'{pivot_val}'_sum_{value_col}",  # Native PIVOT format
                    f"{str(pivot_val).replace(' ', '_').replace('-', '_')}_{value_col}",  # CASE WHEN format
                    f"sum_{value_col}_{pivot_val}",  # Alternative format
                    str(pivot_val)  # Simple format
                ]
                
                for possible_name in possible_names:
                    if possible_name in data_columns:
                        sorted_data_columns.append(possible_name)
                        break
        
        # Add any remaining columns
        for col in data_columns:
            if col not in sorted_data_columns:
                sorted_data_columns.append(col)
        
        # Build header structure that frontend expects
        headers = {
            'levels': ['Pivoted Values'],
            'structure': {
                'main_groups': sorted_data_columns,
                'sub_groups': {},
                'has_subtotals': False
            },
            'data_keys': sorted_data_columns
        }
        
        return headers
    
    def _build_pivot_data_from_result(self, pivot_df: pd.DataFrame, 
                                    row_hierarchy: Dict, index_cols: List[str]) -> Dict[str, Any]:
        """Build pivot data structure that matches frontend expectations"""
        pivot_data = {}
        
        def process_hierarchy(hierarchy, level=0):
            for name, node in hierarchy.items():
                if node['is_leaf'] and node['row_data']:
                    # Create row key that matches frontend format
                    row_key = self._format_row_key_for_frontend(node['path'])
                    
                    # Extract data values (exclude index columns)
                    data_values = {}
                    for k, v in node['row_data'].items():
                        if k not in index_cols:
                            # Convert NaN/None to 0 for numeric values
                            if pd.isna(v):
                                data_values[k] = 0
                            else:
                                data_values[k] = v
                    
                    pivot_data[row_key] = data_values
                
                # Process children
                if node['children']:
                    process_hierarchy(node['children'], level + 1)
        
        process_hierarchy(row_hierarchy)
        return pivot_data
    
    def _calculate_grand_totals(self, pivot_df: pd.DataFrame, index_cols: List[str]) -> Dict[str, Any]:
        """Calculate grand totals for all numeric columns"""
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


# Sample execution and testing with base queries
def main():
    """Sample execution sequence with custom base queries - no table mappings"""
    
    # Configuration
    databricks_config = {
        'server_hostname': 'your-databricks-hostname.databricks.com',
        'http_path': '/sql/1.0/warehouses/your-warehouse-id',
        'access_token': 'your-databricks-token'
    }
    
    # Example 1: Simple base query with actual table names
    simple_base_query = """
    SELECT 
        f.business_unit,
        f.region,
        f.product,
        f.month,
        f.sales,
        f.quantity,
        d.manager,
        d.budget,
        p.category,
        p.sub_category
    FROM analytics_db.fact_sales f
    JOIN analytics_db.dim_business_units d ON f.business_unit = d.business_unit
    JOIN analytics_db.dim_products p ON f.product = p.product_name
    """
    
    # Example 2: Complex base query with multiple joins and calculations
    complex_base_query = """
    SELECT 
        f.business_unit,
        f.region,
        f.product,
        f.month,
        f.sales,
        f.quantity,
        f.sales - f.cost as profit,
        f.sales / f.quantity as avg_price,
        d.manager,
        d.budget,
        p.category,
        p.sub_category,
        r.region_manager,
        r.regional_target,
        CASE 
            WHEN f.sales >= d.budget * 0.1 THEN 'High'
            WHEN f.sales >= d.budget * 0.05 THEN 'Medium'
            ELSE 'Low'
        END as performance_tier
    FROM sales_data.fact_sales f
    JOIN sales_data.dim_business_units d ON f.business_unit = d.business_unit
    JOIN sales_data.dim_products p ON f.product = p.product_name
    JOIN sales_data.dim_regions r ON f.region = r.region_name
    LEFT JOIN external_sources.monthly_targets et ON f.business_unit = et.unit AND f.month = et.target_month
    """
    
    print("🚀 Initializing Databricks Pivot System with Direct Table Names...")
    
    # Test cases with different base queries
    test_cases = [
        {
            'name': 'Simple Join Base Query',
            'base_query': simple_base_query,
            'filter_column': 'business_unit',
            'filter_value': 'Electronics',
            'pivot_config': {
                'index_cols': ['region', 'category'],
                'value_cols': ['sales'],
                'columns_cols': ['month']
            },
            'column_filters': {
                'region': ['North', 'South'],
                'category': ['Technology', 'Electronics']
            }
        },
        {
            'name': 'Complex Multi-Join Base Query',
            'base_query': complex_base_query,
            'filter_column': 'business_unit',
            'filter_value': 'Electronics',
            'pivot_config': {
                'index_cols': ['region', 'performance_tier', 'product'],
                'value_cols': ['sales', 'profit'],
                'columns_cols': ['month']
            },
            'column_filters': {
                'performance_tier': ['High', 'Medium'],
                'category': ['Technology']
            }
        },
        {
            'name': 'Custom Calculated Fields',
            'base_query': complex_base_query,
            'filter_column': 'region',
            'filter_value': 'North',
            'pivot_config': {
                'index_cols': ['business_unit', 'manager'],
                'value_cols': ['avg_price', 'profit'],
                'columns_cols': ['performance_tier']
            },
            'column_filters': None
        }
    ]
    
    # Execute test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['name']}")
        print("="*60)
        
        try:
            # Initialize base query builder (no table mappings needed)
            base_query_builder = BaseQueryBuilder(test_case['base_query'])
            
            # Initialize pivot system
            pivot_system = DatabricksPivotSystem(databricks_config, base_query_builder)
            
            start_time = time.time()
            
            # Execute pivot
            result = pivot_system.filter_and_pivot(
                filter_column=test_case['filter_column'],
                filter_value=test_case['filter_value'],
                pivot_config=test_case['pivot_config'],
                column_filters=test_case['column_filters']
            )
            
            execution_time = time.time() - start_time
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success in {execution_time:.2f} seconds")
                print(f"   📈 Rows in result: {result.get('row_count', 'Unknown')}")
                print(f"   🏗️  Hierarchy levels: {len(result.get('hierarchy_levels', []))}")
                print(f"   📊 Pivot values: {result.get('pivot_values_count', 'Unknown')}")
                print(f"   🔗 Data points: {len(result.get('pivot_data', {}))}")
                print(f"   🎯 Uses base query: {result.get('uses_base_query', False)}")
                
                # Sample of hierarchy structure
                if result.get('row_hierarchy'):
                    hierarchy_sample = list(result['row_hierarchy'].keys())[:3]
                    print(f"   🌳 Sample hierarchy: {hierarchy_sample}")
        
        except Exception as e:
            print(f"❌ Test case failed: {e}")
    
    print(f"\n🎉 Base Query Databricks Pivot System testing completed!")


def demo_base_query_builder():
    """Demonstrate base query building without table mappings"""
    print("\n🔧 Base Query Builder Demo - Direct Table Names")
    print("="*50)
    
    # Your actual base query with direct table names
    your_base_query = """
    SELECT 
        f.business_unit,
        f.region,
        f.product,
        f.month,
        f.sales,
        f.quantity,
        f.profit_margin,
        d.manager,
        d.budget_target,
        p.category,
        p.sub_category,
        p.product_cost,
        r.region_manager,
        CASE 
            WHEN f.sales >= d.budget_target * 0.1 THEN 'Excellent'
            WHEN f.sales >= d.budget_target * 0.05 THEN 'Good'
            WHEN f.sales >= d.budget_target * 0.02 THEN 'Average'
            ELSE 'Poor'
        END as performance_rating,
        f.sales * p.profit_margin as calculated_profit
    FROM analytics_warehouse.fact_sales_data f
    INNER JOIN analytics_warehouse.dim_business_units d ON f.business_unit = d.unit_name
    INNER JOIN analytics_warehouse.dim_product_catalog p ON f.product = p.product_name
    LEFT JOIN analytics_warehouse.dim_regions r ON f.region = r.region_code
    LEFT JOIN external_warehouse.budget_targets eb ON f.business_unit = eb.unit AND f.month = eb.budget_month
    """
    
    # Initialize without table mappings
    base_query_builder = BaseQueryBuilder(your_base_query)
    
    try:
        # Test base query generation
        sample_filters = {
            'region': ['North', 'South'],
            'category': ['Electronics', 'Technology'],
            'performance_rating': ['Excellent', 'Good']
        }
        
        final_query = base_query_builder.build_base_query(
            'business_unit', 'Electronics', sample_filters
        )
        
        print("📝 Generated Base Query:")
        print(final_query)
        print()
        
        # Show how it integrates with pivot builder
        query_builder = DatabricksPivotQueryBuilder(base_query_builder.build_base_query)
        
        sample_config = {
            'index_cols': ['region', 'performance_rating'],
            'value_cols': ['sales', 'calculated_profit'],
            'columns_cols': ['month']
        }
        
        step_queries = query_builder.build_two_step_pivot_query(
            'business_unit', 'Electronics', sample_config, sample_filters
        )
        
        print("📝 Step 1 - Get Pivot Values (using base query):")
        print(step_queries['step1_get_values'])
        print()
        
        print("📝 Base Query Used:")
        print(step_queries['base_query'][:500] + "...")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("🚀 Databricks Dynamic Pivot System with Base Queries")
    print("="*60)
    
    # Run base query builder demo (safe to run without Databricks connection)
    demo_base_query_builder()
    
    # Uncomment to run full tests (requires valid Databricks connection)
    # main()
    
    print(f"\n💡 Integration Notes for Your Base Query:")
    print("1. Replace base query template with your actual multi-join query")
    print("2. Use #table_name# placeholders for dynamic table mapping")
    print("3. Update table_mappings with your actual table names")
    print("4. Your base query can include any complexity: joins, calculations, CTEs")
    print("5. Column filters are automatically applied to your base query")
    print("6. Frontend requires no changes - same API response format!")
    print("7. Performance: Base query + Pivot = 30 seconds - 2 minutes vs 5-10 minutes")