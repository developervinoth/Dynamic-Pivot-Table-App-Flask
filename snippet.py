"""
Simple Databricks Pivot System - Datatype Mismatch Fixed
The issue: Mixed datatypes in aggregations cause binary operation errors
The solution: Cast everything to consistent types BEFORE any operations
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


class SimpleSchemaAnalyzer:
    """Simple schema analyzer to get basic column types"""
    
    def get_column_types(self, base_query: str) -> Dict[str, str]:
        """Get column types from query - simplified approach"""
        try:
            # Method 1: Use DESCRIBE on the query
            describe_query = f"""
            DESCRIBE (
                SELECT * FROM (
                    {base_query}
                ) LIMIT 1
            )
            """
            
            logger.info("Getting column types...")
            schema_df = fire_query(describe_query)
            
            column_types = {}
            for _, row in schema_df.iterrows():
                col_name = row['col_name']
                data_type = str(row['data_type']).lower()
                
                # Simplify to basic categories
                if any(t in data_type for t in ['int', 'bigint', 'smallint', 'tinyint', 'float', 'double', 'decimal', 'numeric']):
                    column_types[col_name] = 'NUMERIC'
                elif any(t in data_type for t in ['date', 'timestamp']):
                    column_types[col_name] = 'DATE'
                elif any(t in data_type for t in ['boolean', 'bool']):
                    column_types[col_name] = 'BOOLEAN'
                else:
                    column_types[col_name] = 'STRING'
            
            logger.info(f"Found {len(column_types)} columns")
            return column_types
            
        except Exception as e:
            logger.warning(f"Schema analysis failed: {e}, using safe defaults")
            # Fallback: analyze sample data
            return self._analyze_sample_data(base_query)
    
    def _analyze_sample_data(self, base_query: str) -> Dict[str, str]:
        """Fallback: analyze using sample data"""
        try:
            sample_query = f"SELECT * FROM ({base_query}) LIMIT 10"
            sample_df = fire_query(sample_query)
            
            column_types = {}
            for col in sample_df.columns:
                # Check sample values to guess type
                sample_vals = sample_df[col].dropna()
                if len(sample_vals) > 0:
                    first_val = sample_vals.iloc[0]
                    if isinstance(first_val, (int, float)):
                        column_types[col] = 'NUMERIC'
                    elif pd.api.types.is_datetime64_any_dtype(sample_vals):
                        column_types[col] = 'DATE'
                    elif isinstance(first_val, bool):
                        column_types[col] = 'BOOLEAN'
                    else:
                        column_types[col] = 'STRING'
                else:
                    column_types[col] = 'STRING'  # Default
            
            return column_types
            
        except Exception as e:
            logger.error(f"Sample analysis failed: {e}")
            return {}


class SimplePivotQueryBuilder:
    """Simple, datatype-safe pivot query builder"""
    
    def __init__(self, base_query_builder):
        self.base_query_builder = base_query_builder
        self.column_types = {}
    
    def set_column_types(self, column_types: Dict[str, str]):
        """Set the column types for safe casting"""
        self.column_types = column_types
        logger.info(f"Set column types for {len(column_types)} columns")
    
    def build_datatype_safe_case_when_pivot(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build CASE WHEN pivot with guaranteed datatype safety"""
        try:
            pivot_column = config['columns_cols'][0]
            index_cols = config['index_cols']
            value_cols = config['value_cols']
            
            # Build case statements with proper type casting
            case_statements = []
            
            for i, pivot_value in enumerate(pivot_values):
                if pivot_value is not None:
                    # Create safe column name
                    safe_col_suffix = f"_{self._make_safe_name(str(pivot_value))}"
                    
                    for value_col in value_cols:
                        col_type = self.column_types.get(value_col, 'NUMERIC')
                        
                        # The key fix: Cast to consistent types BEFORE any operations
                        if col_type == 'NUMERIC':
                            cast_expression = "CAST({} AS DOUBLE)".format(value_col)
                            default_value = "0.0"
                        else:
                            # For non-numeric, just count occurrences
                            cast_expression = "1.0"
                            default_value = "0.0"
                        
                        escaped_pivot_value = str(pivot_value).replace("'", "''")
                        
                        case_statements.append(f"""
                            COALESCE(
                                SUM(
                                    CASE 
                                        WHEN CAST({pivot_column} AS STRING) = '{escaped_pivot_value}' 
                                        THEN {cast_expression}
                                        ELSE {default_value}
                                    END
                                ), 
                                {default_value}
                            ) as {value_col}{safe_col_suffix}
                        """.strip())
            
            if not case_statements:
                raise Exception("No valid case statements generated")
            
            # Build the final query with consistent casting in base data
            query = f"""
            WITH base_data AS (
                {base_query}
            ),
            cleaned_data AS (
                SELECT 
                    {self._build_safe_select_clause(index_cols, value_cols, pivot_column)},
                    CAST({pivot_column} AS STRING) as {pivot_column}
                FROM base_data
                WHERE {pivot_column} IS NOT NULL
            )
            SELECT 
                {', '.join(index_cols)},
                {', '.join(case_statements)}
            FROM cleaned_data
            GROUP BY {', '.join(index_cols)}
            ORDER BY {', '.join(index_cols)}
            """
            
            return query
            
        except Exception as e:
            raise Exception(f"Datatype-safe CASE WHEN pivot failed: {str(e)}")
    
    def _build_safe_select_clause(self, index_cols: List[str], value_cols: List[str], pivot_column: str) -> str:
        """Build SELECT clause with safe casting for all columns"""
        selects = []
        
        # Index columns - cast to STRING for consistency
        for col in index_cols:
            selects.append(f"CAST({col} AS STRING) as {col}")
        
        # Value columns - cast based on type
        for col in value_cols:
            col_type = self.column_types.get(col, 'NUMERIC')
            if col_type == 'NUMERIC':
                selects.append(f"COALESCE(CAST({col} AS DOUBLE), 0.0) as {col}")
            else:
                selects.append(f"CAST({col} AS STRING) as {col}")
        
        return ', '.join(selects)
    
    def _make_safe_name(self, value: str) -> str:
        """Create safe column name suffix"""
        # Replace problematic characters
        safe = str(value).replace(' ', '_').replace('-', '_').replace('.', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c == '_')
        return safe[:30]  # Limit length
    
    def build_simple_pivot_values_query(self, filter_column: str, filter_value: str, 
                                      pivot_config: Dict, column_filters: Optional[Dict] = None) -> str:
        """Build simple query to get pivot values"""
        config = self._extract_config(pivot_config)
        base_query = self.base_query_builder(filter_column, filter_value, column_filters)
        pivot_column = config['columns_cols'][0]
        
        return f"""
        WITH base_data AS (
            {base_query}
        )
        SELECT DISTINCT CAST({pivot_column} AS STRING) as pivot_value
        FROM base_data
        WHERE {pivot_column} IS NOT NULL
          AND TRIM(CAST({pivot_column} AS STRING)) != ''
        ORDER BY CAST({pivot_column} AS STRING)
        LIMIT 500
        """
    
    def _extract_config(self, pivot_config: Dict) -> Dict[str, List]:
        """Extract and validate configuration"""
        index_cols = pivot_config.get('index_cols', [])
        value_cols = pivot_config.get('value_cols', pivot_config.get('value_col', ['Sales']))
        columns_cols = pivot_config.get('columns_cols', pivot_config.get('columns_col', ['Month']))
        
        # Ensure lists
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        if isinstance(columns_cols, str):
            columns_cols = [columns_cols]
        if isinstance(index_cols, str):
            index_cols = [index_cols]
        
        return {
            'index_cols': index_cols,
            'value_cols': value_cols,
            'columns_cols': columns_cols
        }


class BaseQueryBuilder:
    """Simple base query builder"""
    
    def __init__(self, base_query_template: str):
        self.base_query_template = base_query_template
    
    def build_base_query(self, filter_column: str, filter_value: str, 
                        column_filters: Optional[Dict] = None) -> str:
        """Build base query with filters"""
        query = self.base_query_template
        where_conditions = []
        
        # Primary filter
        escaped_filter_value = str(filter_value).replace("'", "''")
        where_conditions.append(f"{filter_column} = '{escaped_filter_value}'")
        
        # Additional filters
        if column_filters:
            for col, values in column_filters.items():
                if values:
                    escaped_values = [f"'{str(v).replace(\"'\", \"''\")}'" for v in values]
                    where_conditions.append(f"{col} IN ({', '.join(escaped_values)})")
        
        # Apply WHERE
        if where_conditions:
            where_clause = ' AND '.join(where_conditions)
            if 'WHERE' in query.upper():
                query += f" AND ({where_clause})"
            else:
                query += f" WHERE {where_clause}"
        
        return query


class SimplePivotSystem:
    """Simple pivot system that prevents datatype mismatches"""
    
    def __init__(self, base_query_builder: BaseQueryBuilder):
        self.base_query_builder = base_query_builder
        self.schema_analyzer = SimpleSchemaAnalyzer()
        self.query_builder = SimplePivotQueryBuilder(base_query_builder.build_base_query)
        self.initialized = False
    
    def initialize_with_sample(self, filter_column: str, filter_value: str, 
                             column_filters: Optional[Dict] = None) -> bool:
        """Initialize by analyzing a sample query"""
        try:
            logger.info("🔍 Analyzing column types...")
            
            # Build sample query
            sample_query = self.base_query_builder.build_base_query(
                filter_column, filter_value, column_filters
            )
            
            # Get column types
            column_types = self.schema_analyzer.get_column_types(sample_query)
            
            if not column_types:
                logger.error("Failed to get column types")
                return False
            
            # Set types in query builder
            self.query_builder.set_column_types(column_types)
            
            # Log the types
            logger.info("📋 Column Types Detected:")
            for col, dtype in column_types.items():
                logger.info(f"  {col}: {dtype}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query safely"""
        try:
            logger.info(f"Executing query: {query[:100]}...")
            start_time = time.time()
            df = fire_query(query)
            exec_time = time.time() - start_time
            logger.info(f"✅ Query executed in {exec_time:.2f}s, {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
    
    def create_pivot(self, filter_column: str, filter_value: str, 
                    pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Create pivot with datatype safety"""
        try:
            if not self.initialized:
                logger.warning("System not initialized, initializing now...")
                if not self.initialize_with_sample(filter_column, filter_value, column_filters):
                    return {'error': 'Failed to initialize system'}
            
            logger.info(f"🚀 Creating pivot for {filter_column} = {filter_value}")
            
            # Step 1: Get pivot values
            values_query = self.query_builder.build_simple_pivot_values_query(
                filter_column, filter_value, pivot_config, column_filters
            )
            
            values_df = self.execute_query(values_query)
            if values_df.empty:
                return {'error': f'No pivot values found for {filter_column} = {filter_value}'}
            
            pivot_values = [v for v in values_df['pivot_value'].tolist() if v is not None and str(v).strip()]
            logger.info(f"Found {len(pivot_values)} pivot values: {pivot_values[:5]}...")
            
            # Step 2: Build base query
            config = self.query_builder._extract_config(pivot_config)
            base_query = self.base_query_builder.build_base_query(
                filter_column, filter_value, column_filters
            )
            
            # Step 3: Create datatype-safe pivot query
            pivot_query = self.query_builder.build_datatype_safe_case_when_pivot(
                pivot_values, config, base_query
            )
            
            # Step 4: Execute pivot
            logger.info("🔄 Executing datatype-safe pivot query...")
            pivot_df = self.execute_query(pivot_query)
            
            if pivot_df.empty:
                return {'error': 'No data returned from pivot query'}
            
            logger.info(f"✅ Pivot successful! {len(pivot_df)} rows returned")
            
            return {
                'pivot_df': pivot_df,
                'success': True,
                'row_count': len(pivot_df),
                'pivot_values_count': len(pivot_values),
                'config_used': config,
                'datatype_safe': True,
                'method': 'case_when_with_type_safety'
            }
            
        except Exception as e:
            logger.error(f"❌ Pivot creation failed: {e}")
            return {'error': f'Pivot failed: {str(e)}'}


# =============================================================================
# SIMPLE USAGE EXAMPLE
# =============================================================================

# Your base query - replace with actual
YOUR_BASE_QUERY = """
SELECT 
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
"""

def create_simple_pivot_system():
    """Create the simple pivot system"""
    base_builder = BaseQueryBuilder(YOUR_BASE_QUERY)
    return SimplePivotSystem(base_builder)


def test_simple_pivot():
    """Test the simple pivot system"""
    
    # Create system
    pivot_system = create_simple_pivot_system()
    
    # Test configuration
    test_config = {
        'filter_column': 'business_unit',
        'filter_value': 'Electronics',
        'pivot_config': {
            'index_cols': ['region', 'manager_name'],
            'value_cols': ['sales_amount', 'profit_amount'],
            'columns_cols': ['month_name']
        },
        'column_filters': {
            'category_name': ['Technology', 'Electronics']
        }
    }
    
    print("🧪 Testing Simple Datatype-Safe Pivot...")
    
    # Initialize system (analyzes column types)
    init_success = pivot_system.initialize_with_sample(
        test_config['filter_column'],
        test_config['filter_value'],
        test_config['column_filters']
    )
    
    if not init_success:
        print("❌ Failed to initialize")
        return
    
    # Create pivot
    result = pivot_system.create_pivot(**test_config)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Success!")
        print(f"📊 Rows: {result['row_count']}")
        print(f"🔢 Pivot values: {result['pivot_values_count']}")
        print(f"🛡️ Datatype safe: {result['datatype_safe']}")
        print(f"⚙️ Method: {result['method']}")


if __name__ == "__main__":
    print("🚀 Simple Datatype-Safe Pivot System")
    print("="*50)
    print("❌ Problem: Mixed datatypes cause binary operation errors")
    print("✅ Solution: Cast everything to consistent types BEFORE operations")
    print("🔍 Method: Analyze column types, then use safe casting")
    print()
    
    # Uncomment to test
    # test_simple_pivot()