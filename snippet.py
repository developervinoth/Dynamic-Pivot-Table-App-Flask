"""
Databricks Dynamic Pivot Module - Metadata-Aware Version (Fixed)
Analyzes table schemas first to prevent datatype mismatch errors
Uses your existing fire_query module for execution
"""

import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
import re

# Import your existing Databricks query module
try:
    from dbx import fire_query
except ImportError:
    def fire_query(sql_query: str) -> pd.DataFrame:
        raise ImportError("Please ensure 'from dbx import fire_query' is available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksSchemaAnalyzer:
    """Analyzes Databricks table schemas to understand data types"""
    
    def __init__(self):
        self.schema_cache = {}
        self.numeric_types = {
            'int', 'bigint', 'smallint', 'tinyint', 'integer',
            'float', 'double', 'decimal', 'numeric', 'real'
        }
        self.string_types = {'string', 'varchar', 'char', 'text'}
        self.date_types = {'date', 'timestamp', 'datetime'}
        self.boolean_types = {'boolean', 'bool'}
    
    def analyze_query_schema(self, base_query: str) -> Dict[str, Dict]:
        """Analyze the schema of columns returned by a query"""
        try:
            # Method 1: Try DESCRIBE on the query
            info_query = f"""
            DESCRIBE (
                SELECT * FROM (
                    {base_query}
                ) LIMIT 1
            )
            """
            
            logger.info("Analyzing query schema using DESCRIBE...")
            schema_df = fire_query(info_query)
            
            column_metadata = {}
            for _, row in schema_df.iterrows():
                col_name = row['col_name']
                data_type = str(row['data_type']).lower()
                
                column_metadata[col_name] = {
                    'name': col_name,
                    'type': data_type,
                    'category': self._categorize_type(data_type),
                    'is_numeric': self._is_numeric_type(data_type),
                    'is_string': self._is_string_type(data_type),
                    'is_date': self._is_date_type(data_type),
                    'needs_casting': self._needs_special_casting(data_type),
                    'safe_cast_target': self._get_safe_cast_target(data_type)
                }
            
            logger.info(f"Analyzed {len(column_metadata)} columns using DESCRIBE")
            return column_metadata
            
        except Exception as e:
            logger.warning(f"DESCRIBE method failed: {e}, trying sample data approach...")
            # Fallback: analyze with sample data
            return self._analyze_with_sample_data(base_query)
    
    def analyze_table_schema(self, table_name: str) -> Dict[str, Dict]:
        """Analyze schema of a specific table"""
        try:
            if table_name in self.schema_cache:
                return self.schema_cache[table_name]
            
            describe_query = f"DESCRIBE {table_name}"
            schema_df = fire_query(describe_query)
            
            column_metadata = {}
            for _, row in schema_df.iterrows():
                col_name = row['col_name']
                data_type = str(row['data_type']).lower()
                
                column_metadata[col_name] = {
                    'name': col_name,
                    'type': data_type,
                    'category': self._categorize_type(data_type),
                    'is_numeric': self._is_numeric_type(data_type),
                    'is_string': self._is_string_type(data_type),
                    'is_date': self._is_date_type(data_type),
                    'needs_casting': self._needs_special_casting(data_type),
                    'safe_cast_target': self._get_safe_cast_target(data_type)
                }
            
            self.schema_cache[table_name] = column_metadata
            logger.info(f"Cached schema for {table_name}: {len(column_metadata)} columns")
            return column_metadata
            
        except Exception as e:
            logger.error(f"Table schema analysis failed for {table_name}: {e}")
            return {}
    
    def _analyze_with_sample_data(self, base_query: str) -> Dict[str, Dict]:
        """Fallback method: analyze using sample data"""
        try:
            sample_query = f"""
            SELECT * FROM (
                {base_query}
            ) LIMIT 100
            """
            
            sample_df = fire_query(sample_query)
            column_metadata = {}
            
            for col in sample_df.columns:
                # Infer type from pandas
                dtype = str(sample_df[col].dtype)
                sample_values = sample_df[col].dropna().head(10).tolist()
                
                inferred_type = self._infer_databricks_type(dtype, sample_values)
                
                column_metadata[col] = {
                    'name': col,
                    'type': inferred_type,
                    'category': self._categorize_type(inferred_type),
                    'is_numeric': self._is_numeric_type(inferred_type),
                    'is_string': self._is_string_type(inferred_type),
                    'is_date': self._is_date_type(inferred_type),
                    'needs_casting': True,  # Conservative approach
                    'safe_cast_target': self._get_safe_cast_target(inferred_type),
                    'inferred': True
                }
            
            logger.info(f"Inferred schema from sample data: {len(column_metadata)} columns")
            return column_metadata
            
        except Exception as e:
            logger.error(f"Sample data analysis failed: {e}")
            return {}
    
    def _categorize_type(self, data_type: str) -> str:
        """Categorize data type into broad categories"""
        data_type = data_type.lower()
        
        if any(t in data_type for t in self.numeric_types):
            return 'numeric'
        elif any(t in data_type for t in self.string_types):
            return 'string'
        elif any(t in data_type for t in self.date_types):
            return 'date'
        elif any(t in data_type for t in self.boolean_types):
            return 'boolean'
        else:
            return 'unknown'
    
    def _is_numeric_type(self, data_type: str) -> bool:
        """Check if data type is numeric"""
        return any(t in data_type.lower() for t in self.numeric_types)
    
    def _is_string_type(self, data_type: str) -> bool:
        """Check if data type is string-like"""
        return any(t in data_type.lower() for t in self.string_types)
    
    def _is_date_type(self, data_type: str) -> bool:
        """Check if data type is date/time"""
        return any(t in data_type.lower() for t in self.date_types)
    
    def _needs_special_casting(self, data_type: str) -> bool:
        """Determine if type needs special handling"""
        problematic_types = ['decimal', 'numeric', 'timestamp', 'array', 'struct', 'map']
        return any(t in data_type.lower() for t in problematic_types)
    
    def _get_safe_cast_target(self, data_type: str) -> str:
        """Get the safest cast target for operations"""
        data_type = data_type.lower()
        
        if self._is_numeric_type(data_type):
            return 'DOUBLE'
        elif self._is_date_type(data_type):
            return 'STRING'  # Convert dates to strings for pivoting
        else:
            return 'STRING'
    
    def _infer_databricks_type(self, pandas_dtype: str, sample_values: List) -> str:
        """Infer Databricks type from pandas dtype and sample values"""
        if 'int' in pandas_dtype:
            return 'bigint'
        elif 'float' in pandas_dtype:
            return 'double'
        elif 'bool' in pandas_dtype:
            return 'boolean'
        elif 'datetime' in pandas_dtype:
            return 'timestamp'
        else:
            # Check sample values for more clues
            if sample_values:
                first_val = str(sample_values[0])
                if re.match(r'^\d{4}-\d{2}-\d{2}', first_val):
                    return 'date'
            return 'string'


class MetadataAwarePivotQueryBuilder:
    """Query builder that uses metadata to create type-safe queries"""
    
    def __init__(self, base_query_builder, schema_analyzer: DatabricksSchemaAnalyzer):
        self.base_query_builder = base_query_builder
        self.schema_analyzer = schema_analyzer
        self.column_metadata = None
    
    def initialize_with_metadata(self, base_query: str) -> bool:
        """Initialize the builder with metadata from the base query"""
        try:
            self.column_metadata = self.schema_analyzer.analyze_query_schema(base_query)
            logger.info(f"Initialized with metadata for {len(self.column_metadata)} columns")
            
            # Log column types for debugging
            for col, meta in self.column_metadata.items():
                logger.debug(f"{col}: {meta['type']} ({meta['category']})")
            
            return len(self.column_metadata) > 0
            
        except Exception as e:
            logger.error(f"Metadata initialization failed: {e}")
            return False
    
    def build_two_step_pivot_query(self, filter_column: str, filter_value: str, 
                                 pivot_config: Dict, column_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Build query in two steps using metadata"""
        try:
            config = self._extract_and_validate_config(pivot_config)
            base_query = self.base_query_builder(filter_column, filter_value, column_filters)
            pivot_column = config['columns_cols'][0]
            
            # Get metadata for pivot column
            pivot_meta = self.column_metadata.get(pivot_column, {})
            cast_target = pivot_meta.get('safe_cast_target', 'STRING')
            
            values_query = f"""
            WITH base_data AS (
                {base_query}
            )
            SELECT DISTINCT CAST({pivot_column} AS {cast_target}) as pivot_value
            FROM base_data
            WHERE {pivot_column} IS NOT NULL
              AND TRIM(CAST({pivot_column} AS STRING)) != ''
            ORDER BY CAST({pivot_column} AS {cast_target})
            LIMIT 500
            """
            
            return {
                'step1_get_values': values_query,
                'config': config,
                'base_query': base_query,
                'pivot_column': pivot_column,
                'pivot_metadata': pivot_meta
            }
            
        except Exception as e:
            raise Exception(f"Metadata-aware query construction failed: {str(e)}")
    
    def build_type_safe_pivot_query(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build pivot query using metadata for type safety"""
        try:
            if not self.column_metadata:
                raise Exception("Metadata not initialized. Call initialize_with_metadata first.")
            
            pivot_column = config['columns_cols'][0]
            pivot_meta = self.column_metadata.get(pivot_column, {})
            
            # Build type-safe value list
            safe_pivot_values = self._build_safe_value_list(pivot_values, pivot_meta)
            
            if not safe_pivot_values:
                raise Exception("No valid pivot values after type conversion")
            
            # Build type-safe aggregations
            aggregations = self._build_safe_aggregations(config['value_cols'])
            
            # Build type-safe base query
            typed_selects = self._build_typed_selects(config)
            
            # Build filter clause
            filter_clause = self._build_safe_filter(pivot_column, pivot_meta, safe_pivot_values)
            
            query = f"""
            WITH base_data AS (
                {base_query}
            ),
            typed_base AS (
                SELECT 
                    {typed_selects}
                FROM base_data
                WHERE {filter_clause}
            )
            SELECT 
                {', '.join(config['index_cols'])},
                {', '.join([f'COALESCE({agg.split(" as ")[1]}, 0.0) as {agg.split(" as ")[1]}' for agg in aggregations])}
            FROM typed_base
            PIVOT (
                {', '.join(aggregations)}
                FOR {pivot_column} IN ({', '.join(safe_pivot_values)})
            )
            ORDER BY {', '.join(config['index_cols'])}
            """
            
            return query
            
        except Exception as e:
            # Fallback to case-when approach
            logger.warning(f"Type-safe pivot failed, using case-when: {e}")
            return self._build_case_when_with_metadata(pivot_values, config, base_query)
    
    def _build_safe_value_list(self, pivot_values: List, pivot_meta: Dict) -> List[str]:
        """Build type-safe pivot value list based on metadata"""
        safe_values = []
        target_type = pivot_meta.get('safe_cast_target', 'STRING')
        
        for val in pivot_values:
            if val is not None:
                try:
                    if target_type == 'STRING':
                        escaped_val = str(val).replace("'", "''").strip()
                        if escaped_val:
                            safe_values.append(f"'{escaped_val}'")
                    elif target_type == 'DOUBLE':
                        # For numeric pivots, ensure proper formatting
                        if isinstance(val, (int, float)):
                            safe_values.append(str(val))
                        else:
                            # Try to convert to number
                            num_val = float(str(val))
                            safe_values.append(str(num_val))
                    else:
                        # Default to string
                        escaped_val = str(val).replace("'", "''")
                        safe_values.append(f"'{escaped_val}'")
                        
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert pivot value: {val}")
                    continue
        
        return safe_values
    
    def _build_safe_aggregations(self, value_cols: List[str]) -> List[str]:
        """Build type-safe aggregations based on column metadata"""
        aggregations = []
        
        for col in value_cols:
            col_meta = self.column_metadata.get(col, {})
            
            if col_meta.get('is_numeric', True):
                target_type = col_meta.get('safe_cast_target', 'DOUBLE')
                aggregations.append(f"SUM(COALESCE(CAST({col} AS {target_type}), 0.0)) as sum_{col}")
            else:
                # For non-numeric columns, count occurrences
                aggregations.append(f"COUNT({col}) as count_{col}")
        
        return aggregations
    
    def _build_typed_selects(self, config: Dict) -> str:
        """Build SELECT clause with proper type casting"""
        selects = []
        
        # Index columns
        for col in config['index_cols']:
            col_meta = self.column_metadata.get(col, {})
            target_type = col_meta.get('safe_cast_target', 'STRING')
            selects.append(f"CAST({col} AS {target_type}) as {col}")
        
        # Pivot column
        pivot_col = config['columns_cols'][0]
        pivot_meta = self.column_metadata.get(pivot_col, {})
        pivot_target = pivot_meta.get('safe_cast_target', 'STRING')
        selects.append(f"CAST({pivot_col} AS {pivot_target}) as {pivot_col}")
        
        # Value columns
        for col in config['value_cols']:
            col_meta = self.column_metadata.get(col, {})
            if col_meta.get('is_numeric', True):
                target_type = col_meta.get('safe_cast_target', 'DOUBLE')
                selects.append(f"COALESCE(CAST({col} AS {target_type}), 0.0) as {col}")
            else:
                selects.append(f"CAST({col} AS STRING) as {col}")
        
        return ', '.join(selects)
    
    def _build_safe_filter(self, pivot_column: str, pivot_meta: Dict, safe_values: List[str]) -> str:
        """Build type-safe WHERE filter"""
        target_type = pivot_meta.get('safe_cast_target', 'STRING')
        values_clause = ', '.join(safe_values)
        
        return f"""
            {pivot_column} IS NOT NULL 
            AND CAST({pivot_column} AS {target_type}) IN ({values_clause})
        """
    
    def _build_case_when_with_metadata(self, pivot_values: List, config: Dict, base_query: str) -> str:
        """Build case-when query using metadata"""
        try:
            pivot_column = config['columns_cols'][0]
            pivot_meta = self.column_metadata.get(pivot_column, {})
            
            case_statements = []
            for i, value in enumerate(pivot_values):
                if value is not None:
                    safe_col_name = f"col_{i}_{str(value).replace(' ', '_').replace('-', '_')}"
                    safe_col_name = ''.join(c for c in safe_col_name if c.isalnum() or c == '_')[:60]
                    
                    for value_col in config['value_cols']:
                        col_meta = self.column_metadata.get(value_col, {})
                        
                        if col_meta.get('is_numeric', True):
                            cast_expr = f"COALESCE(CAST({value_col} AS DOUBLE), 0.0)"
                        else:
                            cast_expr = "1"  # Count for non-numeric
                        
                        escaped_value = str(value).replace("'", "''")
                        case_statements.append(
                            f"SUM(CASE WHEN CAST({pivot_column} AS STRING) = '{escaped_value}' THEN {cast_expr} ELSE 0 END) as {safe_col_name}_{value_col}"
                        )
            
            typed_selects = self._build_typed_selects(config)
            
            query = f"""
            WITH base_data AS (
                {base_query}
            ),
            typed_base AS (
                SELECT {typed_selects}
                FROM base_data
                WHERE {pivot_column} IS NOT NULL
            )
            SELECT 
                {', '.join(config['index_cols'])},
                {', '.join(case_statements)}
            FROM typed_base
            GROUP BY {', '.join(config['index_cols'])}
            ORDER BY {', '.join(config['index_cols'])}
            """
            
            return query
            
        except Exception as e:
            raise Exception(f"Case-when with metadata failed: {str(e)}")
    
    def _extract_and_validate_config(self, pivot_config: Dict) -> Dict[str, List]:
        """Extract and validate pivot configuration"""
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
    
    def get_column_info(self) -> Dict[str, Dict]:
        """Get current column metadata"""
        return self.column_metadata or {}
    
    def print_schema_summary(self):
        """Print a summary of the analyzed schema"""
        if not self.column_metadata:
            print("No metadata available")
            return
        
        print("\n📋 Schema Summary:")
        print("=" * 50)
        
        by_category = {}
        for col, meta in self.column_metadata.items():
            category = meta['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(f"{col} ({meta['type']})")
        
        for category, columns in by_category.items():
            print(f"\n{category.upper()} columns ({len(columns)}):")
            for col in columns:
                print(f"  • {col}")


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
                            if val is not None and str(val).strip():
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


class MetadataAwarePivotSystem:
    """Main pivot system that uses metadata for type safety"""
    
    def __init__(self, base_query_builder: BaseQueryBuilder):
        self.base_query_builder = base_query_builder
        self.schema_analyzer = DatabricksSchemaAnalyzer()
        self.query_builder = None
    
    def initialize_system(self, sample_filter_column: str, sample_filter_value: str, 
                         sample_column_filters: Optional[Dict] = None) -> bool:
        """Initialize the system by analyzing metadata from a sample query"""
        try:
            logger.info("🔍 Initializing system with metadata analysis...")
            
            # Build a sample query to analyze
            sample_query = self.base_query_builder.build_base_query(
                sample_filter_column, sample_filter_value, sample_column_filters
            )
            
            # Initialize query builder with metadata
            self.query_builder = MetadataAwarePivotQueryBuilder(
                self.base_query_builder.build_base_query, self.schema_analyzer
            )
            
            success = self.query_builder.initialize_with_metadata(sample_query)
            
            if success:
                logger.info("✅ System initialized with metadata")
                self.query_builder.print_schema_summary()
            else:
                logger.error("❌ Failed to initialize metadata")
            
            return success
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def execute_databricks_query(self, query: str) -> pd.DataFrame:
        """Execute query using fire_query"""
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
        """Execute pivot with metadata-aware type safety"""
        try:
            if not self.query_builder:
                raise Exception("System not initialized. Call initialize_system first.")
            
            logger.info(f"Starting metadata-aware pivot for {filter_column} = {filter_value}")
            
            # Step 1: Build queries using metadata
            step_queries = self.query_builder.build_two_step_pivot_query(
                filter_column, filter_value, pivot_config, column_filters
            )
            
            # Step 2: Get pivot values
            values_df = self.execute_databricks_query(step_queries['step1_get_values'])
            if values_df.empty:
                return {'error': f'No pivot values found for {filter_column} = {filter_value}'}
            
            pivot_values = [val for val in values_df.iloc[:, 0].tolist() if val is not None]
            logger.info(f"Found {len(pivot_values)} unique pivot values")
            
            if not pivot_values:
                return {'error': 'No valid pivot values found after filtering'}
            
            # Step 3: Build type-safe pivot query
            final_query = self.query_builder.build_type_safe_pivot_query(
                pivot_values, step_queries['config'], step_queries['base_query']
            )
            
            # Step 4: Execute pivot
            pivot_df = self.execute_databricks_query(final_query)
            if pivot_df.empty:
                return {'error': f'No data found after pivot for {filter_column} = {filter_value}'}
            
            logger.info(f"✅ Metadata-aware pivot completed: {len(pivot_df)} result rows")
            
            # Step 5: Create result structure
            return self._create_excel_style_pivot_from_databricks(
                pivot_df, 
                step_queries['config']['index_cols'],
                step_queries['config']['value_cols'],
                step_queries['config']['columns_cols'],
                pivot_values
            )
            
        except Exception as e:
            logger.error(f"Metadata-aware pivot failed: {e}")
            return {'error': f'Pivot generation failed: {str(e)}'}
    
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
                'metadata_used': True,
                'column_metadata': self.query_builder.get_column_info(),
                'pivot_values_count': len(pivot_values),
                'row_count': len(pivot_df),
                'type_safe': True,
                'uses_fire_query': True
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
                            # Ensure numeric values are properly handled
                            if pd.isna(v) or v is None:
                                data_values[k] = 0.0
                            else:
                                try:
                                    data_values[k] = float(v) if isinstance(v, (int, float, str)) else 0.0
                                except (ValueError, TypeError):
                                    data_values[k] = 0.0
                    pivot_data[row_key] = data_values
                if node['children']:
                    process_hierarchy(node['children'], level + 1)
        process_hierarchy(row_hierarchy)
        return pivot_data
    
    def _calculate_grand_totals(self, pivot_df: pd.DataFrame, index_cols: List[str]) -> Dict[str, Any]:
        """Calculate grand totals for numeric columns"""
        grand_totals = {}
        for col in pivot_df.columns:
            if col not in index_cols:
                try:
                    # Try to sum numeric values, default to 0 if failed
                    numeric_series = pd.to_numeric(pivot_df[col], errors='coerce').fillna(0)
                    total = numeric_series.sum()
                    grand_totals[col] = float(total) if not pd.isna(total) else 0.0
                except Exception:
                    grand_totals[col] = 0.0
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

def create_metadata_aware_system():
    """Create metadata-aware pivot system"""
    base_builder = BaseQueryBuilder(YOUR_BASE_QUERY)
    return MetadataAwarePivotSystem(base_builder)


def test_with_metadata():
    """Test the metadata-aware system"""
    
    # Create system
    pivot_system = create_metadata_aware_system()
    
    # Initialize with sample data to get metadata
    print("🔍 Initializing system with metadata analysis...")
    init_success = pivot_system.initialize_system(
        sample_filter_column='business_unit',
        sample_filter_value='Electronics',
        sample_column_filters={'region': ['North']}
    )
    
    if not init_success:
        print("❌ Failed to initialize system with metadata")
        return
    
    # Now run actual pivot
    print("🚀 Running metadata-aware pivot...")
    result = pivot_system.filter_and_pivot(
        filter_column='business_unit',
        filter_value='Electronics',
        pivot_config={
            'index_cols': ['region', 'manager_name'],
            'value_cols': ['sales_amount', 'profit_amount'],
            'columns_cols': ['month_name']
        },
        column_filters={'category_name': ['Technology']}
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Success with metadata!")
        print(f"📊 Type-safe: {result['type_safe']}")
        print(f"📈 Rows: {result['row_count']}")
        print(f"🔧 Uses metadata: {result['metadata_used']}")
        print(f"🛡️ Uses fire_query: {result['uses_fire_query']}")


if __name__ == "__main__":
    print("🚀 Metadata-Aware Databricks Pivot System - Fixed Version")
    print("="*60)
    print("🔍 Analyzes table schemas first for type safety")
    print("🛡️ Prevents datatype mismatch errors")
    print("✅ Complete and syntactically correct")
    print()
    
    # Uncomment to test
    # test_with_metadata()