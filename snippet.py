def filter_and_pivot(self, filter_column, filter_value, pivot_config, column_filters=None):
    """Filter fact table and create Excel-style hierarchical pivot using Databricks PIVOT"""
    try:
        # Step 1: Build Databricks PIVOT query
        query = self._build_databricks_pivot_query(
            filter_column, filter_value, pivot_config, column_filters
        )
        
        print(f"Executing Databricks PIVOT query for {filter_column} = {filter_value}")
        
        # Step 2: Execute query in Databricks
        pivot_df = self._execute_databricks_query(query)
        
        if pivot_df.empty:
            return {'error': f'No data found for {filter_column} = {filter_value}'}
        
        print(f"Databricks PIVOT returned {len(pivot_df)} rows")
        
        # Step 3: Convert Databricks result to hierarchical structure
        # This maintains compatibility with your existing frontend
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
        
        # Step 4: Process pivoted data into Excel-style structure
        # This keeps your frontend working exactly the same!
        return self._create_excel_style_pivot_from_databricks(
            pivot_df, index_cols, value_cols, columns_cols
        )
        
    except Exception as e:
        print(f"Databricks pivot error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Databricks pivot generation failed: {str(e)}'}

def _build_databricks_pivot_query(self, filter_column, filter_value, pivot_config, column_filters=None):
    """Build dynamic Databricks PIVOT query"""
    
    # Extract configuration
    index_cols = pivot_config.get('index_cols', [])
    value_cols = pivot_config.get('value_cols', ['Sales'])
    columns_cols = pivot_config.get('columns_cols', ['Month'])
    
    # Ensure lists
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    if isinstance(columns_cols, str):
        columns_cols = [columns_cols]
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    
    # Get unique values for PIVOT IN clause
    pivot_column = columns_cols[0]  # Use first column for pivot
    pivot_values = self._get_pivot_column_values(filter_column, filter_value, pivot_column, column_filters)
    
    # Build column filters WHERE clause
    where_clause = f"{filter_column} = '{filter_value}'"
    if column_filters:
        for col, included_values in column_filters.items():
            if included_values:
                values_str = "', '".join([str(v) for v in included_values])
                where_clause += f" AND {col} IN ('{values_str}')"
    
    # Build PIVOT query
    query = f"""
    WITH filtered_data AS (
        SELECT 
            {', '.join(index_cols)},
            {pivot_column},
            {', '.join(value_cols)}
        FROM your_fact_table_name
        WHERE {where_clause}
    )
    SELECT * FROM filtered_data
    PIVOT (
        SUM({value_cols[0]}) as {value_cols[0]}
        FOR {pivot_column} IN ({', '.join([f"'{v}'" for v in pivot_values])})
    )
    ORDER BY {', '.join(index_cols)}
    """
    
    return query

def _get_pivot_column_values(self, filter_column, filter_value, pivot_column, column_filters=None):
    """Get unique values for PIVOT IN clause"""
    
    # Build WHERE clause for getting distinct values
    where_clause = f"{filter_column} = '{filter_value}'"
    if column_filters and pivot_column in column_filters:
        included_values = column_filters[pivot_column]
        if included_values:
            values_str = "', '".join([str(v) for v in included_values])
            where_clause += f" AND {pivot_column} IN ('{values_str}')"
    
    query = f"""
    SELECT DISTINCT {pivot_column}
    FROM your_fact_table_name
    WHERE {where_clause}
    ORDER BY {pivot_column}
    LIMIT 50
    """
    
    result_df = self._execute_databricks_query(query)
    return result_df[pivot_column].tolist()

def _execute_databricks_query(self, query):
    """Execute query using your existing Databricks connection"""
    # Use your existing Databricks SQL method
    # Replace this with your actual Databricks execution code
    
    # Example using databricks-sql-connector:
    from databricks import sql
    
    with sql.connect(
        server_hostname=self.databricks_config['server_hostname'],
        http_path=self.databricks_config['http_path'],
        access_token=self.databricks_config['access_token']
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to pandas DataFrame
            import pandas as pd
            return pd.DataFrame(rows, columns=columns)

def _create_excel_style_pivot_from_databricks(self, pivot_df, index_cols, value_cols, columns_cols):
    """Convert Databricks pivot result to Excel-style structure for frontend"""
    
    # This is the key: Transform Databricks PIVOT result into the same format
    # your frontend expects, maintaining all tree functionality
    
    # Step 1: Build row hierarchy from pivoted data
    row_hierarchy = self._build_row_hierarchy_from_pivot_result(pivot_df, index_cols)
    
    # Step 2: Build column headers structure
    column_headers = self._build_column_headers_from_pivot_result(pivot_df, index_cols, value_cols)
    
    # Step 3: Build data structure for tree rendering
    pivot_data = self._build_pivot_data_from_result(pivot_df, row_hierarchy, index_cols)
    
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
        }
    }

def _build_row_hierarchy_from_pivot_result(self, pivot_df, index_cols):
    """Build hierarchical structure from Databricks pivot result"""
    hierarchy = {}
    
    for _, row in pivot_df.iterrows():
        # Build hierarchy path
        current = hierarchy
        path = []
        
        for i, col in enumerate(index_cols):
            level_value = str(row[col])
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

def _build_column_headers_from_pivot_result(self, pivot_df, index_cols, value_cols):
    """Build column headers from Databricks pivot result"""
    
    # Get pivot column names (exclude index columns)
    data_columns = [col for col in pivot_df.columns if col not in index_cols]
    
    # Build header structure that frontend expects
    headers = {
        'levels': ['Pivoted Values'],
        'structure': {
            'main_groups': data_columns,
            'sub_groups': {},
            'has_subtotals': False
        },
        'data_keys': data_columns
    }
    
    return headers

def _build_pivot_data_from_result(self, pivot_df, row_hierarchy, index_cols):
    """Build pivot data structure that matches frontend expectations"""
    pivot_data = {}
    
    def process_hierarchy(hierarchy, level=0):
        for name, node in hierarchy.items():
            if node['is_leaf'] and node['row_data']:
                # Create row key that matches frontend format
                row_key = self._format_row_key_for_frontend(node['path'])
                
                # Extract data values (exclude index columns)
                data_values = {k: v for k, v in node['row_data'].items() 
                             if k not in index_cols}
                
                pivot_data[row_key] = data_values
            
            # Process children
            if node['children']:
                process_hierarchy(node['children'], level + 1)
    
    process_hierarchy(row_hierarchy)
    
    return pivot_data

def _format_row_key_for_frontend(self, path):
    """Format row key to match frontend expectations"""
    if not path:
        return ''
    return "('" + "', '".join(path) + "')"