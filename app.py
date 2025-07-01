from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import re
from itertools import product

app = Flask(__name__)

class DynamicPivotSystem:
    def __init__(self):
        dim_df = pd.read_excel("data.xlsx", sheet_name = 'dim')
        fact_df = pd.read_excel('data.xlsx', sheet_name = 'fact')
        
        self.dimension_df, self.fact_df = dim_df, fact_df
    
    def create_sample_data(self):
        """Create sample dimension and fact tables"""
        np.random.seed(42)
        
        # Dimension Table - Business Units/Categories
        dimension_data = [
            {'Business_Unit': 'Electronics', 'Manager': 'John Smith', 'Budget': 5000000, 'Target': 6000000},
            {'Business_Unit': 'Furniture', 'Manager': 'Sarah Johnson', 'Budget': 2000000, 'Target': 2500000},
            {'Business_Unit': 'Clothing', 'Manager': 'Mike Chen', 'Budget': 3000000, 'Target': 3500000},
            {'Business_Unit': 'Sports', 'Manager': 'Lisa Wang', 'Budget': 1500000, 'Target': 2000000},
            {'Business_Unit': 'Books', 'Manager': 'David Brown', 'Budget': 800000, 'Target': 1000000},
        ]
        dimension_df = pd.DataFrame(dimension_data)
        
        # Fact Table - Detailed sales transactions
        fact_data = []
        regions = ['North', 'South', 'East', 'West']
        business_units = ['Electronics', 'Furniture', 'Clothing', 'Sports', 'Books']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        
        products = {
            'Electronics': ['Laptop', 'Mobile', 'Tablet', 'TV', 'Camera'],
            'Furniture': ['Chair', 'Table', 'Sofa', 'Bed', 'Desk'],
            'Clothing': ['Shirt', 'Pants', 'Dress', 'Jacket', 'Shoes'],
            'Sports': ['Football', 'Basketball', 'Tennis', 'Golf', 'Swimming'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Biography']
        }
        
        for business_unit in business_units:
            for region in regions:
                for product in products[business_unit]:
                    for month in months:
                        if np.random.random() > 0.2:  # 80% chance of having sales
                            if business_unit == 'Electronics':
                                sales = np.random.randint(50000, 300000)
                            elif business_unit == 'Furniture':
                                sales = np.random.randint(20000, 150000)
                            elif business_unit == 'Clothing':
                                sales = np.random.randint(15000, 80000)
                            elif business_unit == 'Sports':
                                sales = np.random.randint(10000, 60000)
                            else:  # Books
                                sales = np.random.randint(5000, 30000)
                            
                            fact_data.append({
                                'Business_Unit': business_unit,
                                'Region': region,
                                'Product': product,
                                'Month': month,
                                'Sales': sales,
                                'Quantity': np.random.randint(10, 100),
                                'Cost': int(sales * 0.7)  # 70% cost ratio
                            })
        
        fact_df = pd.DataFrame(fact_data)
        return dimension_df, fact_df
    
    def get_dimension_table(self):
        """Return dimension table as dict"""
        return {
            'columns': list(self.dimension_df.columns),
            'data': self.dimension_df.to_dict('records')
        }
    
    def filter_and_pivot(self, filter_column, filter_value, pivot_config):
        """Filter fact table and create Excel-style hierarchical pivot"""
        try:
            # Filter the fact dataframe
            filtered_df = self.fact_df[self.fact_df[filter_column] == filter_value].copy()
            
            if filtered_df.empty:
                return {'error': f'No data found for {filter_column} = {filter_value}'}
            
            # Extract pivot configuration
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
            
            print(f"Original config - Index: {index_cols}, Values: {value_cols}, Columns: {columns_cols}")
            
            # Clean the configuration
            valid_columns = set(filtered_df.columns)
            
            index_cols = [col for col in index_cols 
                         if col != filter_column and col in valid_columns]
            value_cols = [col for col in value_cols 
                         if col != filter_column and col in valid_columns and pd.api.types.is_numeric_dtype(filtered_df[col])]
            columns_cols = [col for col in columns_cols 
                           if col != filter_column and col in valid_columns]
            
            # Remove overlaps
            for col in columns_cols:
                if col in index_cols:
                    index_cols.remove(col)
                if col in value_cols:
                    value_cols.remove(col)
            
            for col in value_cols:
                if col in index_cols:
                    index_cols.remove(col)
            
            print(f"Final config - Index: {index_cols}, Values: {value_cols}, Columns: {columns_cols}")
            
            # Validation
            if not index_cols or not value_cols or not columns_cols:
                return self._handle_missing_config(filtered_df, filter_column, index_cols, value_cols, columns_cols)
            
            # Create Excel-style hierarchical pivot
            return self._create_excel_style_pivot(filtered_df, index_cols, value_cols, columns_cols)
                
        except Exception as e:
            print(f"Pivot error details: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Pivot generation failed: {str(e)}'}
    
    def _handle_missing_config(self, df, filter_column, index_cols, value_cols, columns_cols):
        """Handle missing configuration by setting defaults"""
        if not index_cols:
            available_cols = [col for col in df.columns 
                            if col not in [filter_column] + value_cols + columns_cols]
            if available_cols:
                index_cols = [available_cols[0]]
            else:
                return {'error': 'No suitable columns found for pivot index'}
        
        if not value_cols:
            numeric_cols = [col for col in df.columns 
                          if pd.api.types.is_numeric_dtype(df[col]) and col != filter_column]
            if numeric_cols:
                value_cols = [numeric_cols[0]]
            else:
                return {'error': 'No numeric columns found for values'}
        
        if not columns_cols:
            available_cols = [col for col in df.columns 
                            if col not in [filter_column] + index_cols + value_cols]
            if available_cols:
                columns_cols = [available_cols[0]]
            else:
                return {'error': 'No suitable columns found for pivot columns'}
        
        return self._create_excel_style_pivot(df, index_cols, value_cols, columns_cols)
    
    def _create_excel_style_pivot(self, df, index_cols, value_cols, columns_cols):
        """Create Excel-style pivot with hierarchical columns and proper totals"""
        try:
            # Create pivot table for each value column
            pivots = {}
            for value_col in value_cols:
                pivot = pd.pivot_table(
                    df,
                    values=value_col,
                    index=index_cols,
                    columns=columns_cols,
                    aggfunc='sum',
                    fill_value=0,
                    margins=True,
                    margins_name='Grand Total'
                )
                pivots[value_col] = pivot
            
            # Process the pivot data into Excel-style structure
            result = self._process_excel_structure(pivots, index_cols, value_cols, columns_cols)
            
            return {
                'pivot_data': result['data'],
                'column_headers': result['headers'],
                'row_hierarchy': result['row_hierarchy'],
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
            
        except Exception as e:
            print(f"Excel-style pivot error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Excel-style pivot generation failed: {str(e)}'}
    
    def _process_excel_structure(self, pivots, index_cols, value_cols, columns_cols):
        """Process pivots into Excel-style structure with proper headers and totals"""
        
        # Get the first pivot to understand structure
        main_pivot = list(pivots.values())[0]
        
        # Build column header structure
        headers = self._build_excel_headers(main_pivot, value_cols, columns_cols)
        
        # Build row hierarchy with collapsible structure
        row_hierarchy = self._build_row_hierarchy(main_pivot, index_cols)
        
        # Build data structure
        data = {}
        
        # Process each row in the pivot
        for row_key in main_pivot.index:
            if row_key == 'Grand Total':
                continue
                
            row_data = {}
            
            # Process each value column
            for value_col in value_cols:
                pivot = pivots[value_col]
                
                # Process each column in the pivot
                for col_key in pivot.columns:
                    if col_key == 'Grand Total':
                        continue
                    
                    # Create unique key for this data point
                    if len(value_cols) > 1:
                        data_key = f"{value_col}_{self._format_column_key(col_key)}"
                    else:
                        data_key = self._format_column_key(col_key)
                    
                    value = pivot.loc[row_key, col_key]
                    row_data[data_key] = self._safe_convert_value(value)
                
                # Add subtotals for this value column
                if len(columns_cols) > 1:
                    subtotals = self._calculate_subtotals(pivot, row_key, value_col, len(value_cols) > 1)
                    row_data.update(subtotals)
                
                # Add grand total for this row - only ONE grand total per row
                if 'Grand Total' in pivot.columns and value_col == value_cols[0]:
                    grand_total_key = "Grand_Total"  # Always use simple key for single grand total
                    value = pivot.loc[row_key, 'Grand Total']
                    row_data[grand_total_key] = self._safe_convert_value(value)
            
            # Store row data
            row_key_str = self._format_row_key(row_key)
            data[row_key_str] = row_data
        
        # Add Grand Total row
        if 'Grand Total' in main_pivot.index:
            grand_total_row = {}
            
            for value_col in value_cols:
                pivot = pivots[value_col]
                
                for col_key in pivot.columns:
                    if col_key == 'Grand Total':
                        continue
                    
                    if len(value_cols) > 1:
                        data_key = f"{value_col}_{self._format_column_key(col_key)}"
                    else:
                        data_key = self._format_column_key(col_key)
                    
                    value = pivot.loc['Grand Total', col_key]
                    grand_total_row[data_key] = self._safe_convert_value(value)
                
                # Add subtotals for grand total row
                if len(columns_cols) > 1:
                    subtotals = self._calculate_subtotals(pivot, 'Grand Total', value_col, len(value_cols) > 1)
                    grand_total_row.update(subtotals)
                
                # Add overall grand total - only ONE grand total
                if 'Grand Total' in pivot.columns and value_col == value_cols[0]:
                    grand_total_key = "Grand_Total"  # Always use simple key for single grand total
                    value = pivot.loc['Grand Total', 'Grand Total']
                    grand_total_row[grand_total_key] = self._safe_convert_value(value)
            
            data['Grand_Total'] = grand_total_row
        
        return {
            'data': data,
            'headers': headers,
            'row_hierarchy': row_hierarchy
        }
    
    def _build_excel_headers(self, pivot, value_cols, columns_cols):
        """Build Excel-style hierarchical column headers"""
        headers = {
            'levels': [],
            'structure': {},
            'data_keys': []
        }
        
        # Get unique values for each column dimension
        if len(columns_cols) == 1:
            # Single column dimension
            col_values = [str(c) for c in pivot.columns if str(c) != 'Grand Total']
            headers['levels'] = [columns_cols[0]]
            headers['structure'] = {
                'main_groups': sorted(col_values),
                'sub_groups': {},
                'has_subtotals': False
            }
            
            # Build data keys
            for col_val in sorted(col_values):
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        headers['data_keys'].append(f"{value_col}_{col_val}")
                else:
                    headers['data_keys'].append(col_val)
            
        else:
            # Multi-level column dimensions
            headers['levels'] = columns_cols
            main_groups = {}
            
            for col in pivot.columns:
                if col == 'Grand Total':
                    continue
                
                if isinstance(col, tuple):
                    main_col = str(col[0])
                    sub_col = str(col[1]) if len(col) > 1 else ''
                    
                    if main_col not in main_groups:
                        main_groups[main_col] = set()
                    
                    if sub_col:
                        main_groups[main_col].add(sub_col)
                else:
                    main_col = str(col)
                    if main_col not in main_groups:
                        main_groups[main_col] = set()
            
            # Convert sets to sorted lists
            for main_col in main_groups:
                main_groups[main_col] = sorted(list(main_groups[main_col]))
            
            headers['structure'] = {
                'main_groups': sorted(main_groups.keys()),
                'sub_groups': main_groups,
                'has_subtotals': any(len(subs) > 1 for subs in main_groups.values())
            }
            
            # Build data keys
            for main_col in sorted(main_groups.keys()):
                sub_cols = main_groups[main_col]
                if sub_cols:
                    for sub_col in sub_cols:
                        if len(value_cols) > 1:
                            for value_col in value_cols:
                                headers['data_keys'].append(f"{value_col}_{main_col}_{sub_col}")
                        else:
                            headers['data_keys'].append(f"{main_col}_{sub_col}")
                    
                    # Add subtotal if multiple sub-columns
                    if len(sub_cols) > 1:
                        if len(value_cols) > 1:
                            for value_col in value_cols:
                                headers['data_keys'].append(f"{value_col}_{main_col}_Total")
                        else:
                            headers['data_keys'].append(f"{main_col}_Total")
                else:
                    if len(value_cols) > 1:
                        for value_col in value_cols:
                            headers['data_keys'].append(f"{value_col}_{main_col}")
                    else:
                        headers['data_keys'].append(main_col)
        
        # Add only ONE Grand Total at the end
        headers['data_keys'].append("Grand_Total")
        
        return headers
    
    def _build_row_hierarchy(self, pivot, index_cols):
        """Build hierarchical row structure for collapsible rows"""
        hierarchy = {}
        
        for row_key in pivot.index:
            if row_key == 'Grand Total':
                continue
            
            if isinstance(row_key, tuple):
                levels = [str(level) for level in row_key]
            else:
                levels = [str(row_key)]
            
            # Build hierarchy tree
            current = hierarchy
            for i, level in enumerate(levels):
                if level not in current:
                    current[level] = {
                        'level': i,
                        'type': index_cols[i] if i < len(index_cols) else 'item',
                        'children': {},
                        'is_leaf': i == len(levels) - 1,
                        'path': levels[:i+1]
                    }
                
                if i < len(levels) - 1:
                    current = current[level]['children']
        
        return hierarchy
    
    def _safe_convert_value(self, value):
        """Safely convert pandas value to Python int, handling Series and NaN"""
        try:
            # Handle pandas Series by getting the first item
            if hasattr(value, 'iloc'):
                value = value.iloc[0] if len(value) > 0 else 0
            elif hasattr(value, 'item'):
                value = value.item()
            
            # Check for NaN using pandas method
            if pd.isna(value):
                return 0
            
            # Convert to int if not zero
            return int(float(value)) if value != 0 else 0
        except (ValueError, TypeError, AttributeError):
            return 0
    
    def _calculate_subtotals(self, pivot, row_key, value_col, include_value_prefix):
        """Calculate subtotals for main column groups"""
        subtotals = {}
        
        # Group columns by main category
        main_groups = {}
        for col in pivot.columns:
            if col == 'Grand Total':
                continue
            
            if isinstance(col, tuple):
                main_col = str(col[0])
                if main_col not in main_groups:
                    main_groups[main_col] = []
                main_groups[main_col].append(col)
        
        # Calculate subtotals for each main group
        for main_col, group_cols in main_groups.items():
            if len(group_cols) > 1:  # Only create subtotals if multiple sub-columns
                subtotal = 0
                has_value = False
                
                for group_col in group_cols:
                    try:
                        value = pivot.loc[row_key, group_col]
                        converted_value = self._safe_convert_value(value)
                        if converted_value != 0:
                            subtotal += converted_value
                            has_value = True
                    except (KeyError, IndexError):
                        continue
                
                if has_value:
                    subtotal_key = f"{value_col}_{main_col}_Total" if include_value_prefix else f"{main_col}_Total"
                    subtotals[subtotal_key] = subtotal
        
        return subtotals
    
    def _format_column_key(self, col_key):
        """Format column key for data storage"""
        if isinstance(col_key, tuple):
            return '_'.join(str(c) for c in col_key)
        return str(col_key)
    
    def _format_row_key(self, row_key):
        """Format row key for data storage"""
        if isinstance(row_key, tuple):
            return f"({', '.join(repr(str(level)) for level in row_key)})"
        return f"('{row_key}')"

    def get_available_pivot_columns(self, filter_column):
        """Get columns available for pivot configuration, excluding the filter column"""
        fact_columns = list(self.fact_df.columns)
        
        # Separate columns by type
        categorical_cols = []
        numeric_cols = []
        
        for col in fact_columns:
            if col == filter_column:
                continue  # Skip the filter column
                
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.fact_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return {
            'all_columns': fact_columns,
            'categorical_columns': categorical_cols,
            'numeric_columns': numeric_cols,
            'filter_column': filter_column
        }

# Initialize the system
pivot_system = DynamicPivotSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dimensions')
def get_dimensions():
    """Get dimension table data"""
    try:
        return jsonify(pivot_system.get_dimension_table())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pivot', methods=['POST'])
def get_pivot():
    """Get filtered and pivoted data with Excel-style hierarchical columns"""
    try:
        data = request.get_json()
        print(f"Received pivot request: {data}")
        
        filter_column = data.get('filter_column')
        filter_value = data.get('filter_value')
        pivot_config = data.get('pivot_config', {})
        
        if not filter_column or not filter_value:
            return jsonify({'error': 'filter_column and filter_value are required'}), 400
        
        # Convert single column configs to list format for backward compatibility
        if 'index_cols' not in pivot_config and 'index_col' in pivot_config:
            pivot_config['index_cols'] = [pivot_config['index_col']]
        if 'value_cols' not in pivot_config and 'value_col' in pivot_config:
            pivot_config['value_cols'] = [pivot_config['value_col']]
        if 'columns_cols' not in pivot_config and 'columns_col' in pivot_config:
            pivot_config['columns_cols'] = [pivot_config['columns_col']]
        
        print(f"Filtering by {filter_column} = {filter_value}")
        print(f"Pivot config: {pivot_config}")
        
        result = pivot_system.filter_and_pivot(filter_column, filter_value, pivot_config)
        
        if 'error' in result:
            print(f"Pivot error: {result['error']}")
            return jsonify(result), 400
        
        print(f"Excel-style pivot successful.")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Server error in pivot generation: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/pivot-columns', methods=['POST'])
def get_pivot_columns():
    """Get available columns for pivot configuration based on filter selection"""
    try:
        data = request.get_json()
        filter_column = data.get('filter_column')
        
        if not filter_column:
            return jsonify({'error': 'filter_column is required'}), 400
        
        columns_info = pivot_system.get_available_pivot_columns(filter_column)
        return jsonify(columns_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter-values', methods=['POST'])
def get_filter_values():
    """Get unique values for Excel-style filtering"""
    try:
        data = request.get_json()
        filter_column = data.get('filter_column')
        
        if not filter_column:
            return jsonify({'error': 'filter_column is required'}), 400
            
        if filter_column not in pivot_system.fact_df.columns:
            return jsonify({'error': f'Column {filter_column} not found'}), 400
        
        # Get unique values and sort them
        unique_values = pivot_system.fact_df[filter_column].unique()
        
        # Handle NaN values and sort
        unique_values = [val for val in unique_values if pd.notna(val)]
        
        # Sort appropriately based on data type
        try:
            if all(isinstance(val, (int, float)) for val in unique_values):
                unique_values.sort()
            else:
                unique_values = sorted([str(val) for val in unique_values])
        except:
            unique_values = sorted([str(val) for val in unique_values])
        
        return jsonify({'filter_values': unique_values})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fact-columns')
def get_fact_columns():
    """Get available columns from fact table for configuration"""
    try:
        return jsonify({
            'columns': list(pivot_system.fact_df.columns),
            'sample_data': pivot_system.fact_df.head(3).to_dict('records'),
            'data_types': {col: str(dtype) for col, dtype in pivot_system.fact_df.dtypes.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Enhanced Dynamic Pivot System Started with Excel-Style Headers!")
    print(f"📊 Dimension table: {pivot_system.dimension_df.shape}")
    print(f"📈 Fact table: {pivot_system.fact_df.shape}")
    print("🌐 Visit http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)