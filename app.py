from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import re
from itertools import product

app = Flask(__name__)

class EnhancedNColumnPivotSystem:
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
    
    def filter_and_pivot(self, filter_column, filter_value, pivot_config, column_filters=None):
        """Filter fact table and create Excel-style hierarchical pivot with N-column support"""
        try:
            # Filter the fact dataframe
            filtered_df = self.fact_df[self.fact_df[filter_column] == filter_value].copy()
            
            if filtered_df.empty:
                return {'error': f'No data found for {filter_column} = {filter_value}'}
            
            print(f"Initial filter: {len(filtered_df)} rows for {filter_column} = {filter_value}")
            
            # Apply column filters if provided
            if column_filters:
                print(f"Applying column filters: {column_filters}")
                for col, included_values in column_filters.items():
                    if col in filtered_df.columns and included_values:
                        initial_count = len(filtered_df)
                        try:
                            filtered_df_col_str = filtered_df[col].astype(str)
                            included_values_str = [str(val) for val in included_values]
                            filtered_df = filtered_df[filtered_df_col_str.isin(included_values_str)]
                        except Exception as type_error:
                            print(f"Type conversion error for {col}: {type_error}, trying direct comparison")
                            filtered_df = filtered_df[filtered_df[col].isin(included_values)]
                        
                        print(f"Filter {col}: included {len(included_values)} values, {initial_count} -> {len(filtered_df)} rows")
                        
                        if filtered_df.empty:
                            available_values = self.fact_df[self.fact_df[filter_column] == filter_value][col].unique()[:10]
                            return {'error': f'No data found after applying filter on {col}. Available values: {list(available_values)}'}
            else:
                print("No column filters provided")
            
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
            
            print(f"N-Column config - Index: {index_cols}, Values: {value_cols}, Columns: {columns_cols} ({len(columns_cols)} levels)")
            
            # Clean the configuration
            valid_columns = set(filtered_df.columns)
            
            # Check if all required columns exist
            missing_cols = []
            for col in index_cols + value_cols + columns_cols:
                if col not in valid_columns:
                    missing_cols.append(col)
            
            if missing_cols:
                return {'error': f'Missing columns in filtered data: {missing_cols}. Available columns: {list(valid_columns)}'}
            
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
            
            print(f"Final N-column config - Index: {index_cols}, Values: {value_cols}, Columns: {columns_cols}")
            print(f"Final filtered dataset: {len(filtered_df)} rows")
            
            # Validation
            if not index_cols:
                return {'error': 'No valid index columns found after filtering and configuration cleanup'}
            if not value_cols:
                numeric_cols = [col for col in valid_columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
                return {'error': f'No valid numeric value columns found. Available numeric columns: {numeric_cols}'}
            if not columns_cols:
                return {'error': 'No valid column dimensions found after filtering and configuration cleanup'}
            
            # Create Excel-style hierarchical pivot with N-column support
            return self._create_enhanced_excel_style_pivot(filtered_df, index_cols, value_cols, columns_cols)
                
        except Exception as e:
            print(f"Pivot error details: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Pivot generation failed: {str(e)}'}
    
    def _create_enhanced_excel_style_pivot(self, df, index_cols, value_cols, columns_cols):
        """Create Excel-style pivot with N-column hierarchical support"""
        try:
            print(f"Creating enhanced pivot with {len(columns_cols)} column dimensions: {columns_cols}")
            
            # Create pivot table for each value column
            pivots = {}
            for value_col in value_cols:
                pivot = pd.pivot_table(
                    df,
                    values=value_col,
                    index=index_cols,
                    columns=columns_cols,  # N-column support
                    aggfunc='sum',
                    fill_value=0,
                    margins=True,
                    margins_name='Grand Total'
                )
                pivots[value_col] = pivot
                print(f"Created pivot for {value_col} with shape: {pivot.shape}")
                print(f"Column structure: {pivot.columns.tolist()[:10]}...")  # Show first 10 columns
            
            # Process the pivot data into Excel-style structure with N-column support
            result = self._process_enhanced_excel_structure(pivots, index_cols, value_cols, columns_cols)
            
            return {
                'pivot_data': result['data'],
                'column_headers': result['headers'],
                'row_hierarchy': result['row_hierarchy'],
                'hierarchy_levels': index_cols,
                'value_columns': value_cols,
                'column_dimensions': columns_cols,  # IMPORTANT: Send column dimensions to frontend
                'excel_style': True,
                'n_column_support': True,
                'config_used': {
                    'index_cols': index_cols,
                    'value_cols': value_cols,
                    'columns_cols': columns_cols
                }
            }
            
        except Exception as e:
            print(f"Enhanced Excel-style pivot error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Enhanced Excel-style pivot generation failed: {str(e)}'}
    
    def _process_enhanced_excel_structure(self, pivots, index_cols, value_cols, columns_cols):
        """Process pivots into Excel-style structure with N-column header support"""
        
        # Get the first pivot to understand structure
        main_pivot = list(pivots.values())[0]
        
        print(f"Processing structure for {len(columns_cols)} column dimensions")
        print(f"Pivot columns sample: {main_pivot.columns.tolist()[:10]}")
        
        # Build enhanced N-column header structure
        headers = self._build_enhanced_excel_headers(main_pivot, value_cols, columns_cols)
        
        # Build row hierarchy with collapsible structure and rollup totals
        row_hierarchy = self._build_row_hierarchy_with_rollup(main_pivot, index_cols, pivots, value_cols, columns_cols)
        
        # Build data structure with parent rollups
        data = self._build_data_with_rollup(main_pivot, pivots, value_cols, columns_cols, row_hierarchy)
        
        return {
            'data': data,
            'headers': headers,
            'row_hierarchy': row_hierarchy
        }
    
    def _build_enhanced_excel_headers(self, pivot, value_cols, columns_cols):
        """Build Excel-style hierarchical column headers for N dimensions"""
        headers = {
            'levels': [],
            'structure': {},
            'data_keys': []
        }
        
        print(f"Building enhanced headers for {len(columns_cols)} column dimensions: {columns_cols}")
        print(f"Pivot columns count: {len(pivot.columns)}")
        
        if len(columns_cols) == 1:
            # Single column dimension
            col_values = [str(c) for c in pivot.columns if str(c) != 'Grand Total']
            col_values = self._sort_values(col_values)
            
            headers['levels'] = columns_cols
            headers['structure'] = {
                'main_groups': col_values,
                'sub_groups': {},
                'has_subtotals': False,
                'column_count': 1
            }
            
            # Build data keys for single level
            for col_val in col_values:
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        headers['data_keys'].append(f"{value_col}_{col_val}")
                else:
                    headers['data_keys'].append(col_val)
                    
        elif len(columns_cols) >= 2:
            # Multi-level column dimensions (2, 3, 4, 5+ levels)
            headers['levels'] = columns_cols
            headers['structure'] = self._build_multi_level_structure(pivot, columns_cols, value_cols)
            headers['data_keys'] = self._build_multi_level_data_keys(headers['structure'], value_cols, pivot)
        
        print(f"Generated {len(headers['data_keys'])} data keys for {len(columns_cols)} column levels")
        print(f"Header structure: {headers['structure']}")
        
        return headers
    
    def _build_multi_level_structure(self, pivot, columns_cols, value_cols):
        """Build multi-level column structure for N dimensions"""
        structure = {
            'main_groups': [],
            'sub_groups': {},
            'has_subtotals': len(columns_cols) > 1,
            'column_count': len(columns_cols),
            'levels': {},
            'hierarchy': {}
        }
        
        print(f"Building multi-level structure for {len(columns_cols)} dimensions")
        
        # Collect all unique values at each level
        level_values = {}
        column_hierarchy = {}
        
        for col in pivot.columns:
            # *** CRITICAL FIX: Skip Grand Total completely in structure building ***
            if col == 'Grand Total':
                print(f"Skipping Grand Total column in structure building")
                continue
                
            if isinstance(col, tuple):
                # Multi-level column tuple
                for level, value in enumerate(col):
                    if level >= len(columns_cols):
                        break
                    # *** ALSO CHECK FOR Grand Total in tuples ***
                    if str(value) == 'Grand Total':
                        print(f"Skipping Grand Total tuple value: {col}")
                        break
                    if level not in level_values:
                        level_values[level] = set()
                    level_values[level].add(str(value))
                    
                # Only process complete, non-Grand Total paths
                path = tuple(str(v) for v in col[:len(columns_cols)])
                if 'Grand Total' not in path and '' not in path:
                    path_str = '|'.join(path)  # Convert tuple to string for JSON serialization
                    if path_str not in column_hierarchy:
                        column_hierarchy[path_str] = []
            else:
                # Single level case - skip if Grand Total
                if str(col) != 'Grand Total':
                    if 0 not in level_values:
                        level_values[0] = set()
                    level_values[0].add(str(col))
                    path_str = str(col)  # Convert to string
                    column_hierarchy[path_str] = []
        
        # Sort values at each level
        for level in range(len(columns_cols)):
            if level in level_values:
                structure['levels'][level] = self._sort_values(list(level_values[level]))
            else:
                structure['levels'][level] = []
        
        # Build main groups (level 0) - should NOT contain Grand Total
        structure['main_groups'] = structure['levels'].get(0, [])
        
        # *** VERIFICATION: Remove Grand Total if it somehow got in ***
        if 'Grand Total' in structure['main_groups']:
            structure['main_groups'].remove('Grand Total')
            print(f"WARNING: Removed Grand Total from main_groups. Clean list: {structure['main_groups']}")
        
        # Build sub-groups hierarchy
        for main_group in structure['main_groups']:
            structure['sub_groups'][main_group] = {}
            
            # Find all combinations that start with this main group
            for path_str in column_hierarchy.keys():
                path = path_str.split('|')  # Convert back from string
                if len(path) > 1 and path[0] == main_group:
                    # Build nested sub-groups
                    current_level = structure['sub_groups'][main_group]
                    for i, level_value in enumerate(path[1:], 1):
                        if i == 1:  # Second level
                            if level_value not in current_level:
                                current_level[level_value] = {} if i < len(path) - 1 else []
                            if i == len(path) - 1:
                                if isinstance(current_level[level_value], list):
                                    current_level[level_value] = []
                        else:  # Third+ levels
                            if isinstance(current_level, dict) and level_value not in current_level:
                                current_level[level_value] = {} if i < len(path) - 1 else []
        
        structure['hierarchy'] = column_hierarchy  # Now uses string keys instead of tuples
        
        print(f"Built structure with levels: {list(structure['levels'].keys())}")
        print(f"Main groups (CLEAN): {structure['main_groups']}")
        print(f"Total hierarchy paths: {len(column_hierarchy)}")
        
        return structure


    def _build_multi_level_data_keys(self, structure, value_cols, pivot):
        """Build data keys for multi-level headers with IMMEDIATELY INTEGRATED subtotals - FRONTEND COMPATIBLE"""
        data_keys = []
        
        print("=" * 60)
        print("DEBUGGING DATA KEY GENERATION - CLEAN GRAND TOTAL")
        print("=" * 60)
        
        print(f"Building multi-level data keys with IMMEDIATELY integrated subtotals")
        print(f"Value columns: {value_cols}")
        print(f"Structure main groups: {structure.get('main_groups', [])}")
        
        # Use the structure's main_groups order to match frontend expectations
        main_groups = structure.get('main_groups', [])
        
        # *** VERIFICATION: Ensure Grand Total is not in main groups ***
        if 'Grand Total' in main_groups:
            main_groups = [g for g in main_groups if g != 'Grand Total']
            print(f"WARNING: Cleaned Grand Total from main_groups. Using: {main_groups}")
        
        # Build a mapping of main groups to their columns
        main_group_columns = {}
        for main_group in main_groups:
            main_group_columns[main_group] = []
        
        # Collect columns for each main group - EXCLUDE Grand Total
        for col in pivot.columns:
            if col == 'Grand Total':
                print(f"Skipping Grand Total column in data key mapping")
                continue
                
            if isinstance(col, tuple):
                main_group = str(col[0])
                # *** ADDITIONAL CHECK: Skip if tuple contains Grand Total ***
                if 'Grand Total' in [str(v) for v in col]:
                    print(f"Skipping Grand Total tuple: {col}")
                    continue
            else:
                main_group = str(col)
                if main_group == 'Grand Total':
                    print(f"Skipping Grand Total string column: {col}")
                    continue
            
            if main_group in main_group_columns:
                main_group_columns[main_group].append(col)
        
        print(f"Main group columns mapping (CLEAN):")
        for group, cols in main_group_columns.items():
            print(f"  {group}: {len(cols)} columns - {cols[:3]}...")
        
        # Process each main group in the SAME ORDER as frontend expects
        for main_group in main_groups:
            print(f"\n--- Processing main group: {main_group} ---")
            group_columns = main_group_columns.get(main_group, [])
            
            if not group_columns:
                print(f"  No columns found for {main_group}")
                continue
            
            # Sort the group's columns to ensure consistent order
            def sort_key(col):
                if isinstance(col, tuple):
                    return col  # Sort tuples naturally
                else:
                    return (col,)  # Convert string to tuple for consistent sorting
            
            group_columns.sort(key=sort_key)
            print(f"  Sorted columns: {group_columns}")
            
            # Add data keys for this main group's regular columns
            for col in group_columns:
                # Create data key from column
                if isinstance(col, tuple):
                    col_key = '_'.join(str(v) for v in col)
                else:
                    col_key = str(col)
                
                # Add value column prefix if multiple value columns
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        final_key = f"{value_col}_{col_key}"
                        data_keys.append(final_key)
                        print(f"    Added regular key: {final_key}")
                else:
                    data_keys.append(col_key)
                    print(f"    Added regular key: {col_key}")
            
            # Add subtotal for groups with multiple columns
            if len(group_columns) > 1:
                subtotal_key = f"{main_group}_Total"
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        final_subtotal_key = f"{value_col}_{subtotal_key}"
                        data_keys.append(final_subtotal_key)
                        print(f"    Added IMMEDIATE SUBTOTAL key: {final_subtotal_key}")
                else:
                    data_keys.append(subtotal_key)
                    print(f"    Added IMMEDIATE SUBTOTAL key: {subtotal_key}")
            else:
                print(f"    No subtotal needed (only {len(group_columns)} column)")
        
        print(f"\n" + "=" * 60)
        print(f"FINAL DATA KEYS ORDER (CLEAN - NO DUPLICATE GRAND TOTAL):")
        for i, key in enumerate(data_keys):
            marker = " ← SUBTOTAL" if "Total" in key else ""
            print(f"  {i+1:2d}. {key}{marker}")
        print("=" * 60)
        
        return data_keys


    def _get_row_totals(self, row_key_tuple, pivots, value_cols, columns_cols):
        """Get totals for a specific row with N-column support and IMMEDIATELY INTEGRATED subtotals"""
        totals = {}
        
        # Use the structure from headers to maintain consistent order
        # We need to rebuild this structure here to match frontend expectations
        structure = self._build_multi_level_structure(pivots[value_cols[0]], columns_cols, value_cols)
        main_groups = structure.get('main_groups', [])
        
        # *** ENSURE Grand Total is not in main groups ***
        if 'Grand Total' in main_groups:
            main_groups = [g for g in main_groups if g != 'Grand Total']
            print(f"Cleaned Grand Total from main_groups in row totals: {main_groups}")
        
        # Process each value column
        for value_col in value_cols:
            pivot = pivots[value_col]
            
            # Group columns by main groups using the SAME ORDER as frontend
            main_group_columns = {}
            for main_group in main_groups:
                main_group_columns[main_group] = []
            
            # Collect columns for each main group - EXCLUDE Grand Total
            for col_key in pivot.columns:
                if col_key == 'Grand Total':
                    continue
                
                if isinstance(col_key, tuple):
                    main_group = str(col_key[0])
                    # Skip if tuple contains Grand Total
                    if 'Grand Total' in [str(v) for v in col_key]:
                        continue
                else:
                    main_group = str(col_key)
                    if main_group == 'Grand Total':
                        continue
                
                if main_group in main_group_columns:
                    main_group_columns[main_group].append(col_key)
            
            # Sort columns within each group for consistency
            for main_group in main_group_columns:
                def sort_key(col):
                    if isinstance(col, tuple):
                        return col
                    else:
                        return (col,)
                main_group_columns[main_group].sort(key=sort_key)
            
            # Process each main group and add subtotals IMMEDIATELY
            for main_group in main_groups:
                group_columns = main_group_columns.get(main_group, [])
                if not group_columns:
                    continue
                    
                group_total = 0
                
                # Add individual column data
                for col_key in group_columns:
                    if isinstance(col_key, tuple):
                        col_key_str = '_'.join(str(v) for v in col_key)
                    else:
                        col_key_str = str(col_key)
                    
                    if len(value_cols) > 1:
                        data_key = f"{value_col}_{col_key_str}"
                    else:
                        data_key = col_key_str
                    
                    try:
                        value = pivot.loc[row_key_tuple, col_key]
                        converted_value = self._safe_convert_value(value)
                        totals[data_key] = converted_value
                        group_total += converted_value
                    except (KeyError, IndexError):
                        totals[data_key] = 0
                
                # Add subtotal for groups with multiple columns
                if len(group_columns) > 1:
                    subtotal_key = f"{main_group}_Total"
                    if len(value_cols) > 1:
                        subtotal_data_key = f"{value_col}_{subtotal_key}"
                    else:
                        subtotal_data_key = subtotal_key
                    
                    totals[subtotal_data_key] = group_total
                    print(f"Added IMMEDIATE subtotal {subtotal_data_key} = {group_total} for group {main_group}")
        
        # Add grand total for this row (SEPARATELY from main groups)
        try:
            main_pivot = pivots[value_cols[0]]
            if 'Grand Total' in main_pivot.columns:
                if len(value_cols) == 1:
                    value = main_pivot.loc[row_key_tuple, 'Grand Total']
                    totals["Grand_Total"] = self._safe_convert_value(value)
                else:
                    grand_total = 0
                    for value_col in value_cols:
                        try:
                            pivot = pivots[value_col]
                            if 'Grand Total' in pivot.columns:
                                value = pivot.loc[row_key_tuple, 'Grand Total']
                                grand_total += self._safe_convert_value(value)
                        except (KeyError, IndexError):
                            continue
                    totals["Grand_Total"] = grand_total
        except (KeyError, IndexError):
            totals["Grand_Total"] = 0
        
        return totals

    def _build_multi_level_data_keys(self, structure, value_cols, pivot):
        """Build data keys for multi-level headers with IMMEDIATELY INTEGRATED subtotals - FRONTEND COMPATIBLE"""
        data_keys = []
        
        print("=" * 60)
        print("DEBUGGING DATA KEY GENERATION - INTEGRATED SUBTOTALS")
        print("=" * 60)
        
        print(f"Building multi-level data keys with IMMEDIATELY integrated subtotals")
        print(f"Value columns: {value_cols}")
        print(f"Structure main groups: {structure.get('main_groups', [])}")
        
        # Use the structure's main_groups order to match frontend expectations
        main_groups = structure.get('main_groups', [])
        
        # Build a mapping of main groups to their columns
        main_group_columns = {}
        for main_group in main_groups:
            if main_group == 'Grand Total':
                continue
            main_group_columns[main_group] = []
        
        # Collect columns for each main group
        for col in pivot.columns:
            if col == 'Grand Total':
                continue
                
            if isinstance(col, tuple):
                main_group = str(col[0])
            else:
                main_group = str(col)
            
            if main_group in main_group_columns:
                main_group_columns[main_group].append(col)
        
        print(f"Main group columns mapping:")
        for group, cols in main_group_columns.items():
            print(f"  {group}: {len(cols)} columns - {cols[:3]}...")
        
        # Process each main group in the SAME ORDER as frontend expects
        for main_group in main_groups:
            if main_group == 'Grand Total':
                continue
                
            print(f"\n--- Processing main group: {main_group} ---")
            group_columns = main_group_columns.get(main_group, [])
            
            if not group_columns:
                print(f"  No columns found for {main_group}")
                continue
            
            # Sort the group's columns to ensure consistent order
            def sort_key(col):
                if isinstance(col, tuple):
                    return col  # Sort tuples naturally
                else:
                    return (col,)  # Convert string to tuple for consistent sorting
            
            group_columns.sort(key=sort_key)
            print(f"  Sorted columns: {group_columns}")
            
            # Add data keys for this main group's regular columns
            for col in group_columns:
                # Create data key from column
                if isinstance(col, tuple):
                    col_key = '_'.join(str(v) for v in col)
                else:
                    col_key = str(col)
                
                # Add value column prefix if multiple value columns
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        final_key = f"{value_col}_{col_key}"
                        data_keys.append(final_key)
                        print(f"    Added regular key: {final_key}")
                else:
                    data_keys.append(col_key)
                    print(f"    Added regular key: {col_key}")
            
            # *** CRITICAL FIX: Add subtotal IMMEDIATELY after this group's columns ***
            if len(group_columns) > 1:
                subtotal_key = f"{main_group}_Total"
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        final_subtotal_key = f"{value_col}_{subtotal_key}"
                        data_keys.append(final_subtotal_key)
                        print(f"    Added IMMEDIATE SUBTOTAL key: {final_subtotal_key}")
                else:
                    data_keys.append(subtotal_key)
                    print(f"    Added IMMEDIATE SUBTOTAL key: {subtotal_key}")
            else:
                print(f"    No subtotal needed (only {len(group_columns)} column)")
        
        print(f"\n" + "=" * 60)
        print(f"FINAL DATA KEYS ORDER (INTEGRATED SUBTOTALS):")
        for i, key in enumerate(data_keys):
            marker = " ← SUBTOTAL" if "Total" in key else ""
            print(f"  {i+1:2d}. {key}{marker}")
        print("=" * 60)
        
        return data_keys
    
    def _generate_subtotal_keys(self, structure, value_cols):
        """Generate subtotal keys for intermediate column levels"""
        subtotal_keys = []
        
        # Generate subtotals for each main group if there are sub-groups
        for main_group in structure['main_groups']:
            sub_groups = structure['sub_groups'].get(main_group, {})
            if sub_groups and len(sub_groups) > 1:
                subtotal_key = f"{main_group}_Total"
                if len(value_cols) > 1:
                    for value_col in value_cols:
                        subtotal_keys.append(f"{value_col}_{subtotal_key}")
                else:
                    subtotal_keys.append(subtotal_key)
        
        return subtotal_keys
    
    def _build_row_hierarchy_with_rollup(self, pivot, index_cols, pivots, value_cols, columns_cols):
        """Build hierarchical row structure with parent totals calculated"""
        hierarchy = {}
        
        # First, build the basic hierarchy, excluding Grand Total
        for row_key in pivot.index:
            if row_key == 'Grand Total':
                continue  # Skip Grand Total from hierarchy
            
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
                        'path': levels[:i+1],
                        'totals': {}  # Store parent totals
                    }
                
                if i < len(levels) - 1:
                    current = current[level]['children']
        
        # Calculate parent rollup totals
        self._calculate_parent_rollups(hierarchy, pivots, value_cols, columns_cols)
        
        return hierarchy
    
    def _calculate_parent_rollups(self, hierarchy, pivots, value_cols, columns_cols):
        """Calculate rollup totals for parent nodes"""
        
        def rollup_node(node):
            if not node['children']:
                # Leaf node - get actual data
                row_key_tuple = tuple(node['path'])
                node['totals'] = self._get_row_totals(row_key_tuple, pivots, value_cols, columns_cols)
                return node['totals']
            else:
                # Parent node - sum children
                node['totals'] = {}
                for child_name, child_node in node['children'].items():
                    child_totals = rollup_node(child_node)
                    
                    # Add child totals to parent
                    for key, value in child_totals.items():
                        if key not in node['totals']:
                            node['totals'][key] = 0
                        node['totals'][key] += value
                
                return node['totals']
        
        # Process all top-level nodes
        for name, node in hierarchy.items():
            rollup_node(node)
    
    def _get_row_totals(self, row_key_tuple, pivots, value_cols, columns_cols):
        """Get totals for a specific row with N-column support and IMMEDIATELY INTEGRATED subtotals"""
        totals = {}
        
        # Use the structure from headers to maintain consistent order
        # We need to rebuild this structure here to match frontend expectations
        structure = self._build_multi_level_structure(pivots[value_cols[0]], columns_cols, value_cols)
        main_groups = structure.get('main_groups', [])
        
        # Process each value column
        for value_col in value_cols:
            pivot = pivots[value_col]
            
            # Group columns by main groups using the SAME ORDER as frontend
            main_group_columns = {}
            for main_group in main_groups:
                if main_group == 'Grand Total':
                    continue
                main_group_columns[main_group] = []
            
            # Collect columns for each main group
            for col_key in pivot.columns:
                if col_key == 'Grand Total':
                    continue
                
                if isinstance(col_key, tuple):
                    main_group = str(col_key[0])
                else:
                    main_group = str(col_key)
                
                if main_group in main_group_columns:
                    main_group_columns[main_group].append(col_key)
            
            # Sort columns within each group for consistency
            for main_group in main_group_columns:
                def sort_key(col):
                    if isinstance(col, tuple):
                        return col
                    else:
                        return (col,)
                main_group_columns[main_group].sort(key=sort_key)
            
            # *** CRITICAL FIX: Process each main group and add subtotals IMMEDIATELY ***
            for main_group in main_groups:
                if main_group == 'Grand Total':
                    continue
                    
                group_columns = main_group_columns.get(main_group, [])
                if not group_columns:
                    continue
                    
                group_total = 0
                
                # Add individual column data
                for col_key in group_columns:
                    if isinstance(col_key, tuple):
                        col_key_str = '_'.join(str(v) for v in col_key)
                    else:
                        col_key_str = str(col_key)
                    
                    if len(value_cols) > 1:
                        data_key = f"{value_col}_{col_key_str}"
                    else:
                        data_key = col_key_str
                    
                    try:
                        value = pivot.loc[row_key_tuple, col_key]
                        converted_value = self._safe_convert_value(value)
                        totals[data_key] = converted_value
                        group_total += converted_value
                    except (KeyError, IndexError):
                        totals[data_key] = 0
                
                # *** IMMEDIATELY ADD SUBTOTAL for this main group after its columns ***
                if len(group_columns) > 1:
                    subtotal_key = f"{main_group}_Total"
                    if len(value_cols) > 1:
                        subtotal_data_key = f"{value_col}_{subtotal_key}"
                    else:
                        subtotal_data_key = subtotal_key
                    
                    totals[subtotal_data_key] = group_total
                    print(f"Added IMMEDIATE subtotal {subtotal_data_key} = {group_total} for group {main_group}")
        
        # Add grand total for this row
        try:
            main_pivot = pivots[value_cols[0]]
            if 'Grand Total' in main_pivot.columns:
                if len(value_cols) == 1:
                    value = main_pivot.loc[row_key_tuple, 'Grand Total']
                    totals["Grand_Total"] = self._safe_convert_value(value)
                else:
                    grand_total = 0
                    for value_col in value_cols:
                        try:
                            pivot = pivots[value_col]
                            if 'Grand Total' in pivot.columns:
                                value = pivot.loc[row_key_tuple, 'Grand Total']
                                grand_total += self._safe_convert_value(value)
                        except (KeyError, IndexError):
                            continue
                    totals["Grand_Total"] = grand_total
        except (KeyError, IndexError):
            totals["Grand_Total"] = 0
        
        return totals
    
    def _calculate_row_subtotals(self, pivot, row_key, value_col, include_value_prefix):
        """Calculate subtotals for main column groups in N-column scenario"""
        subtotals = {}
        
        # Group columns by main category (first level)
        main_groups = {}
        for col in pivot.columns:
            if col == 'Grand Total':
                continue
            
            if isinstance(col, tuple):
                main_col = str(col[0])
                if main_col not in main_groups:
                    main_groups[main_col] = []
                main_groups[main_col].append(col)
            else:
                main_col = str(col)
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
    
    def _build_data_with_rollup(self, main_pivot, pivots, value_cols, columns_cols, row_hierarchy):
        """Build data structure using the rollup totals from hierarchy with IMMEDIATELY integrated subtotals"""
        data = {}
        
        def extract_data_from_hierarchy(hierarchy, path_prefix=""):
            for name, node in hierarchy.items():
                # Create row key string for this node
                if path_prefix:
                    full_path = path_prefix + ", '" + name + "'"
                else:
                    full_path = "'" + name + "'"
                
                row_key_str = f"({full_path})"
                
                # Store the rollup totals for this node
                data[row_key_str] = node['totals'].copy()
                
                # Process children
                if node['children']:
                    extract_data_from_hierarchy(node['children'], full_path)
        
        extract_data_from_hierarchy(row_hierarchy)
        
        # *** CRITICAL FIX: Add Grand Total row with IMMEDIATELY integrated subtotals ***
        if 'Grand Total' in main_pivot.index:
            grand_total_row = {}
            
            # Get the structure to maintain the same order as data keys
            structure = self._build_multi_level_structure(main_pivot, columns_cols, value_cols)
            main_groups = structure.get('main_groups', [])
            
            # Group columns by main groups for IMMEDIATE subtotal calculation
            main_group_columns = {}
            
            for value_col in value_cols:
                pivot = pivots[value_col]
                
                # First pass: collect columns by main group
                for col_key in pivot.columns:
                    if col_key == 'Grand Total':
                        continue
                    
                    if isinstance(col_key, tuple):
                        main_group = str(col_key[0])
                        if main_group not in main_group_columns:
                            main_group_columns[main_group] = []
                        main_group_columns[main_group].append(col_key)
                    else:
                        main_group = str(col_key)
                        if main_group not in main_group_columns:
                            main_group_columns[main_group] = []
                        main_group_columns[main_group].append(col_key)
                
                # Sort columns within each group
                for main_group in main_group_columns:
                    def sort_key(col):
                        if isinstance(col, tuple):
                            return col
                        else:
                            return (col,)
                    main_group_columns[main_group].sort(key=sort_key)
                
                # *** CRITICAL FIX: Process each main group and add its subtotal IMMEDIATELY ***
                for main_group in sorted(main_group_columns.keys()):
                    group_columns = main_group_columns[main_group]
                    group_total = 0
                    
                    # Add individual column data
                    for col_key in group_columns:
                        if isinstance(col_key, tuple):
                            col_key_str = '_'.join(str(v) for v in col_key)
                        else:
                            col_key_str = str(col_key)
                        
                        if len(value_cols) > 1:
                            data_key = f"{value_col}_{col_key_str}"
                        else:
                            data_key = col_key_str
                        
                        try:
                            value = pivot.loc['Grand Total', col_key]
                            converted_value = self._safe_convert_value(value)
                            grand_total_row[data_key] = converted_value
                            group_total += converted_value
                        except (KeyError, IndexError):
                            grand_total_row[data_key] = 0
                    
                    # *** IMMEDIATELY add subtotal for this main group after its columns ***
                    if len(group_columns) > 1:
                        subtotal_key = f"{main_group}_Total"
                        if len(value_cols) > 1:
                            subtotal_data_key = f"{value_col}_{subtotal_key}"
                        else:
                            subtotal_data_key = subtotal_key
                        
                        grand_total_row[subtotal_data_key] = group_total
                        print(f"Added IMMEDIATE Grand Total subtotal {subtotal_data_key} = {group_total}")
            
            # Add overall grand total
            try:
                if 'Grand Total' in main_pivot.columns:
                    grand_total_key = "Grand_Total"
                    if len(value_cols) == 1:
                        value = main_pivot.loc['Grand Total', 'Grand Total']
                        grand_total_row[grand_total_key] = self._safe_convert_value(value)
                    else:
                        total_value = 0
                        for value_col in value_cols:
                            pivot = pivots[value_col]
                            try:
                                value = pivot.loc['Grand Total', 'Grand Total']
                                total_value += self._safe_convert_value(value)
                            except (KeyError, IndexError):
                                continue
                        grand_total_row[grand_total_key] = total_value
            except (KeyError, IndexError):
                grand_total_row["Grand_Total"] = 0
            
            data['Grand_Total'] = grand_total_row
        
        return data
    
    def _sort_values(self, values):
        """Sort values with intelligent numeric/string sorting"""
        try:
            # Try to convert to numbers for sorting
            numeric_values = []
            string_values = []
            
            for val in values:
                try:
                    numeric_values.append((float(val), val))
                except (ValueError, TypeError):
                    string_values.append(val)
            
            # Sort numeric values by their numeric representation
            numeric_values.sort(key=lambda x: x[0])
            numeric_sorted = [x[1] for x in numeric_values]
            
            # Sort string values alphabetically
            string_values.sort()
            
            # Return numeric values first, then string values
            return numeric_sorted + string_values
            
        except Exception as e:
            print(f"Error sorting values: {e}")
            # Fallback to simple string sort
            return sorted([str(v) for v in values])
    
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

    def get_column_unique_values(self, column_name):
        """Get unique values for a specific column for filtering"""
        try:
            if column_name not in self.fact_df.columns:
                return {'error': f'Column {column_name} not found'}
            
            # Get unique values and sort them
            unique_values = self.fact_df[column_name].dropna().unique()
            
            # Sort appropriately based on data type
            try:
                if pd.api.types.is_numeric_dtype(self.fact_df[column_name]):
                    unique_values = sorted(unique_values)
                else:
                    unique_values = sorted([str(val) for val in unique_values])
            except:
                unique_values = sorted([str(val) for val in unique_values])
            
            return {
                'column': column_name,
                'values': unique_values,
                'count': len(unique_values),
                'data_type': str(self.fact_df[column_name].dtype)
            }
            
        except Exception as e:
            return {'error': f'Error getting values for {column_name}: {str(e)}'}

# Initialize the enhanced system
pivot_system = EnhancedNColumnPivotSystem()

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
    """Get filtered and pivoted data with enhanced N-column hierarchical support"""
    try:
        data = request.get_json()
        print(f"Received N-column pivot request: {data}")
        
        filter_column = data.get('filter_column')
        filter_value = data.get('filter_value')
        pivot_config = data.get('pivot_config', {})
        column_filters = data.get('column_filters', {})
        
        if not filter_column or not filter_value:
            return jsonify({'error': 'filter_column and filter_value are required'}), 400
        
        # Convert single column configs to list format for backward compatibility
        if 'index_cols' not in pivot_config and 'index_col' in pivot_config:
            pivot_config['index_cols'] = [pivot_config['index_col']]
        if 'value_cols' not in pivot_config and 'value_col' in pivot_config:
            pivot_config['value_cols'] = [pivot_config['value_col']]
        if 'columns_cols' not in pivot_config and 'columns_col' in pivot_config:
            pivot_config['columns_cols'] = [pivot_config['columns_col']]
        
        columns_count = len(pivot_config.get('columns_cols', []))
        print(f"Processing N-column pivot with {columns_count} column dimensions")
        print(f"Filtering by {filter_column} = {filter_value}")
        print(f"Pivot config: {pivot_config}")
        print(f"Column filters: {column_filters}")
        
        result = pivot_system.filter_and_pivot(filter_column, filter_value, pivot_config, column_filters)
        
        if 'error' in result:
            print(f"N-column pivot error: {result['error']}")
            return jsonify(result), 400
        
        print(f"Enhanced N-column pivot successful with {len(result.get('pivot_data', {}))} data rows and {columns_count} column levels.")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Server error in N-column pivot generation: {str(e)}'
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

@app.route('/api/dynamic-filter-values', methods=['POST'])
def get_dynamic_filter_values():
    """Get dynamic filter values based on current filter state"""
    try:
        data = request.get_json()
        filter_column = data.get('filter_column')
        columns = data.get('columns', [])
        current_filters = data.get('current_filters', {})
        pivot_config = data.get('pivot_config', {})
        
        if not filter_column or not columns:
            return jsonify({'error': 'filter_column and columns are required'}), 400
        
        print(f"Getting dynamic filter values for columns: {columns}")
        print(f"Current filters: {current_filters}")
        
        # Start with the fact dataframe
        df = pivot_system.fact_df.copy()
        
        # Apply current filters to get the filtered dataset
        for col, included_values in current_filters.items():
            if col in df.columns and included_values:
                initial_count = len(df)
                df = df[df[col].isin(included_values)]
                print(f"Applied filter {col}: {initial_count} -> {len(df)} rows")
        
        result = {'column_values': {}}
        
        # For each column, get unique values from the filtered dataset
        for column in columns:
            if column in df.columns:
                unique_values = df[column].dropna().unique()
                
                # Sort values appropriately
                try:
                    # Try numeric sort first
                    if all(isinstance(val, (int, float)) for val in unique_values if pd.notna(val)):
                        unique_values = sorted(unique_values)
                    else:
                        # String sort
                        unique_values = sorted([str(val) for val in unique_values if pd.notna(val)])
                except:
                    # Fallback to string sort
                    unique_values = sorted([str(val) for val in unique_values if pd.notna(val)])
                
                result['column_values'][column] = unique_values
                print(f"Column {column}: {len(unique_values)} unique values")
            else:
                result['column_values'][column] = []
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Error getting dynamic filter values: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

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

@app.route('/api/column-values/<column_name>')
def get_column_values(column_name):
    """Get unique values for a specific column - enhanced endpoint"""
    try:
        result = pivot_system.get_column_unique_values(column_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-pivot', methods=['POST'])
def debug_pivot():
    """Debug endpoint to check what's happening with N-column pivot generation"""
    try:
        data = request.get_json()
        filter_column = data.get('filter_column')
        filter_value = data.get('filter_value')
        column_filters = data.get('column_filters', {})
        pivot_config = data.get('pivot_config', {})
        
        # Start with the fact dataframe
        df = pivot_system.fact_df.copy()
        
        columns_cols = pivot_config.get('columns_cols', [])
        
        debug_info = {
            'original_shape': df.shape,
            'original_columns': list(df.columns),
            'filter_column': filter_column,
            'filter_value': filter_value,
            'column_filters': column_filters,
            'pivot_config': pivot_config,
            'n_column_dimensions': len(columns_cols),
            'column_dimensions': columns_cols
        }
        
        # Apply main filter
        df = df[df[filter_column] == filter_value].copy()
        debug_info['after_main_filter'] = df.shape
        
        # Apply column filters
        if column_filters:
            for col, included_values in column_filters.items():
                if col in df.columns and included_values:
                    initial_count = len(df)
                    
                    # Get sample of actual values in the column
                    actual_values = df[col].unique()[:10]
                    debug_info[f'{col}_actual_values'] = list(actual_values)
                    debug_info[f'{col}_included_values'] = included_values[:10]
                    debug_info[f'{col}_data_type'] = str(df[col].dtype)
                    
                    # Try filtering
                    try:
                        df_filtered = df[df[col].isin(included_values)]
                        debug_info[f'{col}_filter_result'] = f"{initial_count} -> {len(df_filtered)} rows"
                        df = df_filtered
                    except Exception as e:
                        debug_info[f'{col}_filter_error'] = str(e)
        
        debug_info['final_shape'] = df.shape
        debug_info['final_columns'] = list(df.columns)
        
        # Test pivot creation
        try:
            if columns_cols and len(columns_cols) > 0:
                test_pivot = pd.pivot_table(
                    df,
                    values=pivot_config.get('value_cols', ['Sales'])[0],
                    index=pivot_config.get('index_cols', ['Region'])[0],
                    columns=columns_cols,
                    aggfunc='sum',
                    fill_value=0
                )
                debug_info['test_pivot_shape'] = test_pivot.shape
                debug_info['test_pivot_columns_sample'] = str(test_pivot.columns.tolist()[:10])
                debug_info['test_pivot_success'] = True
        except Exception as e:
            debug_info['test_pivot_error'] = str(e)
            debug_info['test_pivot_success'] = False
        
        return jsonify(debug_info)
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("🚀 Enhanced N-Column Dynamic Pivot System Started!")
    print(f"📊 Dimension table: {pivot_system.dimension_df.shape}")
    print(f"📈 Fact table: {pivot_system.fact_df.shape}")
    print("🌐 Visit http://localhost:5000")
    print("✨ NEW N-Column Features:")
    print("   - Unlimited column dimensions (1, 2, 3, 4, 5+ levels)")
    print("   - Enhanced multi-level header generation")
    print("   - Intelligent column hierarchy detection")
    print("   - Dynamic data key building")
    print("   - Smart subtotal calculations")
    print("   - Robust N-column pivot processing")
    print("🔧 FIXED: Integrated subtotals positioned immediately after each group")
    app.run(debug=True, host='0.0.0.0', port=5000)