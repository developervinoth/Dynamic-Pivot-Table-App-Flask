from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json

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
        """Filter fact table and create pivot"""
        try:
            # Filter the fact dataframe
            filtered_df = self.fact_df[self.fact_df[filter_column] == filter_value].copy()
            
            if filtered_df.empty:
                return {'error': f'No data found for {filter_column} = {filter_value}'}
            
            # Extract pivot configuration
            index_cols = pivot_config.get('index_cols', [])
            value_col = pivot_config.get('value_col', 'Sales')
            columns_col = pivot_config.get('columns_col', 'Month')
            
            print(f"Original config - Index: {index_cols}, Value: {value_col}, Columns: {columns_col}")
            
            # Clean the configuration - remove filter column and duplicates
            index_cols = [col for col in index_cols 
                         if col != filter_column and col != columns_col and col != value_col and col in filtered_df.columns]
            
            # Remove duplicates while preserving order
            seen = set()
            index_cols = [col for col in index_cols if not (col in seen or seen.add(col))]
            
            print(f"Cleaned index_cols: {index_cols}")
            
            if not index_cols:
                # If no valid index columns, use the first suitable column
                available_cols = [col for col in filtered_df.columns 
                                if col not in [filter_column, value_col, columns_col]]
                if available_cols:
                    index_cols = [available_cols[0]]
                else:
                    return {'error': 'No suitable columns found for pivot index'}
            
            if value_col not in filtered_df.columns:
                return {'error': f'Value column "{value_col}" not found'}
            
            if columns_col not in filtered_df.columns:
                return {'error': f'Columns column "{columns_col}" not found'}
            
            # Ensure columns_col is not in index_cols
            if columns_col in index_cols:
                index_cols.remove(columns_col)
                print(f"Removed columns_col from index_cols. New index_cols: {index_cols}")
            
            if not index_cols:
                return {'error': 'No valid index columns remaining after cleanup'}
            
            print(f"Final config - Index: {index_cols}, Value: {value_col}, Columns: {columns_col}")
            
            # Aggregate data first to handle duplicates
            groupby_cols = index_cols + [columns_col]
            print(f"Grouping by: {groupby_cols}")
            
            agg_df = filtered_df.groupby(groupby_cols)[value_col].sum().reset_index()
            print(f"Aggregated data shape: {agg_df.shape}")
            
            # Create pivot table
            pivot = pd.pivot_table(
                agg_df,
                values=value_col,
                index=index_cols,
                columns=columns_col,
                aggfunc='sum',
                fill_value=0,
                margins=True,
                margins_name='Grand Total'
            )
            
            print(f"Pivot shape: {pivot.shape}")
            print(f"Pivot columns: {list(pivot.columns)}")
            print(f"Pivot index: {list(pivot.index)}")
            
            # Convert to JSON format
            result = {}
            for col in pivot.columns:
                result[str(col)] = {}
                for idx in pivot.index:
                    if isinstance(idx, tuple):
                        key = str(idx)
                    else:
                        # Handle single index - pad with empty strings to match expected structure
                        key = f"('{idx}'" + ", ''" * (len(index_cols) - 1) + ")"
                    
                    value = pivot.loc[idx, col]
                    # Convert numpy types to Python types and handle zeros
                    if pd.isna(value) or value == 0:
                        result[str(col)][key] = None
                    else:
                        result[str(col)][key] = int(value)
            
            return {
                'pivot_data': result,
                'hierarchy_levels': index_cols,
                'value_column': value_col,
                'columns': [str(col) for col in pivot.columns if str(col) != 'Grand Total'] + ['Grand Total'],
                'filter_info': f'{filter_column}: {filter_value}',
                'total_records': len(filtered_df),
                'config_used': {
                    'index_cols': index_cols,
                    'value_col': value_col,
                    'columns_col': columns_col
                }
            }
            
        except Exception as e:
            print(f"Pivot error details: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Pivot generation failed: {str(e)}'}

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
    """Get filtered and pivoted data"""
    try:
        data = request.get_json()
        print(f"Received pivot request: {data}")  # Debug log
        
        filter_column = data.get('filter_column')
        filter_value = data.get('filter_value')
        pivot_config = data.get('pivot_config', {})
        
        if not filter_column or not filter_value:
            return jsonify({'error': 'filter_column and filter_value are required'}), 400
        
        print(f"Filtering by {filter_column} = {filter_value}")  # Debug log
        print(f"Pivot config: {pivot_config}")  # Debug log
        
        result = pivot_system.filter_and_pivot(filter_column, filter_value, pivot_config)
        
        if 'error' in result:
            print(f"Pivot error: {result['error']}")  # Debug log
            return jsonify(result), 400
        
        print(f"Pivot successful. Columns: {result.get('columns', [])}")  # Debug log
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Server error in pivot generation: {str(e)}'
        print(error_msg)  # Debug log
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
    print("🚀 Dynamic Pivot System Started!")
    print(f"📊 Dimension table: {pivot_system.dimension_df.shape}")
    print(f"📈 Fact table: {pivot_system.fact_df.shape}")
    print("🌐 Visit http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)