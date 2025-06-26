# Dynamic Tree Table Flask App

A Flask web application that generates interactive hierarchical tree tables from pandas DataFrames using pivot tables. No hardcoded data - works with any CSV file!

## Features

- 📊 **Dynamic Data Processing**: Upload any CSV file and generate pivot tables
- 🌳 **Interactive Tree Structure**: Expandable/collapsible hierarchical view
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- 📈 **Statistics Dashboard**: Key metrics and insights
- 🔄 **Real-time Processing**: Instant table generation from uploaded data

## Quick Start

### 1. Clone or Download Files

Create a new directory and save these files:
- `app.py` - Main Flask application
- `templates/index.html` - HTML template
- `requirements.txt` - Python dependencies
- `sample_sales_data.csv` - Sample CSV for testing

### 2. Setup Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Templates Directory

```bash
mkdir templates
```

Move the `index.html` file to the `templates/` directory.

### 5. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## How to Use

### Option 1: Use Sample Data
1. Click "Load Sample Data" button
2. The app will generate a tree table with built-in sample sales data

### Option 2: Upload Your Own CSV
1. Prepare your CSV file with hierarchical data
2. Fill in the form fields:
   - **Index Columns**: Comma-separated list of columns for hierarchy (e.g., `Region,Category,Product`)
   - **Value Column**: Column containing the numeric values (e.g., `Sales`)
   - **Columns**: Column to pivot on (e.g., `Month`)
3. Upload your CSV file
4. Click "Upload & Generate Table"

## CSV Data Format

Your CSV should have columns for:
- **Hierarchy levels** (e.g., Region, Category, Subcategory, Product)
- **Time periods** or **categories to pivot on** (e.g., Month, Quarter)
- **Numeric values** (e.g., Sales, Revenue, Quantity)

Example structure:
```csv
Region,Category,Product,Month,Sales
East,Electronics,Laptop,Jan,50000
East,Electronics,Laptop,Feb,55000
West,Furniture,Chair,Jan,12000
...
```

## Configuration Examples

### Sales Data by Region/Product/Month
- Index Columns: `Region,Category,Product`
- Value Column: `Sales`
- Columns: `Month`

### Financial Data by Department/Quarter
- Index Columns: `Department,Team,Employee`
- Value Column: `Revenue`
- Columns: `Quarter`

### Inventory by Location/Category/Week
- Index Columns: `Warehouse,Category,Item`
- Value Column: `Quantity`
- Columns: `Week`

## Features Explained

### Interactive Tree Structure
- Click ▶ to expand categories
- Click ▼ to collapse categories
- Hover effects for better UX
- Different styling for each hierarchy level

### Statistics Dashboard
- **Total Sales**: Sum of all values
- **Best Period**: Time period with highest sales
- **Top Category**: Category with highest total
- **Active Items**: Count of items with sales data

### Responsive Design
- Adapts to different screen sizes
- Mobile-friendly interface
- Scrollable tables on small screens

## Technical Details

### Backend (Flask)
- Uses pandas `pivot_table()` for data processing
- Generates JSON structure for frontend consumption
- Handles file uploads and validation
- RESTful API endpoints

### Frontend (JavaScript)
- Dynamic table rendering
- Tree expansion/collapse logic
- Statistics calculation
- Modern CSS with animations

### Data Processing
1. CSV → pandas DataFrame
2. DataFrame → pivot_table()
3. pivot_table → JSON
4. JSON → Interactive Tree Table

## API Endpoints

- `GET /` - Main page
- `GET /api/data` - Get sample data
- `POST /api/upload` - Upload CSV and generate pivot
- `GET /api/sample` - Get sample data info

## Customization

### Styling
Modify the CSS in `templates/index.html` to change:
- Colors and gradients
- Typography
- Layout and spacing
- Animation effects

### Data Processing
Modify `TreeTableGenerator` class in `app.py` to:
- Change sample data structure
- Add custom aggregation functions
- Implement data validation rules
- Add new statistics calculations

## Troubleshooting

### Common Issues

1. **File Upload Fails**
   - Check CSV format and encoding
   - Ensure column names match form inputs
   - Verify file size limits

2. **Table Doesn't Display**
   - Check browser console for JavaScript errors
   - Verify API responses in Network tab
   - Ensure all required columns exist

3. **Missing Data**
   - Check for null values in pivot columns
   - Verify data types (numeric values for aggregation)
   - Review pivot table parameters

### Error Messages
- "Missing columns": Column names don't match CSV headers
- "Upload failed": File format or processing error
- "No file selected": Choose a CSV file before uploading

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Enjoy creating beautiful, interactive data tables! 📊✨**