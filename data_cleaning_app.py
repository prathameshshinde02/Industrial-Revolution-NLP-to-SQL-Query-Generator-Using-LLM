# from dotenv import load_dotenv
# load_dotenv()

# import streamlit as st
# import os
# import sqlite3
# import google.generativeai as genai
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
# import io
# import numpy as np
# from datetime import datetime

# # Configure API key
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# # Current time and user information
# CURRENT_TIME = "2025-02-13 05:04:49"
# CURRENT_USER = "piyushk1304"

# # Fixed colors for visualization
# PRIMARY_COLOR = "#1f77b4"  # Blue
# SECONDARY_COLOR = "#ff7f0e"  # Orange

# def clean_data(df):
#     """Perform comprehensive data cleaning."""
#     stats = {}
    
#     # Store original shape
#     stats['original_rows'] = len(df)
#     stats['original_columns'] = len(df.columns)
    
#     # Check for duplicates
#     stats['duplicates_removed'] = df.duplicated().sum()
#     df = df.drop_duplicates()
    
#     # Handle missing values
#     stats['total_missing_values'] = df.isnull().sum().sum()
#     stats['missing_values_by_column'] = df.isnull().sum().to_dict()
    
#     # Process numeric columns
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     stats['numeric_columns_cleaned'] = []
    
#     for col in numeric_columns:
#         # Store original stats
#         stats[f'{col}_before_mean'] = df[col].mean()
#         stats[f'{col}_before_std'] = df[col].std()
        
#         # Handle outliers
#         mean = df[col].mean()
#         std = df[col].std()
#         df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
        
#         # Store cleaned stats
#         stats[f'{col}_after_mean'] = df[col].mean()
#         stats[f'{col}_after_std'] = df[col].std()
#         stats['numeric_columns_cleaned'].append(col)
    
#     # Clean string columns
#     string_columns = df.select_dtypes(include=['object']).columns
#     for col in string_columns:
#         df[col] = df[col].astype(str).str.strip()
#         df[col] = df[col].replace('', np.nan)
#         df[col] = df[col].replace('null', np.nan)
#         df[col] = df[col].replace('NULL', np.nan)
#         df[col] = df[col].replace('None', np.nan)
    
#     # Final cleaning
#     df = df.dropna()
    
#     # Store final shape
#     stats['final_rows'] = len(df)
#     stats['final_columns'] = len(df.columns)
    
#     return df, stats

# def display_cleaning_summary(stats):
#     """Display data cleaning statistics."""
#     st.subheader("Data Cleaning Summary")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric(
#             label="Rows Processed",
#             value=stats['final_rows'],
#             delta=f"{stats['final_rows'] - stats['original_rows']} rows removed"
#         )
    
#     with col2:
#         st.metric(
#             label="Duplicates Removed",
#             value=stats['duplicates_removed']
#         )
    
#     with col3:
#         st.metric(
#             label="Missing Values Handled",
#             value=stats['total_missing_values']
#         )

#     if stats['missing_values_by_column']:
#         st.write("Missing Values by Column:")
#         missing_df = pd.DataFrame.from_dict(
#             stats['missing_values_by_column'],
#             orient='index',
#             columns=['Count']
#         )
#         missing_df = missing_df[missing_df['Count'] > 0]
#         if not missing_df.empty:
#             st.dataframe(missing_df)

# def get_csv_schema(df):
#     """Get schema information from DataFrame."""
#     schema_info = []
#     for column in df.columns:
#         dtype = str(df[column].dtype)
#         sample = df[column].iloc[0] if not df.empty else "N/A"
#         non_null = df[column].count()
#         null_count = df[column].isnull().sum()
        
#         schema_info.append({
#             "column": column,
#             "type": dtype,
#             "sample": str(sample),
#             "non_null_count": non_null,
#             "null_count": null_count
#         })
#     return schema_info

# def get_gemini_response(question, prompt):
#     """Get response from Gemini model."""
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content([prompt[0], question])
#         return response.text.strip()
#     except Exception as e:
#         st.error(f"Error generating SQL query: {e}")
#         return None

# def read_sql_query(sql, db):
#     """Execute SQL query and return results."""
#     try:
#         conn = sqlite3.connect(db)
#         cur = conn.cursor()
#         cur.execute(sql)
#         rows = cur.fetchall()
#         columns = [column[0] for column in cur.description]
#         conn.close()
#         return pd.DataFrame(rows, columns=columns)
#     except Exception as e:
#         st.error(f"Error executing SQL query: {e}")
#         return pd.DataFrame()

# def csv_to_sqlite(csv_file, db_name):
#     """Convert CSV to SQLite with data cleaning."""
#     try:
#         # Read the CSV file
#         df = pd.read_csv(csv_file)
        
#         # Display original data
#         st.subheader("Original Data Preview")
#         st.write("First 5 rows of original data:")
#         st.dataframe(df.head())
        
#         # Clean the data
#         cleaned_df, cleaning_stats = clean_data(df)
        
#         # Display cleaning statistics
#         display_cleaning_summary(cleaning_stats)
        
#         # Display cleaned data
#         st.subheader("Cleaned Data Preview")
#         st.write("First 5 rows of cleaned data:")
#         st.dataframe(cleaned_df.head())
        
#         # Convert to SQLite
#         conn = sqlite3.connect(db_name)
#         cleaned_df.to_sql('UPLOADED_DATA', conn, if_exists='replace', index=False)
#         conn.close()
        
#         # Get schema information
#         schema_info = get_csv_schema(cleaned_df)
#         return cleaned_df.columns.tolist(), schema_info, cleaned_df
#     except Exception as e:
#         st.error(f"Error converting CSV to SQLite: {e}")
#         return [], [], pd.DataFrame()

# def create_visualization(data, chart_type, x_column, y_column, title):
#     """Create visualization based on data and chart type."""
#     try:
#         if data.empty:
#             st.warning("No data available to visualize.")
#             return None
        
#         common_params = {
#             'title': title,
#             'template': 'plotly_white',
#             'color_discrete_sequence': [PRIMARY_COLOR, SECONDARY_COLOR]
#         }
        
#         if chart_type == "Line":
#             fig = px.line(data, x=x_column, y=y_column, **common_params)
#         elif chart_type == "Bar":
#             fig = px.bar(data, x=x_column, y=y_column, **common_params)
#         elif chart_type == "Scatter":
#             fig = px.scatter(data, x=x_column, y=y_column, **common_params)
#         elif chart_type == "Pie":
#             fig = px.pie(data, names=x_column, values=y_column, **common_params)
#         elif chart_type == "Histogram":
#             fig = px.histogram(data, x=x_column, **common_params)
#         else:
#             st.warning("Invalid chart type selected.")
#             return None

#         fig.update_layout(
#             plot_bgcolor='white',
#             paper_bgcolor='white',
#             font=dict(size=12),
#             margin=dict(l=40, r=40, t=40, b=40),
#             showlegend=True
#         )
        
#         if chart_type in ["Line", "Bar", "Scatter"]:
#             fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#             fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
#         return fig
#     except Exception as e:
#         st.error(f"Error creating visualization: {e}")
#         return None

# def get_fig_as_html(fig):
#     """Convert figure to HTML."""
#     buffer = io.StringIO()
#     fig.write_html(buffer)
#     return buffer.getvalue()

# def get_fig_as_png(fig):
#     """Convert figure to PNG."""
#     return pio.to_image(fig, format="png")

# def add_download_buttons(fig):
#     """Add download buttons for visualization."""
#     col1, col2 = st.columns(2)
#     with col1:
#         html_data = get_fig_as_html(fig)
#         st.download_button(
#             label="Download HTML",
#             data=html_data,
#             file_name="visualization.html",
#             mime="text/html"
#         )
#     with col2:
#         png_data = get_fig_as_png(fig)
#         st.download_button(
#             label="Download PNG",
#             data=png_data,
#             file_name="visualization.png",
#             mime="image/png"
#         )

# # Streamlit app
# st.set_page_config(page_title="Text to SQL Query Generator", layout="wide")

# # Initialize session state
# if 'query_data' not in st.session_state:
#     st.session_state.query_data = None
# if 'generated_query' not in st.session_state:
#     st.session_state.generated_query = ""
# if 'last_executed_query' not in st.session_state:
#     st.session_state.last_executed_query = ""

# # Sidebar
# with st.sidebar:
#     st.markdown("### Session Information")
#     st.write(f"**Current Time (UTC):** {CURRENT_TIME}")
#     st.write(f"**Current User:** {CURRENT_USER}")

# # Main content
# st.title("Text to SQL Query Generator")
# st.markdown("Transform your questions into SQL queries with AI assistance")

# # File uploader
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # Database setup
# if uploaded_file is not None:
#     db_name = 'uploaded_data.db'
#     columns, schema_info, df = csv_to_sqlite(uploaded_file, db_name)
#     table = 'UPLOADED_DATA'
    
#     st.markdown("### Dataset Information")
    
#     cols = st.columns(3)
#     with cols[0]:
#         st.markdown("**Number of Columns:**")
#         st.write(len(columns))
#     with cols[1]:
#         st.markdown("**Number of Rows:**")
#         st.write(len(df))
#     with cols[2]:
#         st.markdown("**File Size:**")
#         file_size = uploaded_file.size / 1024
#         if file_size < 1024:
#             st.write(f"{file_size:.2f} KB")
#         else:
#             st.write(f"{file_size/1024:.2f} MB")
    
#     st.markdown("### Schema Details")
#     schema_df = pd.DataFrame(schema_info)
#     st.dataframe(schema_df)

# else:
#     db_name = 'student.db'
#     table = 'STUDENT'
#     columns = ["NAME", "CLASS", "SECTION", "MARKS"]

# # Query prompt
# prompt = [
#     f"""
#     You are an expert in converting English questions to SQL queries!
#     The SQL database is named {table} and has the following columns: {', '.join(columns)}.
#     Examples:
#     - Show total number of records -> SELECT COUNT(*) FROM {table};
#     - Find students in Data Science class -> SELECT * FROM {table} WHERE CLASS='Data Science';
#     - Get average marks by section -> SELECT SECTION, AVG(MARKS) FROM {table} GROUP BY SECTION;
    
#     Please provide only the SQL query without any additional text or formatting.
#     """
# ]

# # Query interface
# st.markdown("---")
# st.subheader("Ask Your Question")
# question = st.text_input("Your question:", key="nl_input")

# if st.button("Generate SQL", key="generate_btn"):
#     if question:
#         response = get_gemini_response(question, prompt)
#         if response:
#             st.session_state.generated_query = response
#             try:
#                 data = read_sql_query(response, db_name)
#                 if not data.empty:
#                     st.session_state.query_data = data
#                     st.session_state.last_executed_query = response
#                     st.success("✅ Query generated and executed successfully!")
#                 else:
#                     st.warning("Query returned no results")
#                     st.session_state.query_data = None
#             except Exception as e:
#                 st.error(f"Error executing query: {e}")
#                 st.session_state.query_data = None

# # Display generated SQL
# if st.session_state.generated_query:
#     st.markdown("### Generated SQL Query")
#     st.code(st.session_state.generated_query, language="sql")

# # Manual SQL Editor
# st.subheader("SQL Query Editor")
# manual_query = st.text_area(
#     "Edit SQL Query:",
#     value=st.session_state.generated_query,
#     height=150,
#     key="manual_query"
# )

# col1, col2 = st.columns([1, 1])
# with col1:
#     if manual_query.strip():
#         if st.button("Execute Query", key="execute_btn"):
#             try:
#                 data = read_sql_query(manual_query, db_name)
#                 if not data.empty:
#                     st.session_state.query_data = data
#                     st.session_state.last_executed_query = manual_query
#                     st.success("Query executed successfully!")
#                 else:
#                     st.warning("Query returned no results.")
#                     st.session_state.query_data = None
#             except Exception as e:
#                 st.error(f"Error executing query: {e}")
#                 st.session_state.query_data = None

# with col2:
#     if st.button("Clear Query", key="clear_btn"):
#         st.session_state.generated_query = ""
#         st.session_state.query_data = None
#         st.session_state.last_executed_query = ""
#         st.rerun()

# # Query Results
# if st.session_state.query_data is not None and not st.session_state.query_data.empty:
#     st.markdown("### Query Results")
#     st.dataframe(st.session_state.query_data)
    
#     # Visualization section
#     st.subheader("Data Visualization")
    
#     if len(st.session_state.query_data.columns) >= 1:
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             chart_type = st.selectbox(
#                 "Select Chart Type",
#                 ["Line", "Bar", "Scatter", "Pie", "Histogram"]
#             )
        
#         with col2:
#             x_column = st.selectbox("Select X-axis", st.session_state.query_data.columns)
        
#         with col3:
#             if chart_type != "Histogram":
#                 y_column = st.selectbox("Select Y-axis", st.session_state.query_data.columns)
#             else:
#                 y_column = None

#         title = st.text_input("Plot Title", "Data Visualization")

#         tab1, tab2 = st.tabs(["Chart", "Data"])
        
#         with tab1:
#             fig = create_visualization(
#                 st.session_state.query_data,
#                 chart_type,
#                 x_column,
#                 y_column,
#                 title
#             )

#             if fig:
#                 st.plotly_chart(fig, use_container_width=True)
#                 add_download_buttons(fig)
#             else:
#                 st.warning("Could not create visualization with the selected options.")
        
#         with tab2:
#             st.write("Data used for visualization:")
#             st.write(st.session_state.query_data)
#     else:
#         st.write("Not enough data to visualize.")

# # Download query results
# if st.session_state.query_data is not None and not st.session_state.query_data.empty:
#     csv = st.session_state.query_data.to_csv(index=False)
#     st.download_button(
#         label="Download Results as CSV",
#         data=csv,
#         file_name="query_results.csv",
#         mime="text/csv"
#     )

# # Footer
# st.markdown("---")
# st.markdown(f"*Last updated: {CURRENT_TIME} UTC by {CURRENT_USER}*")


from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import numpy as np
from datetime import datetime

# Configure API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Current time and user information
CURRENT_TIME = "2025-02-13 05:14:28"
CURRENT_USER = "piyushk1304"

# Fixed colors for visualization
PRIMARY_COLOR = "#1f77b4"  # Blue
SECONDARY_COLOR = "#ff7f0e"  # Orange

# def clean_data(df):
#     """Perform comprehensive data cleaning."""
#     stats = {}
    
#     # Store original shape
#     stats['original_rows'] = len(df)
#     stats['original_columns'] = len(df.columns)
    
#     # Check for duplicates
#     stats['duplicates_removed'] = df.duplicated().sum()
#     df = df.drop_duplicates()
    
#     # Handle missing values
#     stats['total_missing_values'] = df.isnull().sum().sum()
#     stats['missing_values_by_column'] = df.isnull().sum().to_dict()
    
#     # Process numeric columns
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     stats['numeric_columns_cleaned'] = []
    
#     for col in numeric_columns:
#         # Store original stats
#         stats[f'{col}_before_mean'] = df[col].mean()
#         stats[f'{col}_before_std'] = df[col].std()
        
#         # Handle outliers
#         mean = df[col].mean()
#         std = df[col].std()
#         df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
        
#         # Store cleaned stats
#         stats[f'{col}_after_mean'] = df[col].mean()
#         stats[f'{col}_after_std'] = df[col].std()
#         stats['numeric_columns_cleaned'].append(col)
    
#     # Clean string columns
#     string_columns = df.select_dtypes(include=['object']).columns
#     for col in string_columns:
#         df[col] = df[col].astype(str).str.strip()
#         df[col] = df[col].replace('', np.nan)
#         df[col] = df[col].replace('null', np.nan)
#         df[col] = df[col].replace('NULL', np.nan)
#         df[col] = df[col].replace('None', np.nan)
    
#     # Final cleaning
#     df = df.dropna()
    
#     # Store final shape
#     stats['final_rows'] = len(df)
#     stats['final_columns'] = len(df.columns)
    
#     return df, stats
def clean_data(df):
    """Perform comprehensive data cleaning."""
    stats = {}
    
    # Store original shape
    stats['original_rows'] = len(df)
    stats['original_columns'] = len(df.columns)
    
    # Check for duplicates
    stats['duplicates_removed'] = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Handle missing values with specific replacements
    replacements = {
        'CLASS': 'Not Specified',
        'SECTION': 'Not Specified',
        'MARKS': 0.0,
        'ATTENDANCE': 0.0,
        'GRADE_POINTS': 0.0,
        'STATUS': 'Unknown'
    }
    
    # Handle missing values
    stats['total_missing_values'] = df.isnull().sum().sum()
    stats['missing_values_by_column'] = df.isnull().sum().to_dict()
    
    # Replace NaN values with specific defaults
    for column, replacement in replacements.items():
        if column in df.columns:
            df[column] = df[column].fillna(replacement)
    
    # Process numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    stats['numeric_columns_cleaned'] = []
    
    for col in numeric_columns:
        # Store original stats
        stats[f'{col}_before_mean'] = df[col].mean()
        stats[f'{col}_before_std'] = df[col].std()
        
        # Handle outliers
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
        
        # Store cleaned stats
        stats[f'{col}_after_mean'] = df[col].mean()
        stats[f'{col}_after_std'] = df[col].std()
        stats['numeric_columns_cleaned'].append(col)
    
    # Clean string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        # Convert to string and clean
        df[col] = df[col].astype(str).str.strip()
        # Replace empty strings and 'nan' with appropriate values
        if col in replacements:
            df[col] = df[col].replace(['', 'nan', 'null', 'NULL', 'None'], replacements[col])
    
    # Ensure date format consistency
    if 'SUBMISSION_DATE' in df.columns:
        df['SUBMISSION_DATE'] = pd.to_datetime(df['SUBMISSION_DATE']).dt.strftime('%Y-%m-%d')
    
    # Store final shape
    stats['final_rows'] = len(df)
    stats['final_columns'] = len(df.columns)
    
    return df, stats

def display_cleaning_summary(stats):
    """Display data cleaning statistics."""
    st.subheader("Data Cleaning Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Rows Processed",
            value=stats['final_rows'],
            delta=f"{stats['final_rows'] - stats['original_rows']} rows removed"
        )
    
    with col2:
        st.metric(
            label="Duplicates Removed",
            value=stats['duplicates_removed']
        )
    
    with col3:
        st.metric(
            label="Missing Values Handled",
            value=stats['total_missing_values']
        )

    if stats['missing_values_by_column']:
        st.write("Missing Values by Column:")
        missing_df = pd.DataFrame.from_dict(
            stats['missing_values_by_column'],
            orient='index',
            columns=['Count']
        )
        missing_df = missing_df[missing_df['Count'] > 0]
        if not missing_df.empty:
            st.dataframe(missing_df)

def get_csv_schema(df):
    """Get schema information from DataFrame."""
    schema_info = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample = df[column].iloc[0] if not df.empty else "N/A"
        non_null = df[column].count()
        null_count = df[column].isnull().sum()
        
        schema_info.append({
            "column": column,
            "type": dtype,
            "sample": str(sample),
            "non_null_count": non_null,
            "null_count": null_count
        })
    return schema_info

def get_gemini_response(question, prompt):
    """Get response from Gemini model."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt[0], question])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

def read_sql_query(sql, db):
    """Execute SQL query and return results."""
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [column[0] for column in cur.description]
        conn.close()
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return pd.DataFrame()

def csv_to_sqlite(csv_file, db_name):
    """Convert CSV to SQLite with data cleaning."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Display original data
        st.subheader("Original Data Preview")
        st.write("First 5 rows of original data:")
        st.dataframe(df.head())
        
        # Clean the data
        cleaned_df, cleaning_stats = clean_data(df)
        
        # Display cleaning statistics
        display_cleaning_summary(cleaning_stats)
        
        # Display cleaned data
        st.subheader("Cleaned Data Preview")
        st.write("First 5 rows of cleaned data:")
        st.dataframe(cleaned_df.head())
        
        # Convert to SQLite
        conn = sqlite3.connect(db_name)
        cleaned_df.to_sql('UPLOADED_DATA', conn, if_exists='replace', index=False)
        conn.close()
        
        # Get schema information
        schema_info = get_csv_schema(cleaned_df)
        return cleaned_df.columns.tolist(), schema_info, cleaned_df
    except Exception as e:
        st.error(f"Error converting CSV to SQLite: {e}")
        return [], [], pd.DataFrame()

def create_visualization(data, chart_type, x_column, y_column, title):
    """Create visualization based on data and chart type."""
    try:
        if data.empty:
            st.warning("No data available to visualize.")
            return None
        
        common_params = {
            'title': title,
            'template': 'plotly_white',
            'color_discrete_sequence': [PRIMARY_COLOR, SECONDARY_COLOR]
        }
        
        if chart_type == "Line":
            fig = px.line(data, x=x_column, y=y_column, **common_params)
        elif chart_type == "Bar":
            fig = px.bar(data, x=x_column, y=y_column, **common_params)
        elif chart_type == "Scatter":
            fig = px.scatter(data, x=x_column, y=y_column, **common_params)
        elif chart_type == "Pie":
            fig = px.pie(data, names=x_column, values=y_column, **common_params)
        elif chart_type == "Histogram":
            fig = px.histogram(data, x=x_column, **common_params)
        else:
            st.warning("Invalid chart type selected.")
            return None

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        
        if chart_type in ["Line", "Bar", "Scatter"]:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def get_fig_as_html(fig):
    """Convert figure to HTML."""
    buffer = io.StringIO()
    fig.write_html(buffer)
    return buffer.getvalue()

def get_fig_as_png(fig):
    """Convert figure to PNG."""
    return pio.to_image(fig, format="png")

def add_download_buttons(fig):
    """Add download buttons for visualization."""
    col1, col2 = st.columns(2)
    with col1:
        html_data = get_fig_as_html(fig)
        st.download_button(
            label="Download HTML",
            data=html_data,
            file_name="visualization.html",
            mime="text/html"
        )
    with col2:
        png_data = get_fig_as_png(fig)
        st.download_button(
            label="Download PNG",
            data=png_data,
            file_name="visualization.png",
            mime="image/png"
        )

# Streamlit app
st.set_page_config(page_title="Text to SQL Query Generator", layout="wide")

# Initialize session state
if 'query_data' not in st.session_state:
    st.session_state.query_data = None
if 'generated_query' not in st.session_state:
    st.session_state.generated_query = ""
if 'last_executed_query' not in st.session_state:
    st.session_state.last_executed_query = ""

# Sidebar
with st.sidebar:
    st.markdown("### Session Information")
    st.write(f"**Current Time (UTC):** {CURRENT_TIME}")
    st.write(f"**Current User:** {CURRENT_USER}")

# Main content
st.title("Text to SQL Query Generator")
st.markdown("Transform your questions into SQL queries with AI assistance")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Database setup
if uploaded_file is not None:
    db_name = 'uploaded_data.db'
    columns, schema_info, df = csv_to_sqlite(uploaded_file, db_name)
    table = 'UPLOADED_DATA'
    
    st.markdown("### Dataset Information")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Number of Columns:**")
        st.write(len(columns))
    with cols[1]:
        st.markdown("**Number of Rows:**")
        st.write(len(df))
    with cols[2]:
        st.markdown("**File Size:**")
        file_size = uploaded_file.size / 1024
        if file_size < 1024:
            st.write(f"{file_size:.2f} KB")
        else:
            st.write(f"{file_size/1024:.2f} MB")
    
    st.markdown("### Schema Details")
    schema_df = pd.DataFrame(schema_info)
    st.dataframe(schema_df)

else:
    db_name = 'student.db'
    table = 'STUDENT'
    columns = ["NAME", "CLASS", "SECTION", "MARKS"]

# Query prompt
prompt = [
    f"""
    You are an expert in converting English questions to SQL queries!
    The SQL database is named {table} and has the following columns: {', '.join(columns)}.
    Examples:
    - Show total number of records -> SELECT COUNT(*) FROM {table};
    - Find students in Data Science class -> SELECT * FROM {table} WHERE CLASS='Data Science';
    - Get average marks by section -> SELECT SECTION, AVG(MARKS) FROM {table} GROUP BY SECTION;
    
    Please provide only the SQL query without any additional text or formatting.
    """
]

# Query interface
st.markdown("---")
st.subheader("Ask Your Question")
question = st.text_input("Your question:", key="nl_input")

if st.button("Generate SQL", key="generate_btn"):
    if question:
        response = get_gemini_response(question, prompt)
        if response:
            st.session_state.generated_query = response
            try:
                data = read_sql_query(response, db_name)
                if not data.empty:
                    st.session_state.query_data = data
                    st.session_state.last_executed_query = response
                    st.success("✅ Query generated and executed successfully!")
                else:
                    st.warning("Query returned no results")
                    st.session_state.query_data = None
            except Exception as e:
                st.error(f"Error executing query: {e}")
                st.session_state.query_data = None

# Manual SQL Editor
st.subheader("SQL Query Editor")
manual_query = st.text_area(
    "Edit SQL Query:",
    value=st.session_state.generated_query,
    height=150,
    key="manual_query"
)

col1, col2 = st.columns([1, 1])
with col1:
    if manual_query.strip():
        if st.button("Execute Query", key="execute_btn"):
            try:
                data = read_sql_query(manual_query, db_name)
                if not data.empty:
                    st.session_state.query_data = data
                    st.session_state.last_executed_query = manual_query
                    st.success("Query executed successfully!")
                else:
                    st.warning("Query returned no results.")
                    st.session_state.query_data = None
            except Exception as e:
                st.error(f"Error executing query: {e}")
                st.session_state.query_data = None

with col2:
    if st.button("Clear Query", key="clear_btn"):
        st.session_state.generated_query = ""
        st.session_state.query_data = None
        st.session_state.last_executed_query = ""
        st.rerun()

# Query Results
if st.session_state.query_data is not None and not st.session_state.query_data.empty:
    st.markdown("### Query Results")
    st.dataframe(st.session_state.query_data)
    
    # Download query results
    csv = st.session_state.query_data.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="query_results.csv",
        mime="text/csv"
    )
    
    # Visualization section
    st.subheader("Data Visualization")
    
    if len(st.session_state.query_data.columns) >= 1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line", "Bar", "Scatter", "Pie", "Histogram"]
            )
        
        with col2:
            x_column = st.selectbox("Select X-axis", st.session_state.query_data.columns)
        
        with col3:
            if chart_type != "Histogram":
                y_column = st.selectbox("Select Y-axis", st.session_state.query_data.columns)
            else:
                y_column = None

        title = st.text_input("Plot Title", "Data Visualization")

        tab1, tab2 = st.tabs(["Chart", "Data"])
        
        with tab1:
            fig = create_visualization(
                st.session_state.query_data,
                chart_type,
                x_column,
                y_column,
                title
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
                add_download_buttons(fig)
            else:
                st.warning("Could not create visualization with the selected options.")
        
        with tab2:
            st.write("Data used for visualization:")
            st.write(st.session_state.query_data)
    else:
        st.write("Not enough data to visualize.")

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {CURRENT_TIME} UTC by {CURRENT_USER}*")