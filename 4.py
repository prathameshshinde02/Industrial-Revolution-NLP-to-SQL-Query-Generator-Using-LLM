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

# Streamlit app configuration must come first
st.set_page_config(page_title="IntelliQuery App", layout="wide")

# Configure API key
if not os.getenv('GOOGLE_API_KEY'):
    st.error("🚨 Missing Google API Key! Please set GOOGLE_API_KEY in .env.")
    st.stop()


# Current time and user information (kept in backend)
CURRENT_TIME = "2025-02-13 06:22:12"  # UTC - YYYY-MM-DD HH:MM:SS formatted
CURRENT_USER = "piyushk1304"

# Fixed colors for visualization
PRIMARY_COLOR = "#1f77b4"  # Blue
SECONDARY_COLOR = "#ff7f0e"  # Orange

def get_schema_info(df):
    """Get detailed schema information for the DataFrame."""
    schema_info = []
    
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample = df[column].dropna().iloc[0] if not df[column].dropna().empty else None
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        unique_count = df[column].nunique()
        
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            inferred_type = 'datetime'
        elif pd.api.types.is_numeric_dtype(df[column]):
            inferred_type = 'numeric'
        else:
            inferred_type = 'categorical'
        
        schema_info.append({
            'column_name': column,
            'pandas_dtype': dtype,
            'inferred_type': inferred_type,
            'null_count': null_count,
            'null_percentage': f"{null_percentage:.2f}%",
            'unique_values': unique_count,
            'sample_value': str(sample) if sample is not None else 'None'
        })
    
    return schema_info

def detect_column_types(df):
    """Detect and classify column types in the DataFrame."""
    column_types = {}
    
    for column in df.columns:
        try:
            sample = df[column].dropna().iloc[0] if not df[column].dropna().empty else None
            if sample and isinstance(sample, str):
                if any(date_indicator in sample.lower() for date_indicator in ['date', 'time', ':', '-', '/']):
                    pd.to_datetime(df[column], errors='raise')
                    column_types[column] = 'datetime'
                    continue
        except (ValueError, AttributeError):
            pass
        
        try:
            pd.to_numeric(df[column], errors='raise')
            column_types[column] = 'numeric'
        except (ValueError, TypeError):
            column_types[column] = 'categorical'
    
    return column_types

def clean_data(df):
    """Perform comprehensive data cleaning with appropriate type handling."""
    stats = {}
    
    stats['original_rows'] = len(df)
    stats['original_columns'] = len(df.columns)
    stats['duplicates_removed'] = df.duplicated().sum()
    df = df.drop_duplicates()
    
    column_types = detect_column_types(df)
    stats['column_types'] = column_types
    
    for column, col_type in column_types.items():
        if col_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
            if not df[column].isna().all():
                df[column] = df[column].dt.strftime('%Y-%m-%d %H:%M:%S')
                stats[f'{column}_datetime_format'] = 'YYYY-MM-DD HH:MM:SS'
            
        elif col_type == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
            stats[f'{column}_before_mean'] = df[column].mean()
            stats[f'{column}_before_std'] = df[column].std()
            column_mean = df[column].mean()
            df[column] = df[column].fillna(column_mean)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            stats[f'{column}_after_mean'] = df[column].mean()
            stats[f'{column}_after_std'] = df[column].std()
        
        else:  # categorical
            df[column] = df[column].astype(str).str.strip()
            mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
            df[column] = df[column].replace(['', 'nan', 'null', 'NULL', 'None', 'NaN'], mode_value)
            stats[f'{column}_unique_values'] = df[column].nunique()
            stats[f'{column}_mode'] = mode_value
    
    numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
    if len(numeric_columns) > 1:
        stats['correlations'] = df[numeric_columns].corr().to_dict()
    
    stats['final_rows'] = len(df)
    stats['final_columns'] = len(df.columns)
    stats['missing_values_filled'] = df.isnull().sum().to_dict()
    
    return df, stats

def display_cleaning_summary(stats):
    """📊 Data Cleaning Summary"""
    st.subheader("📊 Data Cleaning Summary")
    
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
            label="Columns Processed",
            value=stats['final_columns']
        )

def get_gemini_response(question, prompt):
    """Get response from Gemini model."""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt[0], question])
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return None


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
        df = pd.read_csv(csv_file, encoding='utf-8', engine='python', on_bad_lines='skip')
        
        st.subheader("📋 CSV Schema Information")
        schema_info = get_schema_info(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### ℹ️ Basic Information")
            st.write(f"Total Rows: {len(df)}")
            st.write(f"Total Columns: {len(df.columns)}")
            
        with col2:
            st.markdown("#### 📊 Data Types Summary")
            dtype_counts = {
                'numeric': len([x for x in schema_info if x['inferred_type'] == 'numeric']),
                'categorical': len([x for x in schema_info if x['inferred_type'] == 'categorical']),
                'datetime': len([x for x in schema_info if x['inferred_type'] == 'datetime'])
            }
            st.write(dtype_counts)
            
        with col3:
            st.markdown("#### ❓ Missing Values")
            total_nulls = sum(df.isnull().sum())
            st.write(f"Total Null Values: {total_nulls}")
            st.write(f"Data Completeness: {((1 - total_nulls/(len(df)*len(df.columns))) * 100):.2f}%")
        
        st.markdown("#### 🔍 Detailed Schema Information")
        schema_df = pd.DataFrame(schema_info)
        st.dataframe(
            schema_df[[
                'column_name', 'inferred_type', 'pandas_dtype',
                'null_count', 'null_percentage', 'unique_values',
                'sample_value'
            ]],
            hide_index=True
        )
        
        st.subheader("👀 Original Data Preview")
        st.write("First 5 rows of original data:")
        st.dataframe(df.head())
        
        cleaned_df, cleaning_stats = clean_data(df)
        
        display_cleaning_summary(cleaning_stats)
        
        st.subheader("✨ Cleaned Data Preview")
        st.write("First 5 rows of cleaned data:")
        st.dataframe(cleaned_df.head())
        
        conn = sqlite3.connect(db_name)
        cleaned_df.to_sql('UPLOADED_DATA', conn, if_exists='replace', index=False)
        conn.close()
        
        return cleaned_df.columns.tolist(), cleaning_stats, cleaned_df
    except Exception as e:
        st.error(f"Error converting CSV to SQLite: {e}")
        return [], {}, pd.DataFrame()

def create_visualization(data, chart_type, x_column, y_column, title):
    """Create visualization based on data and chart_type."""
    try:
        if data.empty:
            st.warning("No data available to visualize.")
            return None
        
        color_palette = [
            "#FF6347", "#4682B4", "#3CB371", "#FFD700", "#BA55D3", 
            "#FFA07A", "#20B2AA", "#DAA520", "#00CED1", "#8B008B"
        ]
        
        common_params = {
            'title': title,
            'template': 'plotly_white',
            'color_discrete_sequence': color_palette
        }
        
        continuous_color = 'viridis'

        if chart_type == "Line":
            fig = px.line(data, x=x_column, y=y_column, **common_params)
        elif chart_type == "Bar":
            fig = px.bar(data, x=x_column, y=y_column, **common_params)
        elif chart_type == "Scatter":
            fig = px.scatter(data, x=x_column, y=y_column, color=y_column if y_column else x_column, 
                           color_continuous_scale=continuous_color, **common_params)
        elif chart_type == "Pie":
            fig = px.pie(data, names=x_column, values=y_column, **common_params)
        elif chart_type == "Histogram":
            if 'color' in data.columns:
                fig = px.histogram(data, x=x_column, color='color', **common_params)
            else:
                fig = px.histogram(data, x=x_column, **common_params)
        else:
            st.warning("Invalid chart_type selected.")
            return None

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        
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

# Initialize session state
if 'query_data' not in st.session_state:
    st.session_state.query_data = None
if 'generated_query' not in st.session_state:
    st.session_state.generated_query = ""
if 'last_executed_query' not in st.session_state:
    st.session_state.last_executed_query = ""

# Display current time and user
st.sidebar.markdown(f"**⏰ Current Time (UTC):** {CURRENT_TIME}")
st.sidebar.markdown(f"**👤 Current User:** {CURRENT_USER}")

# Main content
st.title("🌟 IntelliQuery – Your Data, Your Query, AI-Driven.")
st.markdown("Transform your questions into SQL queries with AI assistance")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

# Database setup
if uploaded_file is not None:
    db_name = 'uploaded_data.db'
    columns, cleaning_stats, df = csv_to_sqlite(uploaded_file, db_name)
    table = 'UPLOADED_DATA'
    
    if len(columns) > 0:
        prompt = [
            f"""
            You are an expert in converting English questions to SQL queries!
            The SQL database has the following columns: {', '.join(columns)}.
            Examples:
            - Show total number of records -> SELECT COUNT(*) FROM {table};
            - Find average values -> SELECT AVG(column_name) FROM {table};
            - Get distribution by category -> SELECT category_column, COUNT(*) FROM {table} GROUP BY category_column;
            
            Please provide only the SQL query without any additional text or formatting.
            """
        ]

        st.markdown("---")
        st.subheader("❓ Ask Your Question")
        question = st.text_input("Your question:", key="nl_input")

        if st.button("🚀 Generate SQL", key="generate_btn"):
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

        st.subheader("✏️ SQL Query Editor")
        manual_query = st.text_area(
            "Edit SQL Query:",
            value=st.session_state.generated_query,
            height=150,
            key="manual_query"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if manual_query.strip():
                if st.button("▶️ Execute Query", key="execute_btn"):
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
            if st.button("🗑️ Clear Query", key="clear_btn"):
                st.session_state.generated_query = ""
                st.session_state.query_data = None
                st.session_state.last_executed_query = ""
                st.rerun()

        if st.session_state.query_data is not None and not st.session_state.query_data.empty:
            st.markdown("### 📈 Query Results")
            st.dataframe(st.session_state.query_data)
            
            csv = st.session_state.query_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
            
            st.subheader("🎨 Data Visualization")
            
            if len(st.session_state.query_data.columns) >= 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    chart_type = st.selectbox(
                        "Select Chart_type",
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

                tab1, tab2 = st.tabs(["📊 Chart", "📋 Data"])
                
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

if st.session_state.query_data is not None and not st.session_state.query_data.empty:
    st.markdown("---")
    st.subheader("💡 Data Insights")
    
    numeric_cols = st.session_state.query_data.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        st.write("Numeric Column Statistics:")
        st.dataframe(st.session_state.query_data[numeric_cols].describe())
    
    categorical_cols = st.session_state.query_data.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        st.write("Categorical Column Distribution:")
        for col in categorical_cols:
            st.write(f"\n**{col}** Distribution:")
            st.dataframe(st.session_state.query_data[col].value_counts().head())

# Update timestamps
CURRENT_TIME = "2025-02-13 06:23:59"  # UTC - YYYY-MM-DD HH:MM:SS formatted
CURRENT_USER = "piyushk1304"

backend_log = {
    'timestamp': CURRENT_TIME,
    'user': CURRENT_USER,
    'session_id': os.getenv('STREAMLIT_SESSION_ID', 'unknown'),
    'last_query': st.session_state.get('last_executed_query', ''),
    'data_processed': bool(st.session_state.get('query_data') is not None)
}