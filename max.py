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
from datetime import datetime

# Configure API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Current time and user information
CURRENT_TIME = "2025-02-08 05:11:15"
CURRENT_USER = "piyushk1304"

# Fixed colors for visualization
PRIMARY_COLOR = "#1f77b4"  # Blue
SECONDARY_COLOR = "#ff7f0e"  # Orange

def get_csv_schema(df):
    """Get the schema information from a pandas DataFrame."""
    schema_info = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample = df[column].iloc[0] if not df.empty else "N/A"
        schema_info.append({
            "column": column,
            "type": dtype,
            "sample": str(sample)
        })
    return schema_info

def get_gemini_response(question, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt[0], question])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

def read_sql_query(sql, db):
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
    try:
        df = pd.read_csv(csv_file)
        conn = sqlite3.connect(db_name)
        df.to_sql('UPLOADED_DATA', conn, if_exists='replace', index=False)
        conn.close()
        
        # Get schema information
        schema_info = get_csv_schema(df)
        return df.columns.tolist(), schema_info, df
    except Exception as e:
        st.error(f"Error converting CSV to SQLite: {e}")
        return [], [], pd.DataFrame()

def create_visualization(data, chart_type, x_column, y_column, title):
    try:
        if data.empty:
            st.warning("No data available to visualize.")
            return None
        
        # Common parameters for all charts
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

        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        
        # Add grid lines for certain chart types
        if chart_type in ["Line", "Bar", "Scatter"]:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def get_fig_as_html(fig):
    buffer = io.StringIO()
    fig.write_html(buffer)
    return buffer.getvalue()

def get_fig_as_png(fig):
    return pio.to_image(fig, format="png")

def add_download_buttons(fig):
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
st.set_page_config(page_title="IntelliQuery App", layout="wide")

# Initialize session state variables
if 'query_data' not in st.session_state:
    st.session_state.query_data = None
if 'generated_query' not in st.session_state:
    st.session_state.generated_query = ""
if 'last_executed_query' not in st.session_state:
    st.session_state.last_executed_query = ""

# Sidebar with session information
with st.sidebar:
    st.markdown("### Session Information")
    st.write(f"**Current Time (UTC):** {CURRENT_TIME}")
    st.write(f"**Current User:** {CURRENT_USER}")

# Main content
st.header("IntelliQuery – Your Data, Your Query, AI-Driven.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Database setup
if uploaded_file is not None:
    db_name = 'uploaded_data.db'
    columns, schema_info, df = csv_to_sqlite(uploaded_file, db_name)
    table = 'UPLOADED_DATA'
    
    # Display CSV schema information
    st.success("CSV file successfully uploaded and converted to SQLite database.")
    
    st.subheader("CSV Schema Information")
    
    # Create three columns for better visualization
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("**Number of Columns:**")
        st.write(len(columns))
    
    with cols[1]:
        st.markdown("**Number of Rows:**")
        st.write(len(df))
    
    with cols[2]:
        st.markdown("**File Size:**")
        file_size = uploaded_file.size / 1024  # Convert to KB
        if file_size < 1024:
            st.write(f"{file_size:.2f} KB")
        else:
            st.write(f"{file_size/1024:.2f} MB")
    
    # Display detailed schema information
    st.markdown("### Detailed Schema")
    
    # Create and style the schema DataFrame
    schema_df = pd.DataFrame(schema_info)
    st.dataframe(
        schema_df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': 'black',
            'border-color': 'white'
        })
    )

else:
    db_name = 'student.db'
    table = 'STUDENT'
    columns = ["NAME", "CLASS", "SECTION", "MARKS"]

# Query prompt
prompt = [
    f"""
        You are an expert in converting English questions to SQL query!
        The SQL database is named {table} and has the following columns: {', '.join(columns)}.
        For example:
        - How many entries of records are present? -> SELECT COUNT(*) FROM {table};
        - Tell me all the students studying in Data Science class? -> SELECT * FROM {table} WHERE CLASS='Data Science';
        Please provide the SQL query without backticks or the word 'sql'.
    """
]

# Query interface
st.subheader("Query Input")

# Natural Language Input
question = st.text_input("Input your question:", key="nl_input")
if st.button("Generate SQL", key="generate_btn"):
    if question:
        response = get_gemini_response(question, prompt)
        if response:
            st.session_state.generated_query = response
            # Execute the generated query automatically
            try:
                data = read_sql_query(response, db_name)
                if not data.empty:
                    st.session_state.query_data = data
                    st.session_state.last_executed_query = response
                    st.success("Query executed successfully!")
                else:
                    st.warning("Query returned no results.")
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
    # Show execute button if there's a query to execute
    if manual_query.strip():  # Check if there's any non-whitespace content
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

# Display query results
if st.session_state.query_data is not None and not st.session_state.query_data.empty:
    st.subheader("Query Results:")
    st.write(st.session_state.query_data)

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

        # Plot customization
        title = st.text_input("Plot Title", "Data Visualization")

        # Create tabs for visualization
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