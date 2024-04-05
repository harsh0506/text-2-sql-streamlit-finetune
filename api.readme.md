# SQL Query Generator API



# Future scope

## Overview
The SQL Query Generator API is a versatile tool that allows users to upload various types of data files (such as CSV, JSON, TSV, DOCX, Parquet) and generate SQL queries based on their questions about the data. The API provides preprocessing capabilities to handle data type conversions, basic datetime operations, and compatibility with multiple file formats. Additionally, it integrates with a machine learning model to ensure safe query execution and supports continuous chat interactions through websockets.

## Features
1. **Support for Multiple File Formats**: Users can upload CSV, JSON, TSV, DOCX, and Parquet files for data analysis.
2. **Data Preprocessing**: The API automatically preprocesses uploaded files, including data type conversions and basic datetime operations.
3. **Statistical Analysis**: Basic analytics, such as statistical reports, are generated for the uploaded data to provide insights.
4. **Continuous Chat Interface**: Users can engage in continuous chat interactions to ask questions about the data and receive SQL query results in real-time.
5. **Safe Query Execution**: The API ensures safe query execution by connecting to a SQL database and performing queries securely.
6. **Multi-User Support**: Each user can have multiple data files associated with their account, enabling seamless interaction with various datasets.

## Requirements
- Flask
- SQLAlchemy
- Pandas
- PySpark
- Transformers
- Hugging Face Models
- Websockets

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Users can interact with the API through HTTP requests or a frontend client.
3. Upload data files using the appropriate endpoint (`/upload`) along with the file format.
4. Ask questions about the data through the chat interface or send queries directly.
5. Receive SQL query results and insights in real-time.
6. Data files and query results are stored securely in the database.

## API Endpoints
- `/upload`: Upload data files (CSV, JSON, TSV, DOCX, Parquet).
- `/chat`: Initiate a chat session for continuous interactions.
- `/query`: Send SQL queries directly to the API.

## Error Handling
- **400 Bad Request**: Returned when the request is malformed or missing required parameters.
- **401 Unauthorized**: Returned when the user authentication fails.
- **404 Not Found**: Returned when the requested resource does not exist.
- **500 Internal Server Error**: Returned when an unexpected server error occurs.

## Security
- **Authentication**: Users are authenticated before accessing any functionalities of the API.
- **Secure Query Execution**: Queries are executed securely to prevent SQL injection attacks.
- **Data Encryption**: User data and query results are encrypted to ensure confidentiality.

## Future Enhancements
- Support for additional file formats (XML, Excel).
- Integration with more advanced machine learning models for better query generation.
- Implementation of advanced analytics and visualization capabilities.
- Scalability improvements for handling large datasets and concurrent users.
- Deployment on cloud platforms for improved accessibility and performance.
