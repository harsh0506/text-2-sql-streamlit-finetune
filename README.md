# SQL Query Generator App

## Overview
This is a Streamlit web application for generating SQL queries based on questions asked by users about a CSV file's data. The application utilizes the T5 model from the Transformers library to convert user questions into SQL queries. Users can upload a CSV file, explore its columns and statistical reports, visualize basic plots, and then ask questions about the data. The application will generate SQL queries corresponding to the questions, execute them using PySpark, and display the query results.

## Working
1. **Upload CSV File**: Users can upload a CSV file containing the data they want to query.
2. **Data Exploration**: Once a CSV file is uploaded, users can view the column names, data types, and statistical reports (like mean, median, etc.).
3. **Data Visualization**: Basic box plots are generated for numerical columns to visualize data distributions.
4. **SQL Query Generation**: Users can ask questions about the data using the chat interface. The application then generates SQL queries based on the questions asked.
5. **SQL Query Execution**: The generated SQL queries are executed using PySpark on the uploaded CSV file.
6. **Display Results**: The query results are displayed back to the user via the chat interface.

## Features
1. **Upload CSV File**: Users can upload their own CSV files for querying.
2. **Data Exploration**: Users can explore the structure and statistical properties of the uploaded data.
3. **Data Visualization**: Basic box plots are generated to visualize data distributions for numerical columns.
4. **Chat Interface**: Users can interact with the application using a chat interface. They can ask questions about the data, and the application responds with SQL queries and query results.
5. **SQL Query Generation**: The application utilizes the T5 model to generate SQL queries based on user questions.
6. **SQL Query Execution**: PySpark is used to execute the generated SQL queries on the uploaded CSV data.
7. **History**: Chat history is maintained, allowing users to see previous interactions.

## Requirements
- Streamlit
- PyTorch
- Transformers
- PySpark
- Pandas
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/harsh0506/text-2-sql-streamlit-finetune
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload a CSV file.
3. Explore the data and visualize basic plots.
4. Ask questions about the data using the chat interface.
5. View generated SQL queries and query results.

## Contributors
- [Harsh joshi](https://github.com/harsh0506/text-2-sql-streamlit-finetune) - Developer

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Apache Spark](https://spark.apache.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Contact
For any inquiries or issues, please open an issue on the [GitHub repository](https://github.com/your-repo/issues).