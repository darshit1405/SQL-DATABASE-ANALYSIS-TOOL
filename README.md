<<<<<<< HEAD
# SQL Database Analysis Tool (V1)

A beginner-friendly Streamlit app that connects to MySQL/PostgreSQL, reads schema, converts natural language to SQL using Together API + LangChain, executes read queries, shows results, charts, and exports.

## Features in first version

- Database connection (MySQL / PostgreSQL)
- Automatic schema detection
- Natural language to SQL generation
- SQL query review and execution
- Result table in Streamlit
- Basic charts (bar/line)
- Export results (CSV, JSON)

## Project structure

```text
sql-database-analysis-tool/
├── app.py
├── config.py
├── database.py
├── schema_utils.py
├── llm_sql_generator.py
├── query_executor.py
├── chart_utils.py
├── export_utils.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Create virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`
4. Fill your real database and Together API values in `.env`

## Run app

```bash
streamlit run app.py
```

## Notes

- First version allows only read queries (`SELECT` / `WITH`) for safety.
- Keep `.env` private and never commit secrets.
=======
# SQL-DATABASE-ANALYSIS-TOOL
>>>>>>> 2eb789286dbf52e3724368b34fb06c159b0318c5
