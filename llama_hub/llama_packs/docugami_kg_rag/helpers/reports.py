import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
import sqlite3
import tempfile
from config import REPORT_DIRECTORY, DOCUGAMI_API_KEY, SQL_GEN_LLM
from prompts import EXPLAINED_QUERY_PROMPT

from docugami import Docugami
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine


HEADERS = {"Authorization": f"Bearer {DOCUGAMI_API_KEY}"}


def download_project_latest_xlsx(project_url: str, local_xlsx: Path) -> Optional[Path]:
    response = requests.get(
        f"{project_url}/artifacts/latest?name=spreadsheet.xlsx",
        headers=HEADERS,
    )

    if response.ok:
        response_json = response.json()["artifacts"]
        xlsx_artifact = next(
            (
                item
                for item in response_json
                if str(item["name"]).lower().endswith(".xlsx")
            ),
            None,
        )
        if xlsx_artifact:
            artifact_id = xlsx_artifact["id"]
            response = requests.get(
                f"{project_url}/artifacts/latest/{artifact_id}/content",
                headers=HEADERS,
            )
            if response.ok:
                os.makedirs(str(local_xlsx.parent), exist_ok=True)
                with open(local_xlsx, "wb") as f:
                    f.write(response.content)
                    return local_xlsx
            else:
                raise Exception(f"Failed to download XLSX for {project_url}")
    elif response.status_code == 404:
        return None  # No artifacts found: this project has never been published
    else:
        raise Exception(f"Failed to download XLSX for {project_url}")


def connect_to_excel(
    file_path: Union[Path, str], table_name: str, sample_rows_in_table_info=0
) -> SQLDatabase:
    conn = sqlite3.connect(":memory:")

    file_path = Path(file_path)
    if not (file_path.exists() and file_path.suffix.lower() == ".xlsx"):
        raise Exception(f"Invalid file path: {file_path}")

    df = pd.read_excel(file_path, sheet_name=0)

    df.to_sql(table_name, conn, if_exists="replace", index=False)

    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite")
    with sqlite3.connect(temp_db_file.name) as disk_conn:
        conn.backup(disk_conn)  # dumps the connection to disk

    return SQLDatabase.from_uri(
        f"sqlite:///{temp_db_file.name}",
        sample_rows_in_table_info=sample_rows_in_table_info,
    )


def get_sql_query_engine(docset_id):
    docugami_client = Docugami()
    projects_response = docugami_client.projects.list()
    projects = [p for p in projects_response.projects if p.docset.id == docset_id]

    project = projects[0]
    report_path = Path(REPORT_DIRECTORY) / f"{project.id}.xlsx"

    local_xlsx_path = download_project_latest_xlsx(
        project.url, Path(REPORT_DIRECTORY) / f"{project.id}.xlsx"
    )

    if not local_xlsx_path:
        raise Exception("Failed to download the latest report")

    report_name = project.name or report_path

    sql_database = connect_to_excel(report_path, report_name)

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[report_name],
        llm=SQL_GEN_LLM,
    )

    sql_query_engine.update_prompts({"prompt": EXPLAINED_QUERY_PROMPT})

    return sql_query_engine
