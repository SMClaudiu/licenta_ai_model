import pandas as pd
from sqlalchemy import create_engine , text

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseConnectionError


class DatabaseManager:
    def __init__(self , config: DatabaseConfig):
        self.config = config
        self.engine = ''

    def connect_to_db(self) -> bool:
        try:
            connection_string = f"postgresql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")

    def extract_data(self):
        query = """
                SELECT t.task_id,
                       t.creation_date,
                       t.description,
                       t.due_date,
                       t.name                                           as task_name,
                       t.status,
                       t.board_id,
                       b.name                                           as board_name,
                       b.dash_board_id,
                       c.client_id,
                       c.email,
                       c.name                                           as client_name,
                       c.phone_number,
                       EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400 as days_to_complete, EXTRACT(DOW FROM t.creation_date::timestamp) as creation_day_of_week,
                       EXTRACT(HOUR FROM t.creation_date::timestamp)    as creation_hour,
                       EXTRACT(MONTH FROM t.creation_date::timestamp)   as creation_month,
                       EXTRACT(QUARTER FROM t.creation_date::timestamp) as creation_quarter,
                       LENGTH(t.description)                            as description_length,
                       LENGTH(t.name)                                   as task_name_length,
                       COUNT(*)                                            OVER (PARTITION BY c.client_id) as client_total_tasks, COUNT(*) FILTER (WHERE t.status = 2) OVER (PARTITION BY c.client_id) as client_completed_tasks, COUNT(*) FILTER (WHERE t.status = 0) OVER (PARTITION BY c.client_id) as client_pending_tasks, AVG(EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400) OVER (PARTITION BY c.client_id) as client_avg_task_duration, COUNT(*) OVER (PARTITION BY t.board_id) as board_total_tasks, COUNT(*) FILTER (WHERE t.status = 2) OVER (PARTITION BY t.board_id) as board_completed_tasks, AVG(EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400) OVER (PARTITION BY t.board_id) as board_avg_task_duration
                FROM task t
                         JOIN board b ON t.board_id = b.board_id
                         JOIN dashboard d ON b.dash_board_id = d.id
                         JOIN client c ON d.client_id = c.client_id
                WHERE t.due_date IS NOT NULL
                  AND t.creation_date IS NOT NULL
                  AND EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp)) > 0
                ORDER BY t.creation_date
                """
        try:
            df = pd.read_sql_query(query , self.engine)
            print(f"Extracted {len(df)} records from database")
            df = df[df['days_to_complete'] <= 365]
            print(f"After filtering outliers: {len(df)} records")
            return df
        except Exception as e:
            print(f"Data extraction failed: {e}")
            return None