import logging
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

from mindscope.pipeline import evaluate_model, prepare_data, preprocess, train_model

log = logging.getLogger(__name__)


def notify_failure(context):
    task_instance = context["task_instance"]
    log.error(
        "PIPELINE ERROR - task '%s' failed (try %s): %s",
        task_instance.task_id,
        task_instance.try_number,
        context.get("exception"),
    )


default_args = {
    "owner": "mindscope",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "on_failure_callback": notify_failure,
}

with DAG(
    dag_id="student_risk_pipeline",
    description="Prepare the data, preprocess, train and evaluate the student mental health risk model",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args=default_args,
    tags=["mindscope"],
) as dag:
    prepare_task = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    prepare_task >> preprocess_task >> train_task >> evaluate_task