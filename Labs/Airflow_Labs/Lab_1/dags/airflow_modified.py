from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab_modified import load_data, data_preprocessing, build_save_model, evaluate_clusters

# Define default arguments
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Modified DAG — uses Iris dataset + DBSCAN + new evaluate_clusters task
with DAG(
    'Airflow_Lab1_Modified',
    default_args=default_args,
    description='Modified Lab 1: Iris dataset + DBSCAN clustering + cluster evaluation',
    catchup=False,
) as dag:

    # Task 1 — Load Iris dataset (replaces CSV loading)
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task 2 — Preprocess data with MinMax scaling
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task 3 — Fit DBSCAN model (replaces K-Means, auto-detects clusters)
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model_modified.sav"],
    )

    # Task 4 — NEW: Evaluate cluster quality with silhouette score
    evaluate_clusters_task = PythonOperator(
        task_id='evaluate_clusters_task',
        python_callable=evaluate_clusters,
        op_args=[data_preprocessing_task.output, build_save_model_task.output],
    )

    # Pipeline
    load_data_task >> data_preprocessing_task >> build_save_model_task >> evaluate_clusters_task


if __name__ == "__main__":
    dag.test()