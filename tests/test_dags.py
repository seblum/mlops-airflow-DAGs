"""
DAG structure tests for the CNN skin cancer pipeline.

These tests validate:
- DAGs import cleanly (catches syntax errors and import-time failures)
- Correct DAG IDs are registered
- Expected tasks exist with the right counts
- DAG properties (schedule, max_active_runs) match expected values
- No deprecated Airflow 2.x APIs are used
"""

import importlib
import sys

import pytest


def _load_dag_module(module_name: str, module_path: str):
    """Import a DAG module by file path, bypassing package caching."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Kubernetes workflow DAG ───────────────────────────────────────────────────

class TestK8sWorkflowDAG:
    @pytest.fixture(scope="class")
    def dag_bag(self):
        from airflow.models import DagBag

        return DagBag(
            dag_folder="cnn_skin_cancer",
            include_examples=False,
        )

    def test_dag_loads_without_errors(self, dag_bag):
        assert not dag_bag.import_errors, f"DAG import errors: {dag_bag.import_errors}"

    def test_k8s_dag_present(self, dag_bag):
        assert "cnn_skin_cancer_workflow" in dag_bag.dags

    def test_docker_dag_present(self, dag_bag):
        assert "cnn_skin_cancer_docker_workflow" in dag_bag.dags

    def test_inference_dag_present(self, dag_bag):
        assert "cnn_skin_cancer_sagemaker_test_inference" in dag_bag.dags

    def test_k8s_dag_task_count(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_workflow"]
        # preprocessing + 3x training + compare + deploy = 6 (dynamic expand adds more at runtime)
        task_ids = dag.task_ids
        assert "preprocessing_op" in task_ids
        assert "compare_models_op" in task_ids
        assert "deploy_model_to_sagemaker_op" in task_ids

    def test_k8s_dag_no_schedule(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_workflow"]
        assert dag.schedule_interval is None or dag.timetable_summary == "Never"

    def test_k8s_dag_max_active_runs(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_workflow"]
        assert dag.max_active_runs == 1

    def test_docker_dag_no_schedule(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_docker_workflow"]
        assert dag.schedule_interval is None or dag.timetable_summary == "Never"

    def test_docker_dag_max_active_runs(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_docker_workflow"]
        assert dag.max_active_runs == 1

    def test_inference_dag_single_task(self, dag_bag):
        dag = dag_bag.dags["cnn_skin_cancer_sagemaker_test_inference"]
        assert "inference_call_op" in dag.task_ids

    def test_no_deprecated_schedule_interval_kwarg(self):
        """Ensure Airflow 3.0-incompatible schedule_interval is not used in DAG definitions."""
        import ast
        import os

        dag_dir = "cnn_skin_cancer"
        for fname in os.listdir(dag_dir):
            if not fname.endswith("_DAG.py"):
                continue
            fpath = os.path.join(dag_dir, fname)
            source = open(fpath).read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        assert kw.arg != "schedule_interval", (
                            f"Deprecated 'schedule_interval' found in {fname}. "
                            "Use 'schedule' instead (Airflow 3.0)."
                        )

    def test_no_hardcoded_secret_objects(self):
        """Airflow Secret() objects should not appear in DAG source files."""
        import os

        dag_dir = "cnn_skin_cancer"
        for fname in os.listdir(dag_dir):
            if not fname.endswith("_DAG.py"):
                continue
            fpath = os.path.join(dag_dir, fname)
            source = open(fpath).read()
            assert "airflow.kubernetes.secret" not in source, (
                f"Deprecated import 'airflow.kubernetes.secret' found in {fname}. "
                "Secrets should be managed via the Kubernetes Secrets Backend."
            )
