from airflow.plugins_manager import AirflowPlugin
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# ponytail: khong co xac thuc, ai vao duoc localhost:8080 la xem duoc /outputs; chi dung local
app = FastAPI()
app.mount("/", StaticFiles(directory="/opt/airflow/project/outputs", check_dir=False), name="outputs")


class OutputsPlugin(AirflowPlugin):
    name = "outputs"
    fastapi_apps = [{"app": app, "url_prefix": "/outputs", "name": "Outputs"}]
    external_views = [{
        "name": "Kết quả",
        "href": "/outputs/gallery/{TASK_ID}.html",
        "destination": "task_instance",
        "url_route": "ket-qua",
    }]
