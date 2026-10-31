from airflow.sdk import dag, task
import time, pathlib, shutil

INBOX = pathlib.Path("/opt/airflow/project/inbox")
ROOT = pathlib.Path("/opt/airflow/project/")

def input_image():
    from preprocessing import SUPPORTED_EXTENSIONS
    files = [f for f in INBOX.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"Chua co anh trong {INBOX}")
    run = ROOT/ "outputs" / "runs" / time.strftime("%Y%m%d-%H%M%S")
    (run / "input").mkdir(parents=True)
    for f in files:
        shutil.move(f, run / "input" / f.name)
    return str(run)

def gallery(name, run, *dirs):
    """Ghi outputs/gallery/<name>.html: moi anh mot hang, moi thu muc (tuong doi voi run) mot cot."""
    out = ROOT / "outputs"
    run = pathlib.Path(run)
    html = ["<meta charset='utf-8'><style>img{width:300px;margin:2px}</style>", f"<h3>{run.name}</h3>"]
    for p in sorted((run / dirs[0]).glob("*.png")):
        html.append(f"<h4>{p.name}</h4>" + "".join(
            f"<img src='/outputs/{(run / d / p.name).relative_to(out).as_posix()}' title='{d}'>" for d in dirs))
    (out / "gallery").mkdir(exist_ok=True)
    (out / "gallery" / f"{name}.html").write_text("".join(html), encoding="utf-8")
    print(f"{len(html) - 2} anh -> /outputs/gallery/{name}.html")

@dag(
    dag_id = "image_processing_dag",
)
def image_processing_dag():

    @task.python
    def input_images(**kwargs):
        run = input_image()
        kwargs["ti"].xcom_push(key="run", value = run)
        return run

    @task.python
    def hsv(**kwargs):
        from preprocessing import process_dataset
        run = kwargs["ti"].xcom_pull(task_ids="input_images", key="run")
        process_dataset(pathlib.Path(run) / "input", pathlib.Path(run) / "pre", save_intermediate=True)
        return run

    @task.python
    def canny(**kwargs):
        import cv2
        from canny import canny_edges
        run = kwargs["ti"].xcom_pull(task_ids="hsv")
        src = pathlib.Path(run) / "pre" / "canny_input"
        dst = pathlib.Path(run) / "edges"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            img = cv2.imread(str(f), 0)
            edges = canny_edges(img, 50, 150, 1.4)
            cv2.imwrite(str(dst / f.name), edges)
        kwargs["ti"].xcom_push(key="run", value = run)

    @task.python
    def hough(**kwargs):
        import cv2
        from hough import detect_road_boundaries, draw_hough_visualization
        run = kwargs["ti"].xcom_pull(task_ids="canny", key="run")
        src = pathlib.Path(run) / "edges"
        bg = pathlib.Path(run) / "pre" / "resized"
        dst = pathlib.Path(run) / "hough"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.png"):
            edges = cv2.imread(str(f), 0)
            result = detect_road_boundaries(edges)
            # hough.py ve mau theo RGB, con cv2 doc/ghi BGR nen phai doi qua lai
            rgb = cv2.cvtColor(cv2.imread(str(bg / f.name)), cv2.COLOR_BGR2RGB)
            vis = draw_hough_visualization(rgb, result)
            cv2.imwrite(str(dst / f.name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"{f.name}: trai={result.left_line is not None}, phai={result.right_line is not None}")
        kwargs["ti"].xcom_push(key="run", value = run)

    @task.python
    def show_hsv(**kwargs):
        run = kwargs["ti"].xcom_pull(task_ids="input_images", key="run")
        gallery("show_hsv", run, "pre/resized", "pre/color_masks", "pre/canny_input")

    @task.python
    def show_canny(**kwargs):
        run = kwargs["ti"].xcom_pull(task_ids="input_images", key="run")
        gallery("show_canny", run, "pre/canny_input", "edges")

    @task.python
    def show_hough(**kwargs):
        run = kwargs["ti"].xcom_pull(task_ids="input_images", key="run")
        gallery("show_hough", run, "hough")

    t1 = input_images()
    t2 = hsv()
    t3 = canny()
    t4 = hough()

    t1 >> t2 >> t3 >> t4
    t2 >> show_hsv()
    t3 >> show_canny()
    t4 >> show_hough()
    
image_processing_dag()