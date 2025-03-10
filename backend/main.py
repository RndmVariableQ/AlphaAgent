import logging
import multiprocessing
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uuid
from datetime import datetime
from alphaagent.app.qlib_rd_loop.factor_mining import main as mine

# 配置 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将共享资源的初始化移到startup事件中
manager = None
tasks = None
task_lock = None

@app.on_event("startup")
def startup_event():
    global manager, tasks, task_lock
    manager = multiprocessing.Manager()
    tasks = manager.dict()
    task_lock = multiprocessing.Lock()
    logger.info("Shared resources initialized")

class TaskRequest(BaseModel):
    direction: str

class TaskStatus(BaseModel):
    task_id: str
    direction: str
    status: str
    progress: float
    created_at: str
    updated_at: str
    result: Optional[str] = None
    error: Optional[str] = None

def mine_wrapper(direction, stop_event, task_id):
    # 子进程独立配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"mine_wrapper 开始执行，task_id={task_id}")
    try:
        result = mine(direction=direction, stop_event=stop_event)
        logger.info(f"mine_wrapper 执行完成，结果：{result}")
        with task_lock:
            if task_id in tasks and not tasks[task_id].get("stop", False):
                task = tasks[task_id]
                task.update({
                    "status": "completed",
                    "progress": 100.0,
                    "result": str(result),
                    "updated_at": datetime.now().isoformat()
                })
                tasks[task_id] = task
    except Exception as e:
        logger.error(f"mine_wrapper 执行出错：{e}", exc_info=True)
        with task_lock:
            if task_id in tasks:
                task = tasks[task_id]
                task.update({
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.now().isoformat()
                })
                tasks[task_id] = task

def run_mine(task_id: str, direction: str):
    try:
        with task_lock:
            if task_id not in tasks:
                return
            task = tasks[task_id]
            task.update({
                "status": "running",
                "progress": 0.0,
                "updated_at": datetime.now().isoformat()
            })
            tasks[task_id] = task

        stop_event = multiprocessing.Event()

        p = multiprocessing.Process(
            target=mine_wrapper,
            args=(direction, stop_event, task_id)
        )
        p.start()
        logger.info(f"Process started: task_id={task_id}, pid={p.pid}")

        with task_lock:
            task = tasks[task_id]
            task["process_pid"] = p.pid
            tasks[task_id] = task

        while p.is_alive():
            with task_lock:
                task = tasks[task_id]
                if task.get("stop", False):
                    stop_event.set()
                    task.update({
                        "status": "stopping",
                        "updated_at": datetime.now().isoformat()
                    })
                    tasks[task_id] = task
                    break
            p.join(timeout=0.5)

        with task_lock:
            task = tasks[task_id]
            if task.get("stop", False):
                task.update({
                    "status": "stopped",
                    "progress": max(task.get("progress", 0.0), 0.0),
                    "updated_at": datetime.now().isoformat(),
                    "stop": False
                })
                tasks[task_id] = task
    except Exception as e:
        logger.error(f"run_mine error: {e}", exc_info=True)
        with task_lock:
            if task_id in tasks:
                task = tasks[task_id]
                task.update({
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.now().isoformat()
                })
                tasks[task_id] = task
    finally:
        with task_lock:
            if task_id in tasks:
                task = tasks[task_id]
                task["stop"] = False
                tasks[task_id] = task

@app.post("/api/tasks", response_model=TaskStatus)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    task_data = {
        "task_id": task_id,
        "direction": request.direction,
        "status": "pending",
        "progress": 0.0,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
        "stop": False,
        "process_pid": None
    }
    
    with task_lock:
        tasks[task_id] = manager.dict(task_data)
    
    background_tasks.add_task(run_mine, task_id, request.direction)
    return dict(tasks[task_id])

@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str):
    with task_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return dict(tasks[task_id])

@app.get("/api/tasks")
def list_tasks():
    with task_lock:
        return [dict(task) for task in tasks.values()]

@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str):
    with task_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        task = tasks[task_id]
        task["stop"] = True
        task["updated_at"] = datetime.now().isoformat()
    return {"message": "Stop signal sent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)