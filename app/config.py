from pydantic import BaseModel, validator
import os

class Config(BaseModel):
    file_path: str
    target_column: str
    task_type: str

    @validator("file_path")
    def check_file_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError("Provided file path does not exist")
        return v

    @validator("task_type")
    def validate_task(cls, v):
        if v not in ["classification", "regression", "clustering"]:
            raise ValueError("Task type must be classification, regression, or clustering")
        return v
