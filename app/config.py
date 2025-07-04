from pydantic import BaseModel, field_validator
import os

class Config(BaseModel):
    file_path: str
    target_column: str
    task_type: str

    @field_validator("file_path")
    @classmethod
    def check_file_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError("Provided file path does not exist")
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task(cls, v):
        if v not in ["classification", "regression", "clustering"]:
            raise ValueError("Task type must be classification, regression, or clustering")
        return v