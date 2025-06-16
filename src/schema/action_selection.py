from pydantic import BaseModel, Field

class ActionSelection(BaseModel):
    intent: str = Field(
        description="Explanation of why this action was chosen and what goal it aims to achieve"
    )
    action: str = Field(
        description="The specific action to perform, must be one of the predefined formats like Click(x, y) or Key(\"key\")"
    )

    def __repr__(self):
        return self.model_dump_json(indent=4, ensure_ascii=False)

    @property
    def intent_len(self):
        return len(self.intent)
        