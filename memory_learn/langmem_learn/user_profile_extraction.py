import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langmem import create_memory_manager
from pydantic import BaseModel
from typing import Optional

load_dotenv()
llm = ChatTongyi(model="qwen-max", api_key=os.getenv("TONGYI_API_KEY"))

# Define profile structure
class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


# Configure extraction
manager = create_memory_manager(
    llm,
    schemas=[UserProfile], # (optional) customize schema
    instructions="Extract user profile information",
    enable_inserts=False,  # Profiles update in-place
)

# First conversation
conversation1 = [{"role": "user", "content": "I'm Alice from California"}]
memories = manager.invoke({"messages": conversation1})
print(memories[0])
# ExtractedMemory(id='profile-1', content=UserProfile(
#    name='Alice',
#    language=None,
#    timezone='America/Los_Angeles'
# ))

# Second conversation updates existing profile
conversation2 = [{"role": "user", "content": "I speak Spanish too!"}]
update = manager.invoke({"messages": conversation2, "existing": memories})
print(update[0])
# ExtractedMemory(id='profile-1', content=UserProfile(
#    name='Alice',
#    language='Spanish',  # Updated
#    timezone='America/Los_Angeles'
# ))