import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langmem import create_memory_manager
from pydantic import BaseModel

load_dotenv()
llm = ChatTongyi(model="qwen-max", api_key=os.getenv("TONGYI_API_KEY"))

class Triple(BaseModel):
    """Store all new facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# Configure extraction
manager = create_memory_manager(
    llm,
    schemas=[Triple],
    instructions="Extract user preferences and any other useful information",
    enable_inserts=True,
    enable_deletes=True,
)

# First conversation - extract triples
conversation1 = [
    {"role": "user", "content": "Alice manages the ML team and mentors Bob, who is also on the team."}
]
memories = manager.invoke({"messages": conversation1})
print("After first conversation:")
for m in memories:
    print(m)
# After first conversation:
# ExtractedMemory(id='a0ad630d-798d-4806-b5b1-daf18d386855', content=Triple(subject='Alice', predicate='is_managed_by', object='Bob', context='Session b2eff1b8-61b5-4905-97ca-f0c7d1382c66'))

conversation2 = [
    {"role": "user", "content": "Bob now leads the ML team and the NLP project."}
]
update = manager.invoke({"messages": conversation2, "existing": memories})
print("After second conversation:")
for m in update:
    print(m)

existing = [m for m in update if isinstance(m.content, Triple)]
print("After all conversations:")
for m in existing:
    print(m)


# Delete triples about an entity
conversation3 = [
    {"role": "user", "content": "Alice left the company."}
]
final = manager.invoke({"messages": conversation3, "existing": existing})
print("After third conversation:")
for m in final:
    print(m)

# ExtractedMemory(id='7ca76217-66a4-4041-ba3d-46a03ea58c1b', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='35b443c7-49e2-4007-8624-f1d6bcb6dc69', content=RemoveDoc(json_doc_id='0214f151-b0c5-40c4-b621-db36b845956c'))
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))