from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=120)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# 自动验证数据
try:
    user = User(name="Jo", age="1", email="test@example.com")
    print(user)  # ✅ 验证通过

    print(user.model_dump_json())
except ValidationError as e:
    print(e)     # ❌ 如果数据不符合规则会抛出异常


from dataclasses import dataclass, field
from typing import Optional

# ✅ 使用 @dataclass 而不是 BaseModel
@dataclass
class User1:
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=120)
    active: bool = field(default=True)

try:
    user = User1(name="", age="a", active="test@example.com")
    print(user)  # ✅ 验证通过
except ValidationError as e:
    print(e)     # ❌ 如果数据不符合规则会抛出异常


from pydantic import BaseModel
from typing import List

# ❌ 错误做法：对可变对象使用 default
class UserBad(BaseModel):
    tags: List[str] = []  # 这样所有实例会共享同一个列表！

user1 = UserBad()
user2 = UserBad()

user1.tags.append("vip")
print(user2.tags)  # 输出: ['vip'] 😱 user2 也被影响了！