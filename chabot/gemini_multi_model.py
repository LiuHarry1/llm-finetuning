from vertexai.preview import generative_models, init
from vertexai.preview.generative_models import GenerativeModel, Image
from typing import List, Union

class GeminiChatBot:
    def __init__(self, project_id: str, location: str = "us-central1"):
        init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-pro-vision")
        self.sessions = {}  # user_id: history list

    def _load_image(self, image_path: str) -> Image:
        return Image.load_from_file(image_path)

    def chat(
        self,
        user_id: str,
        prompt: str,
        image_path: str = None,
    ) -> str:
        if user_id not in self.sessions:
            self.sessions[user_id] = []

        history: List[Union[str, Image]] = self.sessions[user_id]

        # 构建输入
        if image_path:
            image = self._load_image(image_path)
            inputs = history + [prompt, image]
        else:
            inputs = history + [prompt]

        # 调用 Gemini 模型
        response = self.model.generate_content(inputs)

        # 更新历史
        history.append(prompt)
        if image_path:
            history.append(image)
        history.append(response.text)

        return response.text



def main():
    project_id = "your-google-cloud-project-id"
    bot = GeminiChatBot(project_id)

    user_id = "user123"

    # 第一次：上传图片并提问
    print("🔹 用户：这张图片里是什么？")
    response1 = bot.chat(user_id, "这张图片里是什么？", image_path="sample.jpg")
    print("🤖 Gemini：", response1)

    # 第二次：继续追问（上下文中包含上一轮问题和图片）
    print("\n🔹 用户：它还能正常工作吗？")
    response2 = bot.chat(user_id, "它还能正常工作吗？")
    print("🤖 Gemini：", response2)

if __name__ == "__main__":
    main()
