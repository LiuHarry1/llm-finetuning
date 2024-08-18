from unsloth import UnslothModel, Trainer, UnslothConfig

model_name = "facebook/llama"  # Replace with the correct model path
model = UnslothModel.from_pretrained(model_name)


# Load dataset
dataset_path = "path/to/your/dataset.json"  # Adjust the path
train_dataset = UnslothModel.load_json(dataset_path)

# Configuration
config = UnslothConfig(
    model_name="facebook/llama",
    learning_rate=5e-5,
    num_train_epochs=3,
    batch_size=8
)

# Fine-tune the model
trainer = Trainer(model, config)
trainer.train(train_dataset)

trainer.save_model("path/to/save/fine-tuned-model")

fine_tuned_model = UnslothModel.from_pretrained("path/to/save/fine-tuned-model")

# Inference
output = fine_tuned_model.generate("Your input text")
print(output)
