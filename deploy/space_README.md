---
title: Tiny GPT
emoji: 📖
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A 5.8M-parameter GPT trained from scratch on children's stories
---

# Tiny GPT demo

Type the start of a children's story and a 5.8M-parameter GPT, written from scratch in PyTorch and trained on a laptop CPU,
continues it. This Space runs the FastAPI service from the code repository on a small CPU.

- **Model and card:** the Hub model repo named in this Space's `TINYGPT_HF_REPO` variable
- **API docs:** `/docs` on this Space
- **Cold start:** free Spaces sleep when idle; the first request after a wake-up waits while the container starts and the weights download.
- **Limits:** the model only knows simple TinyStories-style English. It repeats itself and loses the plot. It is not an assistant.
