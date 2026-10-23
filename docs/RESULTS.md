# Results

All numbers come from the real run `runs/overnight_long` (config `configs/small.toml`, `--max-steps 9500`, git commit
recorded in the run folder). Nothing here is rounded in our favour.

## The model and the run
| Item | Value |
|---|---|
| Architecture | GPT-2 style, pre-norm, 6 layers, d_model 256, 8 heads, context 256, tied embeddings |
| Parameters | 5,853,184 (4,739,072 without embeddings) |
| Tokenizer | byte-level BPE, vocab 4096, trained on the training split only |
| Data | TinyStories V2 (GPT-4), first 300k stories = 60.8M tokens; 5k validation stories = 1.0M tokens |
| Training | 9,500 steps x 8,192 tokens = 77.8M tokens, about 1.28 passes over the data (the last 28% repeats once) |
| Optimiser | AdamW, peak lr 1e-3, 200 warmup steps, cosine to 1e-4, weight decay 0.1 on matrices, clip 1.0 |
| Hardware | Intel Core Ultra 9 185H laptop CPU, 12 threads, no GPU |
| Training time | 28,388 s = 7.9 h of running time (the laptop also slept for about 10 h mid-run, which is not counted) |
| Throughput | about 2,740 tokens/s on average |

## Quality
| Model | Val loss (nats) | Val perplexity |
|---|---|---|
| Uniform guess | 8.32 | 4096 |
| Unigram counts | 5.88 | 357 |
| Bigram counts (add-one) | 3.65 | 38.4 |
| **Tiny GPT (this run)** | **1.688** | **5.41** |

The GPT is about 7 times lower in perplexity than the bigram baseline, so it uses far more than the previous token.
Final number is the mean over 200 fixed validation batches (32 x 256 tokens each). The in-training value at step 9,500
used 20 batches and read 1.708; the 200-batch figure is the more reliable one. The best checkpoint was the final step, so
choosing it on validation loss did not leak anything into the number.

![loss curve](img/loss_curve.png)

Train and validation fell together for the whole run. Validation stayed below training loss (training loss is averaged
over each logging window and includes dropout), and it kept improving through the repeated data. There was no overfitting
within this budget, and no sign of instability: the curve is smooth, with no spikes.

## Speed
Generating 100 tokens on the laptop CPU takes a median of **0.51 s** (5 runs, range
0.49 to 0.61 s) with no KV cache. Generation latency on the Hugging Face Space
will be added after Phase 8.

## Samples
Same checkpoint (`best`), temperature 0.8, top-k 40, fixed seeds, 80 new tokens. These are not cherry-picked: the first
generation for each prompt is shown, including the bad ones.

### Five fixed prompts (the same ones logged in `samples.md` during training)
- **Once upon a time** (seed 0, length) -> , in a small town, there was a little boy named Tim. Tim had a big, red car. He loved to play with his car every day. He would push the car up and down the hill. Tim's car became very good at driving, and he was very happy. / One day, Tim's toy car was driving down the hill. The car stopped and went...
- **One day, a little girl named** (seed 1, length) -> Lucy went to play outside. She saw a big tree with a pretty flower. She thought it would be fun to draw with it. So, she put the flower on the tree and started to draw a pretty picture. She was very proud of her drawing. / But then, something unexpected happened. The flower began to grow bigger and ...
- **The dog was very** (seed 2, length) -> good at finding things. He saw a big ball and wanted to play with it. He ran to the ball and tried to catch it. But the ball did not move. It moved and made the ball move. The dog was sad. He wanted his ball back. / The dog tried to bite the big ball. But, the ball was too big for him. The dog was s...
- **Tom and his mom went to the** (seed 3, length) -> store. They wanted to buy money. Mom said they could buy money for the toy. Tom was very happy. / At the store, they saw many money. Tom liked money. He asked mom if they could buy money for the money to buy the store for the toy. Mom said yes, and they bought the store for the expensive toy. / Tom ...
- **The sun was shining and** (seed 4, stop) -> he knew that he had to wait.

### Five prompts never used during training
- **The little robot wanted to** (seed 0, length) -> help the little robot. So, they went to the park together. The little robot helped the little boy get off the slide. They took turns sliding down together. The little boy and the robot were very happy. / At the end of the day, the robot and the little boy were tired. They sat on the slide and shared...
- **Mia found a shiny key under her bed.** (seed 1, length) -> She used the key to open the door. Inside the door, she found a lot of shiny key. Mia was very happy. She put the key in the lock and went outside to play. / Later, Mia's friend, Tom, came over to play. He saw the key and knew it was an ordinary key. He took the key and opened the lock with the key....
- **It was raining, so the two friends** (seed 2, length) -> wanted to go to the beach. As they got closer, the sun started to set, and they had a lot of fun. / Suddenly, they realised they had to go to the beach. He said goodbye to it and went to the beach. He was so excited to go on his fun adventure, and soon he was all around the ocean. / But when they ar...
- **A tiny dragon lived in** (seed 3, length) -> a big, green forest. The dragon had many friends, like birds, monkeys, and squirrels. They all played together in the forest all day. / One day, a mean wolf came to the forest. He wanted to destroy the forest. The wolf saw the little bird and wanted to help. They worked together to destroy the fores...
- **Dad said, "Let's bake a** (seed 4, length) -> cake. We can make flour and eggs." / "Let's do it!" Anna said. / They were excited to start. They got on the floor and mixed the flour and the flour. They mixed until they added and poured the mix. Then they poured the mix into two cups. They put the mix and sugar in a bowl. They waited for dad to b...

### What the samples show
- **Strong:** grammar, names, simple plots, quotation marks and paragraphing, and consistent characters within a paragraph.
- **Weak:** repetition loops (the dog and the ball, "buy money" -> "buy the store for the money"), objects that lose their meaning
  ("a lot of shiny key"), and plots that drift or contradict themselves ("The wolf wanted to destroy the forest... The wolf was happy again").
  One prompt ended almost immediately because the model emitted `<|endoftext|>` after a single sentence.
- This is what a 5.8M-parameter model trained for about 8 hours on a CPU can do. It is not a general assistant, and it only knows
  the TinyStories register.

## Temperature and top-k, side by side
Prompt "One day, a little girl named", seed 5, 60 new tokens. Only the sampling settings change.

| Setting | Completion |
|---|---|
| temp 0 (greedy) | Lily went to the park with her mom. They saw a big tree with a lot of leaves. Lily wanted to pick some leaves, but her mom said, "No, Lily. We need to clean up the leaves." / Lily was sad, but she listened to her mom. She picked up... |
| temp 0.2, top-k 40 | Lily went to the park with her mom. They saw a big tree with a lot of leaves. Lily wanted to pick some leaves, but her mom said, "No, Lily. We need to clean up the leaves." / Lily was sad, but she wanted to help her mom. She looked... |
| temp 0.8, top-k 40 | Lily found a pretty dress in her room. She had a shiny dress that could tie. She loved her dress very much. She put it on and went outside to play. The sun was shining, and the birds were singing. / Lily saw her friend, Tom, playing... |
| temp 1.5, top-k 40 | Sue found a toy with a zipper. It was an incredible shiny toy that could play with. It could talk or do fun things, sometimes people could talk. So, she took a bag to a fun party with her mom and dad. They said "reguuate to Sue an... |
| temp 2.5, top-k 40 | Minngera lived in Lily. J organized She had two toys at nightmiling. It looked like red to look on the beds. "Hibbers for today." Tim liked to play with Lily on her sidework where she kept them there all there. / So he looked... |
| temp 1.5, no top-k | Sue found a crown. It looked helpless. She had coat and pedalsIn her mom loved her lemon Bobo to taste. He cooked the food from one cheap from her smell to lose. / Sue thought it would be fun to buy the dig piece of chocolate or mor... |
| temp 0.8, top-k 3 | Lily found a pretty flower in her garden. She was very happy and wanted to show her mom the pretty flower. She picked it up and looked at it. It was very pretty. Lily wanted to show her mom. / "Look, mom! I found a pretty flower!" s... |

Low temperature repeats the most likely story (greedy and 0.2 are nearly identical). Around 0.8 the story varies and stays
fluent. At 1.5 invented words appear, and at 2.5 it is mostly noise. Removing top-k at 1.5 is worse than keeping it, because
top-k cuts off the long tail of very unlikely tokens before sampling.

## Reproduce
```bash
make data                                   # about 1 minute
make train CONFIG=configs/small.toml        # or the --max-steps 9500 run above
uv run tinygpt baselines --config configs/small.toml
uv run tinygpt generate --run runs/overnight_long --prompt "Once upon a time" --seed 1
```
