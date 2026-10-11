# tinyclass

JEV-like classification from the terminal.

```bash
tinyclass decide "Payroll asks for your password on a non-company sign-in page." \
           Legitimate Spam Phishing
#   Legitimate    1.7%  █
#   Spam          0.9%
#   Phishing     97.4%  █████████████████████████████
# → Phishing
```

## Installing

With [mise](https://mise.jdx.dev):

```bash
mise use -g github:monorkin/tinyclass                      # install it and put tinyclass on your PATH
mise exec github:monorkin/tinyclass -- tinyclass --help    # or run it once without installing
```

On [Omarchy](https://omarchy.org):

```bash
omarchy-mise-install github:monorkin/tinyclass tinyclass
```

From source:

```bash
cargo install --path .
```

Then, either way:

```bash
tinyclass shell-completion install zsh
```

## Usage

Every command takes `--help`; the top level also takes `-V/--version`.

```
tinyclass decide INPUT CHOICE [--instruction TEXT] [--device DEV] [--criteria PAIRS] [--json]

tinyclass noul STATEMENT [--instruction TEXT] [--device DEV] [--json]

tinyclass score STATEMENT [--instruction TEXT] [--json]

tinyclass batch STATE [--choice PAIRS] [--criteria PAIRS] [--noul TEXT] [--score TEXT] [--json]

tinyclass play CHOICE [--instruction TEXT] [--device DEV]

tinyclass model list - list the models tinyclass knows, marking the set one and the pulled ones
tinyclass model set NAME - make a model the one that answers
tinyclass model pull [NAME] - fetch the set model, or a named one, from Hugging Face
tinyclass model backend get|set - switch between the llama-cpp and ollama backends

tinyclass host set URL - configure the Ollama API URL
tinyclass host clear - remove the configured URL, reverting to the default
tinyclass host models - query the server and list available models
tinyclass host pull [NAME] - pull a model onto the server

tinyclass backend get BACKEND - show the backend (`llama-cpp` | `ollama`)
tinyclass backend set BACKEND - switch the backend (`llama-cpp` | `ollama`)

tinyclass device list - list the devices llama.cpp sees, with the index `gpu:N` refers to
tinyclass device set DEVICE - make a device the one the model runs on (`auto` | `cpu` | `gpu` | `gpu:N`)

tinyclass upgrade [VERSION] - move a mise install to the latest release, or a specific one

tinyclass shell-completion print SHELL - write the completion script to stdout
tinyclass shell-completion install SHELL - write it where the shell looks for it
```

Models live under `~/.local/share/tinyclass/models`, or `$XDG_DATA_HOME/tinyclass`.

`qwen3-0.6b` is enough for `decide` with clear-cut choices, but it leans hard
toward whichever option is listed first and says yes to nearly everything, so
`noul` needs at least `qwen3-1.7b`.

## Examples

### decide

```bash
tinyclass decide "Payroll asks for your password on a non-company sign-in page." \
                 Legitimate Spam Phishing
#   Legitimate    1.7%  █
#   Spam          0.9%
#   Phishing     97.4%  █████████████████████████████
# → Phishing
```

With a custom instruction and machine-readable output:

```bash
tinyclass decide "Best purchase ever" Positive Negative \
                 --instruction "Classify the sentiment of the review." --json
# {
#   "chosen": "Positive",
#   "scores": [
#     { "choice": "Positive", "logit": 36.61, "logprob": -0.0, "probability": 0.99998 },
#     { "choice": "Negative", "logit": 25.53, "logprob": -11.08, "probability": 0.000015 }
#   ]
# }
```

### noul

```bash
tinyclass noul "The moon is made of cheese."
# 0.125
```

### score

```bash
tinyclass score "The checkout service returned 500 errors for 30% of users." \
                --instruction "Rate the severity of this incident."
# 545.200
#   Routine            8.1%
#   Urgent            61.3%
#   Critical          30.6%
```

### batch

```bash
tinyclass batch "Customer email: charged twice, refund has not landed." \
                --choice billing|technical|frustrated \
                --noul "Is this customer about to cancel their subscription?" \
                --score "How urgently should a human pick this up?"
#   label: billing
#   harm: 0.312
#   urgency: 620.4
```

### play

```bash
tinyclass play Positive Negative   # keeps the model loaded, decides per line typed
# Loading qwen3-1.7b
# Deciding between Positive, Negative for every line; Ctrl-D ends.
# > This is great!
#   Positive  100.0%  ██████████████████████████████
#   Negative    0.0%
# → Positive
# > This sucks!
#   Positive    0.0%
#   Negative  100.0%  ██████████████████████████████
# → Negative
# > The moon is made of cheese
#   Positive   54.9%  ████████████████
#   Negative   45.1%  ██████████████
# → Positive
```

## Acknowledgements

The idea from [Jev in 25 lines of Python](https://www.nobodywho.ai/posts/jev-in-25-lines/): ask a chat model a multiple-choice question, stop after the prompt, and read the logits of the
answer letters instead of sampling. The softmax over just those
letters is the decision, with a probability for each choice.

Inference is done via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Building

```bash
mise trust
mise install
cargo build --release # → target/release/tinyclass
```

Always build and run with `--release`. Without it, the inference library takes about 200x longer
per given answer.

When developing use `cargo run --release -- decide ` instead of plain `cargo run`.

llama.cpp is compiled in, so the build needs cmake and a C++ compiler. On
Linux the Vulkan backend also needs the Vulkan and SPIR-V headers and
`glslc`: on Arch that's `cmake clang vulkan-headers spirv-headers shaderc`,
on Debian and Ubuntu `cmake g++ libvulkan-dev spirv-headers glslc`. On
macOS the Xcode command line tools and `brew install cmake` are enough;
Metal is the backend there. `--features cuda` or `rocm` add those backends
on Linux.

## Benchmark

`script/benchmark` times every pulled model on the phishing example above and
prints a table like the ones below. Load is what a one-off `decide` pays
before it answers; `play` pays it once.

### Ollama (remote)

```
Ollama server: http://localhost:11434
20 answers of "Payroll asks for your password on a non-company sign-in page." between Legitimate Spam Phishing

| Model              | Load   | Per answer | Answers/s |
|--------------------|--------|------------|-----------|
| tev1:4b            | 0.04 s | 130 ms     | 7.7       |
| tev1-4b-cpu:latest | 0.02 s | 242 ms     | 4.1       |
| nimble:latest      | 0.02 s | 131 ms     | 7.6       |
```

### Local (llama.cpp)

`DEVICE=cpu script/benchmark` picks
the device the same way `--device` does.

```
CPU: AMD Ryzen 9 9950X3D 16-Core Processor
Device: auto
Running on AMD Radeon RX 7900 XTX (RADV NAVI31) (Vulkan).
20 answers of "Payroll asks for your password on a non-company sign-in page." between Legitimate Spam Phishing

| Model      | Load   | Per answer | Answers/s |
|------------|--------|------------|-----------|
| qwen3-0.6b | 0.47 s | 14 ms      | 70.3      |
| qwen3-1.7b | 0.64 s | 17 ms      | 58.0      |
```

```
Device: cpu
Running on CPU.

| Model      | Load   | Per answer | Answers/s |
|------------|--------|------------|-----------|
| qwen3-0.6b | 0.41 s | 35 ms      | 28.6      |
| qwen3-1.7b | 0.47 s | 85 ms      | 11.8      |
```

Most of the load time is the Vulkan driver coming up, about 0.35 s here,
which is paid even when the CPU is chosen.

## License

tinyclass is released under the MIT License, see [LICENSE](LICENSE) for details.

It ships with [llama.cpp](https://github.com/ggml-org/llama.cpp) compiled
in, which is MIT licensed, copyright (c) 2023-2026 The ggml authors. The
Qwen3 models it downloads are Alibaba's, under the
[Apache License 2.0](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE).
