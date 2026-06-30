<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "合成数据"
prev-url: "12-synthetic-data"
page-title: 工具使用与函数调用
search-title: "第 13 章：工具使用与函数调用"
meta-description: "将工具使用和函数调用作为后训练目标，用于构建能力更强的语言模型产品和智能体。"
next-chapter: "过度优化"
next-url: "14-over-optimization"
---

# 工具使用与函数调用

让语言模型使用工具，是扩展其能力的自然方式。对于高精度任务尤其如此：外部工具可能包含所需信息；对于需要与复杂 Web 系统交互的智能体也是如此。
工具使用是语言模型需要通过训练获得的一项技能，RLHF 以及本书介绍的所有其他方法都可以对它进行改进。
考虑用户提出的这样一个问题：

> **User**: Who is the president today?

没有工具的语言模型会很难回答这个问题，因为预训练数据存在知识截止日期；但通过一次搜索查询，这类信息很容易获得。
再考虑另一个例子：

> **User**: Move all the arXiv papers in my downloads folder to my ~/research/ directory with names indicating the date of the paper.

这是仅凭模型权重根本无法尝试的任务；工具使用让语言模型能够处理范围广得多的任务。

在深入之前，有必要区分几个经常被混用的相关术语：

- **工具使用（Tool use）**：模型发出结构化请求（工具名和参数）；编排器执行工具；结果被追加到上下文中；模型继续生成。
- **函数调用（Function calling）**：一种工具使用形式，其中参数必须符合一组函数声明的 schema（通常是 JSON Schema），从而支持可靠解析和验证。
- **代码执行（Code execution）**：工具使用的特殊情形，其中“工具”是代码解释器（例如 Python）；结果作为工具输出返回。

## 工具使用概览

AI 模型通过输出特殊 token 来触发某个 endpoint，从而使用外部工具。
这些工具可以是高度专用的，例如返回某地天气的函数；也可以是代码解释器或搜索引擎这类复杂行为的基础构件。
第一个例子展示了语言模型需要更新信息来弥补其权重由过去数据训练而成这一固定性质；但也有代码执行这样的工具，可以让语言模型绕开其概率式、生成式本质，返回精确答案。
考虑打印圆周率近似值到 50 位的任务（不要靠记忆背诵，以免产生幻觉）。
带工具的语言模型可以这样做：

```text
<code>
from decimal import Decimal, getcontext
getcontext().prec = 60

def compute_pi():
    # Chudnovsky algorithm for computing pi
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L, S = 0, 1, 1, 13591409, Decimal(13591409)
    for i in range(1, 100):
        M = M * (K**3 - 16*K) // ((i)**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
    return C / S

print(str(compute_pi())[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>
```

本章概述现代语言模型中工具使用的起源、基础和格式，以及当前在领先模型中良好利用工具时面对的取舍。

“tool use”这个术语的确切起源并不清楚，但其思想远早于 RLHF 大规模扩散的后 ChatGPT 世界。
2015 年前后的早期例子曾尝试构建早于现代语言模型的系统，例如 Neural Programmer-Interpreters (NPI) [@reed2015neural]，它是“一种循环且组合式的神经网络，学习表示并执行程序”。
随着语言模型变得更流行，许多子领域开始通过集成外部能力来提升性能。
为了获取权重之外的信息，很多方法使用 retrieval augmented generation [@lewis2020retrieval] 或 Web 浏览 [@nakano2021webgpt]。
不久之后，也有人探索将语言模型与程序 [@gao2023pal] 或工具 [@parisi2022talm] 集成。

随着领域成熟，这些模型除了底层语言建模能力大幅提升之外，也获得了更复杂的能力。
例如，Toolformer 可以使用 “a calculator, a Q&A system, two different search engines, a translation system, and a calendar” [@schick2023toolformerlanguagemodelsteach]。
不久之后，Gorilla 被训练来使用 1645 个 API（来自 PyTorch Hub、TensorFlow Hub v2 和 Hugging Face），其评估 APIBench 也成为流行的 Berkeley Function Calling Leaderboard 的基础 [@patil2023gorilla]。
自这些早期模型以来，可调用动作的多样性已经显著增长。

工具使用模型如今已经与常规语言模型交互深度交织。
Model Context Protocol (MCP) 作为一种常见格式出现，用于把语言模型连接到外部数据源（或工具）[@anthropic_mcp_2024]。
随着模型更强、格式更好，工具使用型语言模型被用于许多场景，包括 Microsoft Office 或 Google Workspace 等流行应用中的生产力 copilots、科学领域 [@bran2023chemcrow]、医疗领域 [@li2024mmedagent]、Claude Code 或 Cursor 等编码智能体 [@zhang2024codeagent]、数据库集成，以及许多其他自主工作流。

评估工具使用模型涉及多个维度：工具名称和参数正确性的 exact-match 指标、schema 有效性，以及在模拟环境中的端到端任务完成。
跨试验的可靠性也很重要：$\tau$-bench 引入了 pass^k 指标（不同于 pass@k），用于衡量智能体是否稳定成功，而不是偶尔成功 [@yao2024taubench]。
ToolLLM 及其 ToolBench 数据集提供了一个大规模框架，用于在 16,000+ 真实 API 上训练和评估工具使用 [@qin2023toollm]；Berkeley Function Calling Leaderboard (BFCL) 仍是比较模型函数调用准确率的流行 benchmark [@patil2023gorilla]。

## 在生成中交织工具调用

函数调用的训练数据很像其他后训练数据，只多了一项：system prompt 会指示模型有哪些可用工具。
下面展示了一个带 system prompt，并以 JSON 格式提供可用工具的格式化数据点示例：
```xml
<system>
You are a function-calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
</system>

<functions>
[
  {
    "name": "search_movies",
    "description": "Search for movies by title and return matching results with IDs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search string for the movie title."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "get_movie_details",
    "description": "Fetch detailed information about a movie including cast, runtime, and synopsis.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        }
      },
      "required": ["movie_id"]
    }
  },
  {
    "name": "get_showtimes",
    "description": "Get movie showtimes for a given location and date.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        },
        "zip_code": {
          "type": "string",
          "description": "ZIP code for theater location."
        },
        "date": {
          "type": "string",
          "description": "Date for showtimes in YYYY-MM-DD format."
        }
      },
      "required": ["movie_id", "zip_code"]
    }
  }
]
</functions>

<user>
...
</user>
```
虽然语言模型是在生成一个补全，但如果它遵循这个示例，就会生成 token `search_movies("Star Wars")` 来搜索 Star Wars。
这通常会编码在特殊格式 token 中，随后插入序列的下一个 token 将包含工具输出。
借助这一机制，模型能够完成许多简单独立模型无法完成的更有挑战性的任务。

一种流行的工具使用形式是代码执行，它允许模型为复杂逻辑或数学问题获得精确答案。
例如，在语言模型执行过程中，代码执行可以发生在推理模型的 thinking tokens 内。
与函数调用一样，先有要执行代码的标签（由模型生成），然后有单独的输出标签。
```text
<|user|>
What is the 50th Fibonacci number? (Use the standard F_0=0, F_1=1 indexing.)</s>
<|assistant|>
<think>
Okay, I will compute the 50th Fibonacci number with a simple loop, then return the result.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
The 50th Fibonacci number is 12 586 269 025.
</answer>
```

底层发生的是：语言模型将工具输入和输出与标准自回归生成 token 交织在一起。
使这成为可能的编排循环大致如下：

```python
messages = [...]
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        result = execute_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

![工具使用将模型生成与外部执行交织起来：模型生成 token，直到发出工具调用（橙色）；外部系统执行工具并把输出（紫色）注入序列；随后模型继续生成。模型可以在一次生成中发出多个工具调用。训练期间，工具调用和输出 token 通常会从损失中 mask 掉。](images/tool_use_generation.png){#fig:tool-use-generation data-dark-src="images/tool_use_generation-dark.png"}

工具使用训练的目标，是让模型在这种不同的 token 流中表现得可预测：知道何时发出工具调用，如何正确格式化参数，以及如何把结果纳入回复。
开放模型必须经过训练，才能适配用户可能现成连接的各种工具。

## 多步工具推理

OpenAI 的 o3 模型代表了多步工具使用与语言模型集成方式的一次重要跃迁。
这种行为与社区中早得多的研究趋势有关。
例如，ReAct [@yao2023react] 展示了如何把动作和推理交织在一次模型生成中：

> In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments.

随着工具使用能力定型、推理模型起飞，多轮工具使用已经成为一个令人兴奋的研究方向 [@wang2025ragenunderstandingselfevolutionllm]。
用 RL 训练这些多步行为，相比逐样本的 RLHF 循环更接近经典强化学习：智能体在完整轨迹上与环境及其工具交互，然后才获得奖励，如 @fig:tool-use-rl 所示。

![多步工具使用的强化学习。从训练数据中采样一个 prompt，智能体（策略 $\pi_\theta$）在一条轨迹上与环境和工具交互，在动作 $a_t$ 与观测 $o_t$ 之间交替。完成的轨迹在末尾被评分或验证，产生单个奖励 $r_T$，该奖励驱动策略更新。不同于逐样本 RLHF 循环，奖励只在多步 rollout 之后到来，更接近经典 RL。](images/tool_use_rl_loop.png){#fig:tool-use-rl data-dark-src="images/tool_use_rl_loop-dark.png"}

## Model Context Protocol

Model Context Protocol (MCP) 是一个开放标准，用于把语言模型连接到外部数据源和信息系统 [@anthropic_mcp_2024]。
在数据层，MCP 使用 JSON-RPC 2.0，并为其基本组件提供发现和执行方法。
MCP 不要求每个外部系统都有特定工具调用格式，而是让模型能够通过标准化协议访问丰富的上下文信息。

MCP 是建立在本章工具使用内容之上的简单补充：它是应用程序以可预测 JSON schema 向语言模型传递上下文（数据 + 动作）的方式。
模型交互的 MCP servers 具有核心 primitives：resources（只读数据 blob）、prompts（模板化消息/工作流）和 tools（模型可调用函数）。
借此，MCP 架构可以概括为：

- MCP servers 封装特定数据源或能力。
- MCP clients（例如 Claude Desktop、IDE 插件）聚合一个或多个 servers。
- Hosts，例如 Claude 或 ChatGPT 应用，提供用户/LLM 接口；切换模型供应商或后端工具时，只需要替换中间的 client。

MCP 让工具使用模型的开发者可以使用同一套基础设施，把自己的 servers 或 clients 连接到不同模型；同时，模型也拥有可预测格式来集成外部组件。
二者结合，为真实领域中的工具使用模型提供了可预测得多的开发环境。

MCP server 通过标准化 JSON schema 向 clients 暴露工具：
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

一个实现该工具的最小 Python MCP server：
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="get_weather",
        description="Get current weather",
        inputSchema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = fetch_weather(arguments["location"])
        return [TextContent(type="text", text=weather)]
```

## 实现细节

实现工具使用模型时，有多种格式化和 masking 决策：

- **Python vs. JSON formatting**：本章包含了把工具使用格式化为 JSON 数据结构和 Python 代码的示例。模型往往会选择其中一种结构，而行业中不同供应商使用不同格式。
- **Masking tool outputs**：训练工具使用模型时，一个重要细节是工具输出中的 token 会从模型训练损失中 mask 掉。这确保模型不会学习预测处理工具调用的系统输出（因为这些结果不是模型生成的 token）。
- **Multi-turn formatting for tool invocations**：实现工具调用模型时，常见做法是在数据加载格式中加入更多结构。后训练数据集的标准形式是用户和助手（通常还有 system message）交替出现的消息列表。工具使用的整体结构相同，但模型的轮次会被拆成由每次工具调用分隔的内容小节。下面是一个例子。

```python
messages = [
{
"content": "You are a function calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.",
"function_calls": null,
"functions": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]",
"role": "system"
},
{
"content": "Where can I find live giveaways for beta access and games?",
"function_calls": null,
"functions": null,
"role": "user"
},
{
"content": null,
"function_calls": "live_giveaways_by_type(type='beta')\nlive_giveaways_by_type(type='game')",
"functions": null,
"role": "assistant"
}
]
```

- **Tokenization and message format details**：OpenAI messages 格式中的工具调用通常会通过 chat templates（控制发送给模型的消息格式的代码）进行 tokenization，把结构化 JSON 表示转换为原始 token 流。这个过程因模型架构而异：有些使用特殊 token 标记工具调用，有些则在 token 流本身保留结构化格式。[Chat template playgrounds](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B) 提供了交互环境，可探索不同模型如何把消息格式转换为 token 流。
- **Reasoning token continuity**：随着推理模型出现，它们在答案之前拥有独立的“reasoning” token 流；不同实现对工具使用循环中如何处理这些 token 有不同选择。有些模型会在单轮内的工具调用步骤之间保留 reasoning tokens，从而在多次工具调用之间维持上下文。不过，这些 token 通常会在轮次之间被清除，以降低服务化成本（但并非总是如此，这是一项设计决策）。
- **API formatting across providers**（截至 2026 年 5 月）：不同供应商使用概念相似但技术上不同的格式。OpenAI 的 Chat Completions API 使用带唯一 ID 的 `tool_calls` 数组，而较新的 Responses API 把调用表示为 `function_call` items，并把结果作为以 `call_id` 为键的 `function_call_output` items 返回。Anthropic 用 `input_schema` 定义工具，并把调用和结果表示为 `tool_use` 与 `tool_result` content blocks。Gemini 暴露 `AUTO`、`ANY`、`NONE` 等函数调用模式，并在受支持的 Gemini 和 Vertex AI 配置中提供 `VALIDATED`。
- **Schema conformance and constrained decoding**：生产系统通常用 constrained decoding 或 “strict mode” 选项来强制生成有效 JSON 和正确参数类型，减少格式错误输出带来的重试。一些闭源模型供应商会专门做额外后训练，让结构化 JSON 输出更可靠；而对于开放模型，这通常作为 vLLM 等系统中的推理标志处理。
- **Tool output context consumption**：工具输出会快速消耗模型上下文窗口，尤其是搜索或检索工具返回大量结果时。系统必须决定如何截断、总结或分页工具输出，以在保持上下文可控的同时保留模型继续执行所需的信息。

回到后训练：工具使用训练数据来自哪里，使用什么目标函数？
人工编写的工具轨迹收集成本高昂，因此现代工具使用语料大多是合成或 bootstrapped 的，例如 Toolformer 风格的自标注 [@schick2023toolformerlanguagemodelsteach]，或 ToolBench 中的大规模生成 [@qin2023toollm]。
对于训练目标，在工具轨迹上做监督微调（SFT）可以教授基本格式和工具选择。
这会 bootstraps 该行为，通常足以建立这项技能的基础。
对轨迹做偏好优化（例如 DPO）可以改善何时调用工具、何时直接回答的决策。
对于涉及多步工具使用的智能体任务，带环境反馈的 RL（任务成功、约束满足）成为自然目标：模型从其带工具增强的动作是否真的解决了问题中学习。
