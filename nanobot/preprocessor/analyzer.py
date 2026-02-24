"""
消息预处理器 - 信息质量分析与熵值计算

流程:
用户消息 -> [主客观拆解 + 事实核实 + 偏见检测 + 信息质量参数] (one-shot) -> 熵值计算 -> 复核 (one-shot) -> 输出原文+分析数据

熵值公式 (v3 动态权重 + sigmoid 归一化):
H_raw = (1 - C) + C × η + uncertainty
H = sigmoid(H_raw, k=4.0, center=0.75)  # sigmoid 归一化到 [0, 1]

- C (覆盖率) = 动态权重 × 各项清晰度
  - 基础权重: θ=0.4, κ=0.3, ρ=0.3
  - 动态调整: 主观性影响 κ 和 ρ 的权重
    - w_context = 0.3 × (1 - 0.2 × subjective_ratio)
    - w_requirement = 0.3 × (1 + 0.2 × subjective_ratio)
  - θ: 主题明确度 (topic_clarity)
  - κ: 上下文完整性 (context_completeness)
  - ρ: 需求清晰度 (requirement_clarity)

- η (噪声) = 基础噪声 + 偏见噪声
  - 基础噪声: 0.3δ + 0.4μ + 0.3ε
  - 偏见噪声: bias_score × 0.2
  - δ: 冗余度 (redundancy)
  - μ: 模糊度 (ambiguity)
  - ε: 情绪干扰 (emotional_interference)

- uncertainty (不确定性) = (1 - analysis_confidence) × 0.1
  - analysis_confidence: 分析置信度（基于复核补充比例计算）

总计 2 次 LLM 调用 - 支持 Anthropic 和 OpenAI 兼容格式
"""

import json
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from loguru import logger

from nanobot.providers.base import LLMProvider


class MessageAnalyzer:
    """消息分析器 - 使用 Anthropic SDK 直接执行 LLM 调用"""

    # JSON Schema for analysis result (用于 OpenAI structured output)
    ANALYSIS_SCHEMA = {
        "type": "object",
        "properties": {
            "subjective_parts": {"type": "array", "items": {"type": "string"}},
            "objective_parts": {"type": "array", "items": {"type": "string"}},
            "verifiable_claims": {"type": "array", "items": {"type": "string"}},
            "unverifiable_claims": {"type": "array", "items": {"type": "string"}},
            "potential_errors": {"type": "array", "items": {"type": "string"}},
            "bias_types": {"type": "array", "items": {"type": "string"}},
            "emotional_words": {"type": "array", "items": {"type": "string"}},
            "clear_topics": {"type": "array", "items": {"type": "string"}},
            "vague_expressions": {"type": "array", "items": {"type": "string"}},
            "provided_context": {"type": "array", "items": {"type": "string"}},
            "missing_context": {"type": "array", "items": {"type": "string"}},
            "explicit_requests": {"type": "array", "items": {"type": "string"}},
            "implied_expectations": {"type": "array", "items": {"type": "string"}},
            "repeated_info": {"type": "array", "items": {"type": "string"}},
            "ambiguous_phrases": {"type": "array", "items": {"type": "string"}},
            "strong_emotional_words": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "subjective_parts", "objective_parts", "verifiable_claims",
            "unverifiable_claims", "potential_errors", "bias_types",
            "emotional_words", "clear_topics", "vague_expressions",
            "provided_context", "missing_context", "explicit_requests",
            "implied_expectations", "repeated_info", "ambiguous_phrases",
            "strong_emotional_words"
        ],
        "additionalProperties": False,
    }

    # JSON Schema for review result
    REVIEW_SCHEMA = {
        "type": "object",
        "properties": {
            "additional_subjective_parts": {"type": "array", "items": {"type": "string"}},
            "additional_objective_parts": {"type": "array", "items": {"type": "string"}},
            "additional_bias_types": {"type": "array", "items": {"type": "string"}},
            "additional_emotional_words": {"type": "array", "items": {"type": "string"}},
            "additional_clear_topics": {"type": "array", "items": {"type": "string"}},
            "additional_vague_expressions": {"type": "array", "items": {"type": "string"}},
            "additional_provided_context": {"type": "array", "items": {"type": "string"}},
            "additional_missing_context": {"type": "array", "items": {"type": "string"}},
            "additional_explicit_requests": {"type": "array", "items": {"type": "string"}},
            "additional_implied_expectations": {"type": "array", "items": {"type": "string"}},
            "additional_repeated_info": {"type": "array", "items": {"type": "string"}},
            "additional_ambiguous_phrases": {"type": "array", "items": {"type": "string"}},
            "additional_strong_emotional_words": {"type": "array", "items": {"type": "string"}},
            "corrections": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "additional_subjective_parts", "additional_objective_parts",
            "additional_bias_types", "additional_emotional_words",
            "additional_clear_topics", "additional_vague_expressions",
            "additional_provided_context", "additional_missing_context",
            "additional_explicit_requests", "additional_implied_expectations",
            "additional_repeated_info", "additional_ambiguous_phrases",
            "additional_strong_emotional_words", "corrections"
        ],
        "additionalProperties": False,
    }

    # One-shot 分析 prompt - 列表化输出，避免直接数值估计
    ANALYSIS_PROMPT = """请对以下用户消息进行全面分析：

用户消息：{message}

【重要规则】
所有提取内容必须是原文的**精确引用**（Quote）：
- 直接复制原文中的词句，不要改写、不要推断、不要扩展
- 如果原文是"不喜欢"，不要提取为"厌恶"或"憎恨"
- 如果原文没有明确说出某个词，就不要提取该词
- 情绪词、主题词等都必须是原文中实际出现的词汇

请严格按照以下 JSON 格式输出分析结果（所有字段都是列表，空列表用 [] 表示）：

{{
  "subjective_parts": ["原文中的主观陈述"],
  "objective_parts": ["原文中的客观陈述"],
  "verifiable_claims": ["原文中可验证的事实"],
  "unverifiable_claims": ["原文中不可验证的陈述"],
  "potential_errors": ["原文中可能的错误点"],
  "bias_types": ["偏见类型名称"],
  "emotional_words": ["原文中的情绪词"],
  "clear_topics": ["原文中的主题词"],
  "vague_expressions": ["原文中的模糊表达"],
  "provided_context": ["原文中的背景信息"],
  "missing_context": ["缺失背景（此项可概括）"],
  "explicit_requests": ["原文中的明确请求"],
  "implied_expectations": ["原文中的隐含期望"],
  "repeated_info": ["原文中的重复信息"],
  "ambiguous_phrases": ["原文中的歧义表述"],
  "strong_emotional_words": ["原文中的强烈情绪词"]
}}

字段说明：
- subjective_parts: 原文中的主观陈述（精确引用，如原文的"我觉得..."）
- objective_parts: 原文中的客观事实陈述（精确引用）
- verifiable_claims: 原文中可验证的事实（精确引用）
- unverifiable_claims: 原文中不可验证的陈述（精确引用）
- potential_errors: 原文中可能的错误点（精确引用）
- bias_types: 偏见类型名称（如"确认偏见"、"灾难化思维"等，此项为分类名）
- emotional_words: 原文中的情绪化词汇（精确引用，如原文的"肯定"、"巨大"）
- clear_topics: 原文中明确提及的主题词（精确引用）
- vague_expressions: 原文中的模糊表达（精确引用，如原文的"可能"、"大概"）
- provided_context: 原文中提供的背景信息（精确引用）
- missing_context: 缺失的背景信息（此项可概括，不必精确引用）
- explicit_requests: 原文中的明确请求（精确引用）
- implied_expectations: 原文中的隐含期望（精确引用或从原文推断）
- repeated_info: 原文中重复出现的信息（精确引用）
- ambiguous_phrases: 原文中有歧义的表述（精确引用）
- strong_emotional_words: 原文中强烈情绪色彩的词汇（精确引用）

只输出 JSON，不要任何解释。
"""

    # 复核 prompt - 检查列表完整性，补充遗漏项
    REVIEW_PROMPT = """请对以下分析结果进行复核，检查是否有遗漏或错误：

原始消息：{message}
初步分析结果：
{analysis_json}

请检查：
1. 主客观拆解是否完整？
2. 偏见类型和情绪词是否识别充分？
3. 信息质量分析是否有明显遗漏？

请严格按照以下 JSON 格式输出补充项（无补充则用空列表 []）：

{{
    "additional_subjective_parts": ["遗漏的主观陈述"],
    "additional_objective_parts": ["遗漏的客观陈述"],
    "additional_bias_types": ["遗漏的偏见类型"],
    "additional_emotional_words": ["遗漏的情绪词"],
    "additional_clear_topics": ["遗漏的主题"],
    "additional_vague_expressions": ["遗漏的模糊表达"],
    "additional_provided_context": ["遗漏的背景信息"],
    "additional_missing_context": ["遗漏的缺失背景"],
    "additional_explicit_requests": ["遗漏的明确请求"],
    "additional_implied_expectations": ["遗漏的隐含期望"],
    "additional_repeated_info": ["遗漏的重复信息"],
    "additional_ambiguous_phrases": ["遗漏的歧义表述"],
    "additional_strong_emotional_words": ["遗漏的强烈情绪词"],
    "corrections": ["需要修正的说明"]
}}

只输出 JSON，不要任何解释。
"""

    def __init__(self, provider: LLMProvider, model: str, temperature: float = 0.3,
                 api_key: str | None = None, api_base: str | None = None,
                 client_type: str = "anthropic", max_tokens: int = 4000):
        """
        初始化分析器

        Args:
            provider: LLM Provider 实例（用于获取配置）
            model: 模型名称
            temperature: 温度参数
            api_key: API key（可选，不提供则从 provider 获取）
            api_base: API base URL（可选，不提供则从 provider 获取）
            client_type: 客户端类型 - "anthropic" 或 "openai"
            max_tokens: 最大输出 token 数（默认 800，适合 glm-4.7 等模型）
        """
        self.temperature = temperature
        self.client_type = client_type
        self.max_tokens = max_tokens

        # 从 provider 获取配置
        self.api_key = api_key or getattr(provider, 'api_key', '') or ''
        self.api_base = api_base or getattr(provider, 'api_base', None)

        # 处理模型名称
        self.model = model
        if client_type == "anthropic" and '/' in model:
            parts = model.split('/', 1)
            if parts[0].lower() == 'anthropic':
                self.model = parts[1]

        # 创建对应的客户端
        if client_type == "openai":
            client_kwargs = {"api_key": self.api_key} if self.api_key else {}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            self.client = AsyncOpenAI(**client_kwargs) if client_kwargs else AsyncOpenAI()
        else:  # anthropic
            client_kwargs = {"api_key": self.api_key} if self.api_key else {}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            self.client = AsyncAnthropic(**client_kwargs) if client_kwargs else AsyncAnthropic()

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str = "你是一个专业的分析助手，只输出 JSON 格式，不输出任何解释。",
        max_tokens: int | None = None,
        json_schema: dict | None = None
    ) -> dict[str, Any]:
        """
        使用 Anthropic 或 OpenAI SDK 直接调用 LLM 并解析 JSON 响应

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            max_tokens: 最大 token 数
            json_schema: JSON Schema（用于 OpenAI structured output）
        """
        max_tokens = max_tokens or self.max_tokens
        try:
            if self.client_type == "openai":
                # 构建请求参数
                request_params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "timeout": 120  # 2 分钟超时
                }

                # 如果提供了 JSON Schema，使用 structured output
                if json_schema:
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analysis_result",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.debug("使用 JSON Schema structured output")
                else:
                    # 至少确保输出是 JSON 格式
                    request_params["response_format"] = {"type": "json_object"}

                response = await self.client.chat.completions.create(**request_params)
                # 优先使用 content
                content = response.choices[0].message.content or ""
                # 如果 content 为空，尝试从 reasoning_content 提取
                if not content:
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''
                    # 尝试从 reasoning 中提取 JSON（模型可能在思考中构建了 JSON）
                    import re
                    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', reasoning)
                    if json_match:
                        content = json_match.group(1)
                        logger.debug("从 reasoning_content 提取到 JSON（markdown 格式）")
                    else:
                        # 尝试匹配最后一个完整的 JSON 对象
                        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', reasoning)
                        if json_matches:
                            content = json_matches[-1]  # 取最后一个
                            logger.debug("从 reasoning_content 提取到 JSON（裸格式）")
                    # 如果还是空，记录警告
                    if not content and response.choices[0].finish_reason == "length":
                        logger.warning(f"max_tokens={max_tokens} 不够，模型输出被截断，无法提取 JSON")
            else:
                # Anthropic 格式
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # 提取文本内容（处理可能的 ThinkingBlock 或其他类型）
                content = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        content = block.text
                        break
                    elif hasattr(block, 'type') and block.type == 'text':
                        content = block.text
                        break

            if not content:
                logger.warning(f"无法从响应中提取文本：{response}")
                return {}

            # 提取 JSON（处理可能的 markdown 包裹）
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group()

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败：{e}, 原始内容：{content[:500]}...")
                # 尝试使用 json-repair 修复
                try:
                    import json_repair
                    return json_repair.loads(content)
                except Exception:
                    return {}

        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return {}

    async def analyze(self, message: str) -> dict[str, Any]:
        """One-shot 分析：主客观拆解 + 事实核实 + 偏见检测"""
        prompt = self.ANALYSIS_PROMPT.format(message=message)
        return await self._call_llm(prompt, json_schema=self.ANALYSIS_SCHEMA)

    async def review(self, message: str, analysis: dict[str, Any]) -> dict[str, Any]:
        """复核分析结果"""
        prompt = self.REVIEW_PROMPT.format(
            message=message,
            analysis_json=json.dumps(analysis, ensure_ascii=False, indent=2)
        )
        return await self._call_llm(prompt, json_schema=self.REVIEW_SCHEMA)


def calculate_entropy_v2(
    topic_clarity: float,
    context_completeness: float,
    requirement_clarity: float,
    redundancy: float,
    ambiguity: float,
    emotional_interference: float,
) -> tuple[float, float, float]:
    """
    使用新公式计算信息熵

    公式: H_info = (1 - C) + C × η

    - C (覆盖率) = 0.4θ + 0.3κ + 0.3ρ
      - θ: 主题明确度 (topic_clarity)
      - κ: 上下文完整性 (context_completeness)
      - ρ: 需求清晰度 (requirement_clarity)
    - η (噪声) = 0.3δ + 0.4μ + 0.3ε
      - δ: 冗余度 (redundancy)
      - μ: 模糊度 (ambiguity)
      - ε: 情绪干扰 (emotional_interference)

    Returns:
        (entropy, coverage, noise) - 熵值、覆盖率、噪声
    """
    # 计算覆盖率 C = 0.4θ + 0.3κ + 0.3ρ
    coverage = 0.4 * topic_clarity + 0.3 * context_completeness + 0.3 * requirement_clarity

    # 计算噪声 η = 0.3δ + 0.4μ + 0.3ε
    noise = 0.3 * redundancy + 0.4 * ambiguity + 0.3 * emotional_interference

    # 计算信息熵 H_info = (1 - C) + C × η
    entropy = (1 - coverage) + coverage * noise

    # 确保值在 0-1 范围内
    entropy = min(1.0, max(0.0, entropy))
    coverage = min(1.0, max(0.0, coverage))
    noise = min(1.0, max(0.0, noise))

    return entropy, coverage, noise


def calculate_entropy_v3(
    topic_clarity: float,
    context_completeness: float,
    requirement_clarity: float,
    redundancy: float,
    ambiguity: float,
    emotional_interference: float,
    subjective_ratio: float = 0.5,
    bias_score: float = 0.0,
    analysis_confidence: float = 1.0,
) -> tuple[float, float, float, dict[str, float]]:
    """
    动态权重熵值计算（sigmoid 归一化）

    公式: H_raw = (1 - C) + C × η + uncertainty
    归一化: H = sigmoid(H_raw, k=4.0, center=0.75)

    - C (覆盖率) = 动态权重 × 各项清晰度
    - η (噪声) = 固定权重 × 各项噪声 + bias_score × 0.2
    - uncertainty = (1 - analysis_confidence) × 0.1

    Sigmoid 归一化说明:
    - 由于公式特性，H_raw 实际分布约为 [0.5, 0.95]
    - 使用 sigmoid 将其平滑映射到 [0, 1]
    - center=0.75: 原始值 0.75 映射到 0.5（作为分界点）
    - k=4.0: 控制曲线陡峭度
    - 0: 非常容易处理的消息
    - 1: 非常难以处理的消息

    动态权重说明:
    - 主观性 (subjective_ratio) 影响覆盖率权重
      - 高主观性时：上下文参考价值降低，需求理解更重要
    - 偏见评分 (bias_score) 作为额外噪声
    - 分析置信度 (analysis_confidence) 影响熵值不确定性

    Args:
        topic_clarity: 主题明确度
        context_completeness: 上下文完整性
        requirement_clarity: 需求清晰度
        redundancy: 冗余度
        ambiguity: 模糊度
        emotional_interference: 情绪干扰
        subjective_ratio: 主观性比例（影响覆盖率权重）
        bias_score: 偏见评分（作为额外噪声）
        analysis_confidence: 分析置信度（影响不确定性）

    Returns:
        (entropy, coverage, noise, weights_info) - 熵值、覆盖率、噪声、权重信息
    """
    # ========== 动态覆盖率权重 ==========
    # 基础权重
    base_w_topic = 0.4
    base_w_context = 0.3
    base_w_requirement = 0.3

    # 主观性调整：高主观性时，需求理解更重要，上下文参考价值降低
    w_topic = base_w_topic
    w_context = base_w_context * (1 - 0.2 * subjective_ratio)
    w_requirement = base_w_requirement * (1 + 0.2 * subjective_ratio)

    # 归一化
    w_sum = w_topic + w_context + w_requirement
    w_topic /= w_sum
    w_context /= w_sum
    w_requirement /= w_sum

    # 计算覆盖率
    coverage = w_topic * topic_clarity + w_context * context_completeness + w_requirement * requirement_clarity

    # ========== 噪声计算 ==========
    # 基础噪声（固定权重）
    base_noise = 0.3 * redundancy + 0.4 * ambiguity + 0.3 * emotional_interference

    # 偏见作为额外噪声
    bias_noise = bias_score * 0.2

    noise = base_noise + bias_noise

    # ========== 基础熵值 ==========
    entropy_base = (1 - coverage) + coverage * noise

    # ========== 置信度调整（不确定性） ==========
    uncertainty = (1 - analysis_confidence) * 0.1

    # 基础熵值
    entropy_raw = entropy_base + uncertainty

    # ========== sigmoid 归一化映射 ==========
    # 由于公式特性，H_raw 实际分布约为 [0.5, 0.95]
    # 使用 sigmoid 进行平滑映射，获得更均匀的 [0, 1] 分布
    # - center=0.75: 原始值 0.75 映射到 0.5（作为分界点）
    # - k=4.0: 控制曲线陡峭度，使分布更均匀
    #
    # 映射效果:
    # H_raw=0.60 -> H≈0.28 (容易处理)
    # H_raw=0.75 -> H=0.50 (中等)
    # H_raw=0.85 -> H≈0.73 (较难处理)
    # H_raw=0.95 -> H≈0.92 (很难处理)
    entropy = sigmoid(entropy_raw, k=4.0, center=0.75)

    # 确保覆盖率在 0-1 范围内
    coverage = min(1.0, max(0.0, coverage))
    noise = min(1.0, max(0.0, noise))

    # 返回权重信息用于调试
    weights_info = {
        "w_topic": w_topic,
        "w_context": w_context,
        "w_requirement": w_requirement,
        "bias_noise": bias_noise,
        "uncertainty": uncertainty,
    }

    return entropy, coverage, noise, weights_info


def get_original_metrics(message: str) -> dict[str, int]:
    """
    提取原文的客观属性（参照系）

    类似相对论中光速作为恒定参照物，
    原文的客观属性是语言分析中恒定的参照系。

    Returns:
        {"total_chars": 总字数, "total_sentences": 总句数}
    """
    # 总字数（去除空格和换行，适用于中英文）
    total_chars = len(message.replace(" ", "").replace("\n", ""))

    # 总句数（按中英文标点分割，过滤空句）
    import re
    sentences = [s.strip() for s in re.split(r'[。！？.!?]', message) if s.strip()]
    total_sentences = max(len(sentences), 1)

    return {
        "total_chars": total_chars,
        "total_sentences": total_sentences,
    }


def sigmoid(x: float, k: float = 4.0, center: float = 0.5) -> float:
    """
    带参数的 Sigmoid 函数，用于平滑映射到 0-1 范围

    公式: σ(x) = 1 / (1 + e^(-k*(x-center)))

    特点:
    - x = center 时，σ = 0.5
    - k 控制曲线陡峭度（越大越陡）
    - center 控制中心点位置

    Args:
        x: 输入值
        k: 陡峭度参数（默认 4.0）
        center: 中心点位置（默认 0.5）

    Returns:
        0-1 之间的平滑映射值
    """
    import math
    # 防止数值溢出
    exponent = -k * (x - center)
    if exponent > 700:  # e^700 已经非常大
        return 0.0
    if exponent < -700:
        return 1.0
    return 1 / (1 + math.exp(exponent))


def validate_extraction(item: str, original_message: str, min_overlap: float = 0.5) -> bool:
    """
    验证提取的内容是否真实存在于原文中

    用于防止 LLM "幻觉"导致的提取内容膨胀。

    验证逻辑：
    1. 如果提取项是分类名称（如偏见类型），不需要在原文中存在
    2. 如果提取项是原文引用，检查是否有足够的字符重叠

    Args:
        item: 提取的内容
        original_message: 原始消息
        min_overlap: 最小重叠比例（默认 0.5，即 50% 的字符需要在原文中出现）

    Returns:
        True 如果验证通过，False 如果是幻觉
    """
    if not item or not original_message:
        return False

    # 单字符直接检查是否存在
    if len(item) <= 1:
        return item in original_message

    # 计算字符重叠率
    item_chars = set(item)
    original_chars = set(original_message)
    overlap = len(item_chars & original_chars) / len(item_chars)

    return overlap >= min_overlap


def filter_valid_extractions(
    items: list,
    original_message: str,
    skip_validation: bool = False
) -> list:
    """
    过滤掉不在原文中的提取项

    Args:
        items: 提取的列表
        original_message: 原始消息
        skip_validation: 是否跳过验证（用于分类名称字段）

    Returns:
        过滤后的列表
    """
    if skip_validation or not items:
        return items

    return [item for item in items if validate_extraction(item, original_message)]


def calculate_metrics_from_lists(result: dict[str, Any], original_message: str) -> dict[str, Any]:
    """
    从列表数据计算各项指标

    核心原则：
    - 分母使用原文的客观属性（总字数、总句数）作为参照系
    - 分子使用 LLM 识别的列表（转换为字数）
    - 过滤掉不在原文中的"幻觉"提取项

    Args:
        result: LLM 返回的分析结果（列表格式）
        original_message: 原始消息

    Returns:
        计算后的指标字典
    """
    # 获取原文参照系
    ref = get_original_metrics(original_message)
    total_chars = ref["total_chars"]
    total_sentences = ref["total_sentences"]

    # 辅助函数：计算列表内容的总字数
    def list_chars(items: list) -> int:
        return sum(len(str(item)) for item in items) if items else 0

    # ========== 验证并过滤提取项 ==========
    # 需要验证的字段（必须是原文精确引用）
    validated_subjective = filter_valid_extractions(result.get("subjective_parts", []), original_message)
    validated_objective = filter_valid_extractions(result.get("objective_parts", []), original_message)
    validated_verifiable = filter_valid_extractions(result.get("verifiable_claims", []), original_message)
    validated_emotional = filter_valid_extractions(result.get("emotional_words", []), original_message)
    validated_topics = filter_valid_extractions(result.get("clear_topics", []), original_message)
    validated_vague = filter_valid_extractions(result.get("vague_expressions", []), original_message)
    validated_context = filter_valid_extractions(result.get("provided_context", []), original_message)
    validated_requests = filter_valid_extractions(result.get("explicit_requests", []), original_message)
    validated_repeated = filter_valid_extractions(result.get("repeated_info", []), original_message)
    validated_ambiguous = filter_valid_extractions(result.get("ambiguous_phrases", []), original_message)
    validated_strong_emotional = filter_valid_extractions(result.get("strong_emotional_words", []), original_message)

    # 不需要验证的字段（分类名称或允许概括）
    bias_types = result.get("bias_types", [])  # 分类名称，如"确认偏见"

    # 记录过滤统计
    original_counts = {
        "subjective_parts": len(result.get("subjective_parts", [])),
        "objective_parts": len(result.get("objective_parts", [])),
        "verifiable_claims": len(result.get("verifiable_claims", [])),
        "emotional_words": len(result.get("emotional_words", [])),
        "clear_topics": len(result.get("clear_topics", [])),
    }
    filtered_counts = {
        "subjective_parts": len(validated_subjective),
        "objective_parts": len(validated_objective),
        "verifiable_claims": len(validated_verifiable),
        "emotional_words": len(validated_emotional),
        "clear_topics": len(validated_topics),
    }
    hallucination_count = sum(original_counts[k] - filtered_counts[k] for k in original_counts)
    if hallucination_count > 0:
        logger.debug(f"过滤了 {hallucination_count} 个疑似幻觉提取项")

    # ========== 计算各项指标 ==========

    # 1. 主客观比例 = 主观部分字数 / 原文总字数
    subj_chars = list_chars(validated_subjective)
    subjective_ratio = subj_chars / total_chars if total_chars > 0 else 0.5

    # 2. 事实置信度 = 可验证陈述数 / 总句数
    verifiable_count = len(validated_verifiable)
    confidence_score = verifiable_count / total_sentences if total_sentences > 0 else 0.5

    # 3. 偏见评分 = sigmoid((偏见类型数 + 情绪词数) / 总句数)
    # 使用 sigmoid 平滑映射，保留高值区分度
    bias_count = len(bias_types)  # 分类名称，不需要验证
    emotional_count = len(validated_emotional)
    bias_raw = (bias_count + emotional_count) / total_sentences if total_sentences > 0 else 0
    bias_score = sigmoid(bias_raw, k=4.0, center=0.5)  # 0.5 为中心点，每句0.5个偏见词

    # 4. 主题明确度 = 明确主题字数 / 原文总字数
    clear_topics_chars = list_chars(validated_topics)
    topic_clarity = clear_topics_chars / total_chars if total_chars > 0 else 0.5

    # 5. 上下文完整性 = 已提供背景字数 / 原文总字数
    provided_chars = list_chars(validated_context)
    context_completeness = provided_chars / total_chars if total_chars > 0 else 0.5

    # 6. 需求清晰度 = 明确请求字数 / 原文总字数
    explicit_chars = list_chars(validated_requests)
    requirement_clarity = explicit_chars / total_chars if total_chars > 0 else 0.5

    # 7. 冗余度 = 重复信息字数 / 原文总字数
    repeated_chars = list_chars(validated_repeated)
    redundancy = repeated_chars / total_chars if total_chars > 0 else 0

    # 8. 模糊度 = 有歧义表述字数 / 原文总字数
    ambiguous_chars = list_chars(validated_ambiguous)
    # 加上模糊表达
    vague_chars = list_chars(validated_vague)
    ambiguity = (ambiguous_chars + vague_chars) / total_chars if total_chars > 0 else 0

    # 9. 情绪干扰 = sigmoid((强烈情绪词 + 情绪词) 字数 / 原文总字数 * 10)
    # 乘以10放大后用sigmoid映射，使敏感度更合适
    emotional_chars = list_chars(validated_strong_emotional)
    # 加上普通情绪词
    emotional_words_chars = list_chars(validated_emotional)
    emotional_raw = (emotional_chars + emotional_words_chars) / total_chars if total_chars > 0 else 0
    # 放大10倍后用sigmoid，5%情绪词密度为中心点(0.5)
    emotional_interference = sigmoid(emotional_raw * 10, k=4.0, center=0.5)

    return {
        "subjective_ratio": subjective_ratio,
        "confidence_score": confidence_score,
        "bias_score": bias_score,
        "topic_clarity": min(1.0, topic_clarity),
        "context_completeness": min(1.0, context_completeness),
        "requirement_clarity": min(1.0, requirement_clarity),
        "redundancy": min(1.0, redundancy),
        "ambiguity": min(1.0, ambiguity),
        "emotional_interference": min(1.0, emotional_interference),
        # 保留参照系信息用于调试
        "_reference": {
            "total_chars": total_chars,
            "total_sentences": total_sentences,
            "hallucination_filtered": hallucination_count,
        }
    }


# 保留旧函数以兼容
def calculate_entropy(
    subjective_ratio: float,
    bias_score: float,
    confidence_score: float,
    fact_check_results: list[dict[str, Any]]
) -> float:
    """
    [已弃用] 计算消息的熵值 - 保留用于向后兼容

    请使用 calculate_entropy_v2 替代
    """
    import math

    ratio = max(0.01, min(0.99, subjective_ratio))
    ratio_entropy = -ratio * math.log2(ratio) - (1 - ratio) * math.log2(1 - ratio)
    bias_entropy = bias_score
    confidence_entropy = 1 - confidence_score

    if fact_check_results:
        confidences = [r.get("confidence_score", 0.5) for r in fact_check_results if r]
        if len(confidences) > 1:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            inconsistency = math.sqrt(variance)
        else:
            inconsistency = 0
    else:
        inconsistency = 0

    entropy = (
        0.3 * ratio_entropy +
        0.3 * bias_entropy +
        0.2 * confidence_entropy +
        0.2 * inconsistency
    )

    return min(1.0, max(0.0, entropy))


async def analyze_message(
    message: str,
    analyzer: MessageAnalyzer,
    num_iterations: int = 1,
    skip_review: bool = False,
) -> dict[str, Any]:
    """
    执行优化的消息分析流程

    流程:
    1. One-shot 分析（LLM 输出列表）
    2. 从列表计算指标（代码计算，基于原文参照系）
    3. 熵值计算（使用公式 H_info = (1-C) + C×η）
    4. 迭代复核 (num_iterations 轮，默认 1 轮，可跳过)
    5. 返回原文 + 分析数据

    设计原则：
    - LLM 做分类/识别（输出列表）
    - 代码做计算（输出数值）
    - 原文属性作为参照系（类似相对论中的光速）

    Args:
        message: 要分析的消息
        analyzer: 消息分析器实例
        num_iterations: 复核迭代次数
        skip_review: 跳过复核步骤
    """
    logger.info(f"开始分析消息：{message[:50]}...")

    # ========== 第 1 轮：One-shot 分析（列表输出） ==========
    logger.info("第 1 轮：One-shot 分析（列表格式）")
    result = await analyzer.analyze(message)

    # 保存原始列表数据
    list_data = {
        # 主客观拆解
        "subjective_parts": result.get("subjective_parts", []),
        "objective_parts": result.get("objective_parts", []),
        # 事实核实
        "verifiable_claims": result.get("verifiable_claims", []),
        "unverifiable_claims": result.get("unverifiable_claims", []),
        "potential_errors": result.get("potential_errors", []),
        # 偏见检测
        "bias_types": result.get("bias_types", []),
        "emotional_words": result.get("emotional_words", []),
        # 信息质量
        "clear_topics": result.get("clear_topics", []),
        "vague_expressions": result.get("vague_expressions", []),
        "provided_context": result.get("provided_context", []),
        "missing_context": result.get("missing_context", []),
        "explicit_requests": result.get("explicit_requests", []),
        "implied_expectations": result.get("implied_expectations", []),
        "repeated_info": result.get("repeated_info", []),
        "ambiguous_phrases": result.get("ambiguous_phrases", []),
        "strong_emotional_words": result.get("strong_emotional_words", []),
    }

    # 从列表计算指标（使用原文参照系）
    metrics = calculate_metrics_from_lists(list_data, message)

    # 构建分析结果（初始置信度为 1.0，复核后更新）
    current_analysis = {
        **list_data,
        **metrics,
        "analysis_confidence": 1.0,  # 初始值，复核后更新
    }

    # 计算熵值（使用动态权重公式 v3）
    # 初始时 analysis_confidence = 1.0，无不确定项
    entropy, coverage, noise, weights_info = calculate_entropy_v3(
        topic_clarity=metrics["topic_clarity"],
        context_completeness=metrics["context_completeness"],
        requirement_clarity=metrics["requirement_clarity"],
        redundancy=metrics["redundancy"],
        ambiguity=metrics["ambiguity"],
        emotional_interference=metrics["emotional_interference"],
        subjective_ratio=metrics["subjective_ratio"],
        bias_score=metrics["bias_score"],
        analysis_confidence=1.0,  # 初始值
    )
    current_analysis["entropy"] = entropy
    current_analysis["coverage"] = coverage
    current_analysis["noise"] = noise
    current_analysis["_weights_info"] = weights_info

    # ========== 迭代复核（补充列表项） ==========
    if not skip_review:
        for i in range(num_iterations):
            logger.info(f"第{i + 2}轮：迭代复核")
            review = await analyzer.review(message, current_analysis)

            # 记录原始项数（用于计算置信度）
            original_count = sum(len(current_analysis.get(k, [])) for k in list_data.keys())

            # 统计补充项数
            supplement_count = 0
            for key in list_data.keys():
                additional_key = f"additional_{key}"
                if review.get(additional_key):
                    supplement_count += len(review[additional_key])
                    current_analysis[key] = list(set(current_analysis.get(key, []) + review[additional_key]))

            current_analysis["corrections"] = review.get("corrections", [])

            # 基于补充比例计算置信度
            # 补充越多，说明第一次分析越不完整，置信度越低
            if original_count > 0:
                supplement_ratio = supplement_count / original_count
                # 置信度 = 1 - sigmoid(补充比例)，补充比例为0时置信度最高
                analysis_confidence = 1 - sigmoid(supplement_ratio, k=4.0, center=0.2)
            else:
                analysis_confidence = 0.5  # 无法判断时默认中等

            # 直接使用数值，不再转换为等级
            current_analysis["analysis_confidence"] = analysis_confidence

            # 重新计算指标
            updated_list_data = {k: current_analysis.get(k, []) for k in list_data.keys()}
            updated_metrics = calculate_metrics_from_lists(updated_list_data, message)

            # 更新指标
            for k, v in updated_metrics.items():
                current_analysis[k] = v

            # 重新计算熵值（使用动态权重公式 v3，包含 analysis_confidence）
            entropy, coverage, noise, weights_info = calculate_entropy_v3(
                topic_clarity=updated_metrics["topic_clarity"],
                context_completeness=updated_metrics["context_completeness"],
                requirement_clarity=updated_metrics["requirement_clarity"],
                redundancy=updated_metrics["redundancy"],
                ambiguity=updated_metrics["ambiguity"],
                emotional_interference=updated_metrics["emotional_interference"],
                subjective_ratio=updated_metrics["subjective_ratio"],
                bias_score=updated_metrics["bias_score"],
                analysis_confidence=analysis_confidence,
            )
            current_analysis["entropy"] = entropy
            current_analysis["coverage"] = coverage
            current_analysis["noise"] = noise
            current_analysis["_weights_info"] = weights_info

    # 保存原文消息
    current_analysis["original_message"] = message

    logger.info(f"分析完成，熵值：{current_analysis['entropy']:.4f}, 覆盖率：{current_analysis['coverage']:.2%}, 噪声：{current_analysis['noise']:.2%}, 置信度：{current_analysis.get('analysis_confidence', 1.0):.2%}")
    if "_reference" in current_analysis:
        ref = current_analysis["_reference"]
        logger.info(f"参照系：总字数={ref['total_chars']}, 总句数={ref['total_sentences']}")
    if "_weights_info" in current_analysis:
        w = current_analysis["_weights_info"]
        logger.info(f"动态权重：w_topic={w['w_topic']:.3f}, w_context={w['w_context']:.3f}, w_requirement={w['w_requirement']:.3f}, bias_noise={w['bias_noise']:.3f}, uncertainty={w['uncertainty']:.3f}")

    return current_analysis


def create_analyzer_from_config(config: "PreprocessorConfig") -> MessageAnalyzer | None:
    """
    从配置创建消息分析器

    Args:
        config: PreprocessorConfig 实例

    Returns:
        MessageAnalyzer 实例，如果配置无效则返回 None
    """
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from nanobot.config.schema import PreprocessorConfig

    if not config.enabled:
        return None

    provider_config = config.provider
    if not provider_config.api_key or not provider_config.api_base:
        logger.warning("Preprocessor enabled but missing api_key or api_base")
        return None

    return MessageAnalyzer(
        provider=None,
        model=provider_config.model,
        temperature=provider_config.temperature,
        api_key=provider_config.api_key,
        api_base=provider_config.api_base,
        client_type=provider_config.client_type,
        max_tokens=provider_config.max_tokens,
    )


async def preprocess_message(
    message: str,
    config: "PreprocessorConfig"
) -> tuple[str, dict[str, Any] | None]:
    """
    预处理消息的便捷函数

    Args:
        message: 原始消息
        config: 预处理器配置

    Returns:
        (处理后的消息, 分析结果) - 如果预处理器未启用，返回原消息和 None
    """
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from nanobot.config.schema import PreprocessorConfig

    analyzer = create_analyzer_from_config(config)
    if analyzer is None:
        return message, None

    try:
        result = await analyze_message(
            message=message,
            analyzer=analyzer,
            num_iterations=config.num_iterations,
            skip_review=config.skip_review,
        )

        # 构建增强后的消息（附加分析摘要）
        enhanced = f"{message}\n\n[分析摘要: 主观{result['subjective_ratio']:.0%}, 置信度{result['confidence_score']:.0%}, 偏见{result['bias_score']:.0%}]"
        return enhanced, result

    except Exception as e:
        logger.error(f"Message preprocessing failed: {e}")
        return message, None
