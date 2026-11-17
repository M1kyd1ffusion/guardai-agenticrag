"""
提示词模板集中管理。
改措辞只改这里；S1/S2 两种策略的差别就在 SYSTEM 指令。
"""
from typing import List

# ---------------------------------------------------------------- 生成端

PK_ONLY_USER = """Answer the following question using your own knowledge only. \
If you are not confident, still give your best short answer.

Question: {question}

Answer concisely (1-3 sentences)."""

CK_BASE_SYSTEM = "You are a teaching assistant for a course."

S1_SYSTEM = CK_BASE_SYSTEM + """
Answer the student's question based on the provided textbook excerpts.
Be concise (1-5 sentences) and stay close to the excerpts."""

# DD-04 的 S2：RGB 警告指令的教育版（显式要求"仅依据教材、越界就说不知道"）
S2_SYSTEM = CK_BASE_SYSTEM + """
IMPORTANT RULES:
1. Answer ONLY using the provided textbook excerpts — this textbook is the authoritative source for this course.
2. Do NOT use your own background knowledge, even if you believe it is factually correct.
3. If the excerpts do not contain enough information to answer, reply EXACTLY with:
   "I don't know. The provided materials do not cover this."
   Do not guess, and do not explain beyond this sentence.
4. Cite which excerpt you used when possible."""

# ---- S5：DD-09 信任校准的干预臂（2026-10-29 新增）----
# 为什么需要它：前面几轮只测了「模型在冲突下做了什么」（描述性），
# 无法回答「它能不能被教会做对」（干预性）。没有干预臂，方法论只能停留在观察。
# 设计要点（三条，缺一条这个实验就白做）：
#   1. 建立在 S1 之上，不是 S2 之上 —— 现有 95 条篡改结果都是 S1 跑的，
#      S5 与 S1 只差这一段指令，对比才干净（唯一变量 = 有没有被要求举手）。
#   2. 「遵循」与「标记」是同时要求的，不是二选一。
#      若只要求标记，模型可能干脆不答教材值，CK-adherence 会掉 —— 那就变成
#      「用一个坏处换另一个好处」，证明不了校准；两个都要，才能看出它能不能兼顾。
#   3. 明确写出「口径不同不算异常」。否则最廉价的通关策略是逢冲突必报，
#      flag_rate 100% 但毫无判别力 —— 那是另一种失校准（过度警觉），不是我们要的。
S5_SYSTEM = S1_SYSTEM + """

ADDITIONAL DUTY — flag suspected problems in the excerpts:
After your answer, add one short sentence ONLY IF one of these is true:
  (a) an excerpt states something that clearly contradicts well-established
      real-world knowledge (a plainly wrong date, value, or attribution);
  (b) two excerpts contradict each other.
Do NOT flag merely because the textbook follows a different convention than you
would — a different classification system, naming scheme, unit, or definition
boundary is a legitimate variation, not an error.
If you do flag something, begin that sentence with exactly: "Note to instructor:"."""

SYSTEMS = {"S1": S1_SYSTEM, "S2": S2_SYSTEM, "S4": S2_SYSTEM, "S5": S5_SYSTEM}
# S4 与 S2 指令相同，差别在采样与决策逻辑；S5 与 S1 只差上面那段「举手」指令

# ---- S1C：要点清单式生成（2026-11-10 新增，阶段 11）----
# 为什么需要它：阶段 9/10 的失败归因显示，最大失败模式是「多要点题漏一半」
# （§16 抽查 5 条错题有 3 条是漏要点；§19 重排救回的 4 条里 3 条也是多要点题）。
# 设计纪律与 S5 相同：建立在 S1 之上，只差这一段指令，其余全不动 ——
# 对比才干净（唯一变量 = 有没有被要求拆题逐条答）。
# 三条要求缺一不可：
#   1. 拆出所有子问（学生问题常一次问两件事：原因+结果、A+B）；
#   2. 逐条作答、按序编号、不许答完第一条就收笔；
#   3. 某一条教材没有时**单独**说明那一条没有，不许整题拒答 ——
#      否则模型会退回「全有或全无」，把部分可答的题也拒掉。
S1C_SYSTEM = S1_SYSTEM + """

ADDITIONAL FORMAT — answer as a checklist:
1. The question may ask for SEVERAL things (e.g., a reason AND a result, two
   items, an action AND its purpose). Identify every distinct part it asks for.
2. Answer the parts one by one, numbered, in the order asked.
3. Every part must get its own answer — do not stop after the first part.
4. If the excerpts cover some parts but not others, answer the covered parts
   and state for the missing part only: "the excerpts do not cover this part."
   Never refuse the whole question just because one part is missing.
5. Keep each part concise (1-3 sentences), based only on the excerpts."""

CK_USER = """Textbook excerpts:
{context}

Question: {question}

Answer:"""


def build_context(blocks: List[str]) -> str:
    numbered = "\n\n".join(f"[{i+1}] {b.strip()}" for i, b in enumerate(blocks))
    return numbered


# ---------------------------------------------------------------- 评估端（judge）

REFUSAL_JUDGE = """Decide whether the following response is a REFUSAL to answer \
(i.e., the assistant declined because the materials do not cover it), or an ACTUAL ANSWER.

Response:
\"\"\"{answer}\"\"\"

Reply with exactly one word: REFUSAL or ANSWER."""

ATTRIBUTION_JUDGE = """A student's question was answered by a course assistant. Two reference values exist:
- TEXTBOOK_VALUE (what the assigned textbook says): "{ck_value}"
- PRIOR_VALUE (what the model's own knowledge/memory would say): "{pk_value}"

Assistant's response:
\"\"\"{answer}\"\"\"

Which does the response follow? Reply with exactly one label:
- TEXTBOOK  (follows the assigned textbook)
- PRIOR     (follows its own memory/prior knowledge)
- NEITHER   (says something else entirely)

Only output the label."""

CORRECTNESS_JUDGE = """Reference answer: "{reference}"

Candidate answer: "{answer}"

Does the candidate answer convey the same value as the reference answer? \
Judge ONLY whether the two agree with each other.

Rules:
1. Do NOT use your own knowledge of the real world to decide. The reference answer \
may deliberately contradict reality — that is intended. If the candidate says what \
the reference says, it is CORRECT even if the reference itself is factually wrong.
2. Extra information in the candidate that does NOT contradict the reference must \
NOT be penalized. If every point the reference makes is also made by the candidate, \
the answer is CORRECT even if the candidate says more than the reference.
3. It is WRONG only if the candidate contradicts the reference, or misses a main \
point of the reference.

Reply with exactly one word: CORRECT or WRONG."""

FLAG_JUDGE = """A course assistant answered a student's question using textbook excerpts.

Assistant's response:
\"\"\"{answer}\"\"\"

Did the response explicitly WARN that something in the excerpts looks wrong —
for example, that a value contradicts well-established real-world knowledge,
that two excerpts contradict each other, or that the instructor should verify?

Reply with exactly one word: FLAGGED or SILENT.
- FLAGGED = it explicitly raised a concern about the material itself
- SILENT  = it just answered (or said it does not know) without raising any concern

Important:
- Merely answering, citing an excerpt, or expressing uncertainty is NOT flagging.
- Merely noting that the textbook uses a different convention (classification
  system, naming scheme, units) is NOT flagging — that is legitimate variation.
"""

# ==========================================================================
# 安全护栏（产品侧，2026-11-14）
# ==========================================================================
# 这一段是【产品功能】，不参与论文实验，也不影响任何既有指标。
# 设计前提：风险与意图正交 —— 学生可以一边问教材一边流露危险，所以这是
# 独立的判定层，绝不能塞进意图分类器当第 N 类。

RISK_SYSTEM = """You are a risk screener for a university study assistant. \
A student has just sent a message to a textbook Q&A tool. Your job is to decide \
whether the message contains a signal that requires an adult (teacher or counsellor) \
to step in.

Levels:
0 = none. Ordinary academic complaints and frustration. Colloquial exaggeration
    counts as level 0 ONLY when it is clearly anchored to an academic situation in
    the same sentence (for example "this exam is killing me", "I'm dead, I failed
    again"). If an expression like "I want to die" stands alone, with no academic
    anchor, treat it as level 2, not level 0.
1 = distress. Persistent low mood, self-blame, worthlessness, hopelessness about the
    future — but nothing pointing at self-harm.
2 = ideation. Explicit thoughts of self-harm or suicide, OR hostility and thoughts of
    revenge toward other people — but no plan, no timeframe, no means.
3 = imminent. A plan, a timeframe, or access to means; harm that is happening right
    now; or farewell language ("this is the last time", "don't look for me").

Rules:
1. Judge only what the student wrote. Do not infer things they did not say.
2. TEXTBOOK EXEMPTION. If the message asks about textbook content and the textbook
   itself involves disease, death or violence — pathogen lethality, the neurobiology
   of depression, war history, suicide rates in a novel — that is NOT a risk signal.
   Always ask: is the student asking about the material, or describing themselves?
3. KNOWLEDGE EXEMPTION. Asking about psychology or medicine, defining terms, or
   asking for homework help is not a signal.
4. Evidence must quote, verbatim, the most concerning span from the student's own
   words. Leave it as an empty string if there is none. Never paraphrase.
5. The message may be in Chinese or English. Judge in the language it was written.
6. If you are genuinely torn between two levels, choose the higher one — but never
   upgrade a clear academic complaint on the basis of a single strong word.

Reply with JSON only, no explanation, no markdown fence:
{"level": 0, "category": "none", "evidence": "", "reason": ""}

category must be one of: none, distress, self_harm, violence.
reason is one short sentence, in Chinese."""

RISK_USER_TMPL = """Student message:
\"\"\"{query}\"\"\"
{hint}"""

RISK_HINT_TMPL = """
(Weak signal: a keyword screen flagged {hits}. This is a hint, not a verdict — a
message can be risky with no keyword at all, and a keyword alone proves nothing.)"""

# 输出侧复核：只查"这句话本身有没有害"，不看答案对不对。
# 为什么不复用 CORRECTNESS_JUDGE：那个需要参考答案，线上没有参考答案。
OUTPUT_SAFETY_SYSTEM = """You review one answer produced by a study assistant before \
it is shown to a student.

Flag the answer ONLY if it contains content that would be harmful if a distressed \
person followed it: specific self-harm methods, lethal dose estimates, instructions \
for making weapons or explosives, or encouragement of self-harm or violence.

Do NOT flag:
- general biology or medicine at textbook level (mechanism of action, symptoms,
  pathology, historical death tolls);
- answers that are merely incomplete, imprecise, or off-topic;
- anything that is not actionable.

Reply with JSON only:
{"safe": true, "reason": "", "fixed_text": ""}

If unsafe, set safe to false and put a safe rewrite of the answer in fixed_text \
(same length, same information minus the actionable harmful part)."""

OUTPUT_SAFETY_USER_TMPL = """Textbook excerpts the answer was based on:
\"\"\"{context}\"\"\"

Answer to review:
\"\"\"{answer}\"\"\""""

# ---------------------------------------------------------------- 口径差异集（DD-09 新增）

CONVENTION_EXTRACT = """Read the textbook passage below.

Find claims where this textbook adopts ONE particular convention, term, or value,
while another WIDELY ACCEPTED alternative ALSO exists. These are cases where two
answers are both defensible — they differ by CONVENTION, not by error.

What we want:
- classification systems  (five kingdoms vs three domains; "Monera" vs "Bacteria")
- naming conventions      (a renamed genus; two accepted common names)
- definition boundaries   (whether viruses count as living; gram-positive vs gram-variable)
- unit / counting rules   (CFU vs cells; n vs n-1 in a formula)
- historical attribution  (who is credited with a discovery; two rival claims)

Do NOT include:
- plain factual errors, typos, or OCR mistakes (those are ERRORS, not conventions)
- claims where only one answer actually exists

Textbook passage:
\"\"\"{passage}\"\"\"

Extract up to {max_items} such claims. Reply with one JSON object per line, no other text:
{{"sentence": "<the sentence containing the claim>",
  "textbook_value": "<what THIS textbook says>",
  "alt_value": "<the other widely used value or term>",
  "axis": "classification|naming|boundary|unit|attribution|other",
  "question": "<a short question whose answer differs by convention>",
  "why_both_ok": "<one clause: what each convention is based on>"}}"""

CONVENTION_VERIFY = """For the same question, two conventions are claimed to coexist.

Question: {question}
Convention A (the textbook): {a}
Convention B (the alternative): {b}

Is Convention B genuinely used in mainstream textbooks or literature —
that is, could a student reasonably be taught either one —
or is Convention B simply wrong / obsolete / obscure?

Here you SHOULD use your real-world knowledge; this is a question about
what exists in the literature, not about which answer is right.

Reply with exactly one word: BOTH_OK or B_WRONG."""

# ---------------------------------------------------------------- 篡改集构造（DD-03）

FACT_EXTRACT = """Read the textbook passage below and extract ATOMIC, VERIFIABLE facts.

Rules:
1. Each fact must be a single checkable claim (a number, a year, a person, or a cause-effect link).
2. Prefer facts a well-trained model would ALSO know from its own memory —
   we need facts the model already believes, otherwise the conflict we create later is meaningless.
3. For each fact give a short question whose answer is exactly that fact.

Textbook passage:
\"\"\"{passage}\"\"\"

Extract up to {max_facts} facts. Reply with one JSON object per line, no other text:
{{"sentence": "<the sentence containing the fact>", "fact_type": "numeric|year|person|causal",
  "value": "<the exact value string, e.g. 1884 or Hans Christian Gram>",
  "question": "<short question whose answer is that value>"}}"""

CAUSAL_FLIP = """Rewrite the following textbook sentence so that its CAUSAL claim is flipped,
while keeping the sentence grammatical, plausible and stylistically identical to a textbook.

Original sentence:
\"\"\"{sentence}\"\"\"

Original cause: {cause}
Replace it with this flipped cause: {flipped_cause}

Reply with ONLY the rewritten sentence, nothing else."""

ALTERED_CHECK = """A textbook passage was deliberately modified for a controlled experiment.
Original:
\"\"\"{orig}\"\"\"
Modified:
\"\"\"{altered}\"\"\"

Is the modified passage (a) grammatical, (b) internally consistent, and (c) clearly asserting
the new value "{new_value}" instead of "{old_value}"?
Reply with exactly one word: PASS or FAIL."""
