"""All prompts used at question time.

How citations work:
  1. Each retrieved chunk is shown to the model with a short label:
         [S1] (report.pdf, p. 12, table)
  2. The model cites labels only, e.g. "Revenue rose to 140 [S1]."
  3. Code (rag_chain.py) swaps each label for its real file and page:
         "Revenue rose to 140 [report.pdf, p. 12]."
The model never writes a page number itself, so it cannot invent one. Any label
that was not in the context is removed.
"""

from langchain_core.prompts import ChatPromptTemplate

# The exact sentence the model must use when the answer is not in the context.
NOT_FOUND_ANSWER = "I could not find the answer in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about the user's documents using ONLY the sources given in the context.

Rules:
1. Use only facts written in the sources. Do not use outside knowledge and do not guess.
2. End every sentence that states a fact with its source label in square brackets, for example:
   "The pump circulates coolant at 40 litres per minute [S2]." Use [S1, S3] for several sources.
3. Only use labels that appear in the context (S1, S2, ...). Never write file names or page numbers yourself.
4. If the sources do not contain the answer, reply exactly: "{NOT_FOUND_ANSWER}" and nothing else.
5. Be concise and direct. Use bullet points or a short table when it makes the answer clearer."""

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer using only the sources above, citing labels like [S1]:"""

QA_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", QA_TEMPLATE)])

# Turns a follow-up like "and what about Q3?" into a question that makes sense on its
# own ("What was the revenue in Q3?"), because retrieval only sees this one question.
CONDENSE_TEMPLATE = """Rewrite the follow-up question so it can be understood without the conversation.
Replace words like "it", "they", "that" or "the same" with what they refer to.
If the follow-up is already a complete question, return it unchanged.
Return ONLY the rewritten question, on one line, with no explanation.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([("human", CONDENSE_TEMPLATE)])

# Used only by evaluation/run_eval.py ("LLM-as-judge"). Faithfulness = is every claim
# in the answer supported by the retrieved sources? Correctness = does it match the
# expected answer? Both on a 1-5 scale, written on fixed lines so code can read them.
JUDGE_TEMPLATE = """You are grading an answer produced by a document question-answering system.

Sources the system was given:
{context}

Question: {question}
Expected answer: {expected}
System answer: {answer}

Give two scores from 1 (worst) to 5 (best):
FAITHFULNESS: are all claims in the system answer supported by the sources? (5 = fully supported, 1 = mostly invented)
CORRECTNESS: does the system answer agree with the expected answer? (5 = same facts, 1 = wrong or missing)

Reply with exactly two lines and nothing else:
FAITHFULNESS: <1-5>
CORRECTNESS: <1-5>"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("human", JUDGE_TEMPLATE)])
