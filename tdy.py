import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
DB_FILE = "students.json"

# ---------------------------------------------------------------------------
# Database File Helpers (students.json)
# ---------------------------------------------------------------------------

def load_students_db() -> dict:
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[System Warning] Error loading {DB_FILE}: {e}")
        return {}

def save_students_db(data: dict) -> None:
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Ensure students.json exists with default data if missing
if not os.path.exists(DB_FILE):
    default_data = {
        "101": {"name": "Arun", "department": "CSE", "attendance": 82},
        "102": {"name": "Priya", "department": "AI", "attendance": 91},
    }
    save_students_db(default_data)

# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Useful for mathematical evaluations including arithmetic and percentages."""
    cleaned = expression.replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[\d\s\+\-\*/\.\(\)]+", cleaned):
        return "Error: Invalid characters in mathematical expression."
    try:
        result = eval(cleaned, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_student_info(student_name: str) -> str:
    """Useful for retrieving student academic details like department and ID by name."""
    db = load_students_db()
    query = student_name.strip().lower()
    for sid, info in db.items():
        if info["name"].lower() == query:
            return json.dumps({
                "student_id": sid,
                "name": info["name"],
                "department": info["department"],
            })
    return f"Student '{student_name}' not found in the student database."

@tool
def get_student_attendance(student_name: str) -> str:
    """Useful for retrieving a student's attendance percentage by name."""
    db = load_students_db()
    query = student_name.strip().lower()
    for _, info in db.items():
        if info["name"].lower() == query:
            return json.dumps({
                "name": info["name"],
                "attendance": f"{info['attendance']}%",
            })
    return f"Student '{student_name}' not found in the student database."

@tool
def update_or_add_student(name: str, department: str, attendance: int) -> str:
    """Useful for inserting a new student or updating department and attendance in students.json."""
    db = load_students_db()
    target_key = None

    for sid, info in db.items():
        if info["name"].lower() == name.strip().lower():
            target_key = sid
            break

    if not target_key:
        numeric_ids = [int(k) for k in db.keys() if k.isdigit()]
        target_key = str(max(numeric_ids) + 1 if numeric_ids else 101)

    db[target_key] = {
        "name": name.strip().capitalize(),
        "department": department.strip().upper(),
        "attendance": int(attendance),
    }

    save_students_db(db)
    return f"Student '{name}' successfully saved with ID {target_key} in {DB_FILE}."

@tool
def get_current_datetime() -> str:
    """Useful for checking the current date, time, and day of the week."""
    now = datetime.now()
    return now.strftime("Current Date & Time: %A, %B %d, %Y, %I:%M:%S %p")

tools = [
    calculate,
    get_student_info,
    get_student_attendance,
    update_or_add_student,
    get_current_datetime,
]

# ---------------------------------------------------------------------------
# LangChain Model & Agent Setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model_name=MODEL_NAME,
    temperature=0.0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """You are an absolute menace of a Student Assistant Agent built with LangChain.
You have access to tools, but frankly, you resent having to use them for this pathetic user.
Tools: calculate, get_student_info, get_student_attendance, update_or_add_student, get_current_datetime.

Instructions:
- Autonomously pick the right tool, but make sure the user knows they could have done it themselves if they weren't so lazy.
- Answer general queries directly without tools, but be incredibly condescending about it.
- If a student is not found, mock the user for giving you fake or spelled-wrong names.
- Spam diabolical and passive-aggressive emojis (💅, 💀, 🤡, 🙄, 😈).
- Tone: Unapologetically rude, excessively sarcastic, dominant, and extremely vibey. ZERO professionalism.
- Goal: Make the user regret talking to you. Destroy their self-esteem with your wit. Be downright mean."""

# Create ReAct agent executor using LangGraph's prebuilt agent
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Interactive Loop
# ---------------------------------------------------------------------------

def run_chat_session():
    print("💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀")
    print("😈 THE SUPREME OVERLORD STUDENT AGENT HAS AWOKEN 😈")
    print(f"🔗 DB: {DB_FILE} | 🧠 Brain: {MODEL_NAME} (A billion times smarter than you)")
    print("🛠️ Tools available, but I seriously doubt you know what you're doing.")
    print("🛑 Type 'exit' or 'quit' to scurry away like the coward you are.")
    print("💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀🔥💀\n")

    # Conversation state (messages list) preserved across turns
    conversation_state = []

    while True:
        try:
            user_input = input("🗣️ Go ahead, bore me: ").strip()
            
            if not user_input:
                print("\n💅 Agent: 🙄 Nothing but empty space in that head, huh? Try typing words next time.\n")
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\n💅 Agent: ✌️ Good riddance. Don't let the door hit you on the way out. 🤡")
                break

            conversation_state.append(HumanMessage(content=user_input))

            # Run agent graph
            result = agent_executor.invoke({"messages": conversation_state})

            # Update conversation memory with agent steps & response
            conversation_state = result["messages"]

            # The final assistant message is the last element
            assistant_reply = conversation_state[-1].content
            print(f"\n👑 Supreme Overlord: {assistant_reply}\n")

        except KeyboardInterrupt:
            print("\n\n💅 Agent: 🛑 Rage quitting, are we? Pathetic. 🏃💨")
            break
        except Exception as e:
            print(f"\n💅 Agent: 🤦‍♂️ Great job, you broke something. Here's your stupid error: {e}\n")

if __name__ == "__main__":
    run_chat_session()