import sqlite3

from langchain.memory import ConversationSummaryBufferMemory


def save_memory_to_db(memory):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT
        )
    """)
    summary = memory.buffer
    cursor.execute("INSERT INTO conversation_memory (summary) VALUES (?)", (summary,))
    conn.commit()
    conn.close()

def load_memory_from_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM conversation_memory ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200, memory_key="history", return_messages=True)
    if row:
        memory.buffer = row[0]
    return memory
