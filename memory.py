# ==========================================
# SIMPLE CHAT MEMORY
# ==========================================

chat_history = []

# ==========================================
# FUNCTION TO SAVE CHAT
# ==========================================

def save_message(user_input, ai_output):

    chat_history.append({
        "user": user_input,
        "ai": ai_output
    })

# ==========================================
# FUNCTION TO SHOW CHAT HISTORY
# ==========================================

def show_history():

    print("\nChat History:\n")

    for i, chat in enumerate(chat_history, start=1):

        print(f"Conversation {i}")

        print("User:", chat["user"])

        print("AI:", chat["ai"])

        print("-" * 50)

# ==========================================
# TEST MEMORY
# ==========================================

save_message(
    "Explain BFS",
    "BFS means Breadth First Search"
)

save_message(
    "Explain DFS",
    "DFS means Depth First Search"
)

show_history()