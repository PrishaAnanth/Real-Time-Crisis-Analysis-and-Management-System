'''1 Text Normalization (Lowercasing)
2. Keyword-based Classification (Rule-Based NLP)
3. Weighted Scoring System (Rule-Based Inference)
4. Lexical Resource for Urgency Detection
5. Information Extraction with Regex
match = re.search(r"\b(?:near|at|in)\s+([A-Za-z0-9\s]+)", message, re.IGNORECASE)
Extracts location entities based on patterns.
Looks for phrases like "at hospital", "in school", "near bridge".
This is a rule-based Named Entity Recognition (NER) approach, but using regular expressions instead of ML.
6. Categorization + Scoring = Hybrid Rule-Based NLP'''


import re

def process_message(message: str):
    message_lower = message.lower()
    emergency_type = "Unknown"
    urgency = "Low"
    location = "Unknown"
    priority = 1  # base priority

    # --- Emergency type detection ---
    if "fire" in message_lower:
        emergency_type, priority = "Fire", 3
    elif "flood" in message_lower:
        emergency_type, priority = "Flood", 3
    elif "earthquake" in message_lower:
        emergency_type, priority = "Earthquake", 4
    elif "accident" in message_lower:
        emergency_type, priority = "Accident", 2
    elif "storm" in message_lower:
        emergency_type, priority = "Storm", 2
    elif "landslide" in message_lower:
        emergency_type, priority = "Landslide", 3

    # --- Urgency detection ---
    if any(word in message_lower for word in ["help", "immediately", "urgent", "asap", "emergency"]):
        urgency, priority = "High", priority + 2
    elif any(word in message_lower for word in ["soon", "quickly"]):
        urgency, priority = "Medium", priority + 1

    # --- Location detection ---
    match = re.search(r"\b(?:near|at|in)\s+([A-Za-z0-9\s]+)", message, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
        location = re.sub(r"(please|urgent|immediately|help).*", "", location, flags=re.IGNORECASE).strip()

    return {
        "message": message,
        "location": location,
        "emergency_type": emergency_type,
        "urgency": urgency,
        "priority": priority
    }
