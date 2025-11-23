'''from flask import Flask, render_template, request, redirect, url_for
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# NLP model
nlp = spacy.load("en_core_web_sm")

# Storage for crisis issues
issues = []
issue_counter = {"total": 0, "pending": 0, "under_process": 0, "resolved": 0}

# Departments mapping
departments_map = {
    "Fire": "🔥 Fire Department",
    "Flood": "💧 Disaster Management",
    "Accident": "🚑 Ambulance Services",
    "Earthquake": "🏢 National Disaster Team"
}

# TF-IDF vectorizer for duplicate detection
vectorizer = TfidfVectorizer()

def is_duplicate(new_message):
    if not issues:
        return False
    messages = [issue["message"] for issue in issues]
    messages.append(new_message)
    tfidf = vectorizer.fit_transform(messages)
    sim_matrix = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim_matrix.max() > 0.8  # threshold for duplicates

def analyze_message(message):
    doc = nlp(message)

    # Initialize points
    points = 0

    # 1️⃣ Emergency type points
    emergency_type = "Unknown"
    if "fire" in message.lower():
        emergency_type = "Fire"
        points += 5
    elif "flood" in message.lower():
        emergency_type = "Flood"
        points += 4
    elif "accident" in message.lower():
        emergency_type = "Accident"
        points += 3
    elif "earthquake" in message.lower():
        emergency_type = "Earthquake"
        points += 6

    # 2️⃣ Urgency points
    urgency_words = ["urgent", "emergency", "immediate", "help"]
    if any(word in message.lower() for word in urgency_words):
        urgency = "High"
        points += 4
    else:
        urgency = "Low"
        points += 1

    # 3️⃣ Location points (schools, hospitals, crowded places = higher points)
    location = "Unknown"
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LOC"]:
            location = ent.text
            if any(word in location.lower() for word in ["school", "hospital", "market"]):
                points += 3
            else:
                points += 1
            break

    # 4️⃣ Severity keywords (injured, death, major, critical)
    severity_keywords = ["injured", "death", "fatal", "major", "critical", "collapsed"]
    if any(word in message.lower() for word in severity_keywords):
        points += 5

    # 5️⃣ Assign department
    department = departments_map.get(emergency_type, "📞 General Helpline")

    # 6️⃣ Determine Priority based on points
    if points >= 12:
        priority = "Critical"
    elif points >= 7:
        priority = "High"
    else:
        priority = "Moderate"

    return {
        "id": issue_counter["total"] + 1,
        "message": message,
        "emergency_type": emergency_type,
        "urgency": urgency,
        "location": location,
        "priority": priority,
        "department": department,
        "status": "Pending"
    }

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    msg = None
    if request.method == "POST":
        message = request.form["message"].strip()
        if is_duplicate(message):
            msg = "⚠️ Similar report already exists!"
        else:
            result = analyze_message(message)
            issues.append(result)
            issue_counter["total"] += 1
            issue_counter["pending"] += 1
            msg = "✅ Your issue has been submitted successfully!"
    return render_template("user.html", msg=msg, issues=issues, stats=issue_counter)

@app.route("/admin")
def admin():
    return render_template("admin.html", issues=issues, stats=issue_counter)

@app.route("/update/<int:issue_id>/<action>")
def update_issue(issue_id, action):
    for issue in issues:
        if issue["id"] == issue_id:
            if action == "process":
                if issue["status"] == "Pending":
                    issue["status"] = "Under Process"
                    issue_counter["pending"] -= 1
                    issue_counter["under_process"] += 1
            elif action == "resolve":
                if issue["status"] in ["Pending", "Under Process"]:
                    if issue["status"] == "Pending":
                        issue_counter["pending"] -= 1
                    elif issue["status"] == "Under Process":
                        issue_counter["under_process"] -= 1
                    issue["status"] = "Resolved"
                    issue_counter["resolved"] += 1
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)'''
    

'''FINAL RUNNING CODE
from flask import Flask, render_template, request, redirect, url_for, session
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"  # needed for session tracking

# NLP model
nlp = spacy.load("en_core_web_sm")

# Storage for crisis issues
issues = []
issue_counter = {"total": 0, "pending": 0, "under_process": 0, "resolved": 0}

# Departments mapping
departments_map = {
    "Fire": "🔥 Fire Department",
    "Flood": "💧 Disaster Management",
    "Accident": "🚑 Ambulance Services",
    "Earthquake": "🏢 National Disaster Team"
}

# TF-IDF vectorizer for duplicate detection
vectorizer = TfidfVectorizer()

def is_duplicate(new_message):
    if not issues:
        return False
    messages = [issue["message"] for issue in issues]
    messages.append(new_message)
    tfidf = vectorizer.fit_transform(messages)
    sim_matrix = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim_matrix.max() > 0.8  # threshold for duplicates

def analyze_message(message, user_id):
    doc = nlp(message)

    # Initialize points
    points = 0

    # 1️⃣ Emergency type points
    emergency_type = "Unknown"
    if "fire" in message.lower():
        emergency_type = "Fire"
        points += 5
    elif "flood" in message.lower():
        emergency_type = "Flood"
        points += 4
    elif "accident" in message.lower():
        emergency_type = "Accident"
        points += 3
    elif "earthquake" in message.lower():
        emergency_type = "Earthquake"
        points += 6

    # 2️⃣ Urgency points
    urgency_words = ["urgent", "emergency", "immediate", "help"]
    if any(word in message.lower() for word in urgency_words):
        urgency = "High"
        points += 4
    else:
        urgency = "Low"
        points += 1

    # 3️⃣ Location points
    location = "Unknown"
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LOC"]:
            location = ent.text
            if any(word in location.lower() for word in ["school", "hospital", "market"]):
                points += 3
            else:
                points += 1
            break

    # 4️⃣ Severity keywords
    severity_keywords = ["injured", "death", "fatal", "major", "critical", "collapsed"]
    if any(word in message.lower() for word in severity_keywords):
        points += 5

    # 5️⃣ Assign department
    department = departments_map.get(emergency_type, "📞 General Helpline")

    # 6️⃣ Determine Priority
    if points >= 12:
        priority = "Critical"
    elif points >= 7:
        priority = "High"
    else:
        priority = "Moderate"

    return {
        "id": issue_counter["total"] + 1,
        "user_id": user_id,  # track which user submitted it
        "message": message,
        "emergency_type": emergency_type,
        "urgency": urgency,
        "location": location,
        "priority": priority,
        "department": department,
        "status": "Pending"
    }

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    msg = None
    # Assign session user id if not already set
    if "user_id" not in session:
        session["user_id"] = len(session) + 1  # simple user id

    user_id = session["user_id"]

    if request.method == "POST":
        message = request.form["message"].strip()
        if is_duplicate(message):
            msg = "⚠️ Similar report already exists!"
        else:
            result = analyze_message(message, user_id)
            issues.append(result)
            issue_counter["total"] += 1
            issue_counter["pending"] += 1
            msg = "✅ Your issue has been submitted successfully!"

    # Filter only issues reported by this user
    user_issues = [i for i in issues if i["user_id"] == user_id]

    return render_template("user.html", msg=msg, user_issues=user_issues, stats=issue_counter)

@app.route("/admin")
def admin():
    return render_template("admin.html", issues=issues, stats=issue_counter)

@app.route("/update/<int:issue_id>/<action>")
def update_issue(issue_id, action):
    for issue in issues:
        if issue["id"] == issue_id:
            if action == "process":
                if issue["status"] == "Pending":
                    issue["status"] = "Under Process"
                    issue_counter["pending"] -= 1
                    issue_counter["under_process"] += 1
            elif action == "resolve":
                if issue["status"] in ["Pending", "Under Process"]:
                    if issue["status"] == "Pending":
                        issue_counter["pending"] -= 1
                    elif issue["status"] == "Under Process":
                        issue_counter["under_process"] -= 1
                    issue["status"] = "Resolved"
                    issue_counter["resolved"] += 1
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)
'''




'''from flask import Flask, render_template, request, redirect, url_for, session
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # New for Semantic Similarity
from scipy.cluster.hierarchy import linkage, fcluster # New for Clustering
import numpy as np                                   # New for numerical operations

app = Flask(__name__)
app.secret_key = "secret123"  # needed for session tracking

# NLP model
nlp = spacy.load("en_core_web_sm")

# NEW: Sentence Transformer model for Semantic Similarity
# Using 'all-MiniLM-L6-v2' for a good balance of speed and performance
# Note: This requires the 'sentence-transformers' package
try:
    model = SentenceTransformer('all-MiniLM-L6-v2') 
except OSError:
    print("Warning: SentenceTransformer model failed to load. Ensure 'all-MiniLM-L6-v2' is downloaded.")
    model = None


# Storage for crisis issues
issues = []
issue_counter = {"total": 0, "pending": 0, "under_process": 0, "resolved": 0}

# Departments mapping
departments_map = {
    "Fire": "🔥 Fire Department",
    "Flood": "💧 Disaster Management",
    "Accident": "🚑 Ambulance Services",
    "Earthquake": "🏢 National Disaster Team",
    "Storm": "⛈️ Weather Response",
    "Landslide": "⛰️ Geological Response"
}

# TF-IDF vectorizer (kept, but we use semantic duplicate check)
vectorizer = TfidfVectorizer()

# --- Advanced Semantic Duplicate Check ---
def is_duplicate_semantic(new_message):
    """Checks for semantic similarity against active reports."""
    if not model:
        print("Semantic model unavailable. Skipping semantic check.")
        return False

    if not issues:
        return False
        
    # Exclude already resolved or duplicate issues from the comparison set
    comparable_issues = [i for i in issues if i["status"] not in ["Resolved", "Duplicate"]]
    if not comparable_issues:
        return False

    messages = [issue["message"] for issue in comparable_issues]
    
    # 1. Generate Embeddings for all relevant messages (existing + new)
    all_messages = messages + [new_message]
    embeddings = model.encode(all_messages)
    
    new_embedding = embeddings[-1]
    existing_embeddings = embeddings[:-1]
    
    # 2. Calculate Cosine Similarity
    # np.newaxis ensures compatible dimensions for cosine_similarity
    sim_scores = cosine_similarity(new_embedding[np.newaxis, :], existing_embeddings)[0]
    
    # 3. Check against a semantic similarity threshold (tune this value)
    SEMANTIC_THRESHOLD = 0.75 
    
    if np.any(sim_scores > SEMANTIC_THRESHOLD):
        # Return the ID of the detected original issue
        duplicate_index = np.argmax(sim_scores)
        return comparable_issues[duplicate_index]["id"]

    return False

# Alias the new function
def is_duplicate(new_message):
    return is_duplicate_semantic(new_message)


def analyze_message(message, user_id):
    """Performs NLP analysis on the message to determine type, urgency, and priority."""
    doc = nlp(message)

    # Initialize points
    points = 0

    # 1️⃣ Emergency type points
    emergency_type = "Unknown"
    message_lower = message.lower()
    
    # Improved type detection (matches departments_map keys)
    if "fire" in message_lower:
        emergency_type = "Fire"
        points += 5
    elif "flood" in message_lower:
        emergency_type = "Flood"
        points += 4
    elif "accident" in message_lower:
        emergency_type = "Accident"
        points += 3
    elif "earthquake" in message_lower:
        emergency_type = "Earthquake"
        points += 6
    elif "storm" in message_lower or "hurricane" in message_lower:
        emergency_type = "Storm"
        points += 3
    elif "landslide" in message_lower or "mudslide" in message_lower:
        emergency_type = "Landslide"
        points += 5


    # 2️⃣ Urgency points
    urgency_words = ["urgent", "emergency", "immediate", "help", "dying", "danger"]
    if any(word in message_lower for word in urgency_words):
        urgency = "High"
        points += 4
    else:
        urgency = "Low"
        points += 1

    # 3️⃣ Location points
    location = "Unknown"
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LOC"]:
            location = ent.text
            # Higher point for sensitive locations
            if any(word in location.lower() for word in ["school", "hospital", "market", "power plant"]):
                points += 3
            else:
                points += 1
            break

    # 4️⃣ Severity keywords
    severity_keywords = ["injured", "death", "fatal", "major", "critical", "collapsed", "trapped"]
    if any(word in message_lower for word in severity_keywords):
        points += 5

    # 5️⃣ Assign department
    department = departments_map.get(emergency_type, "📞 General Helpline")

    # 6️⃣ Determine Priority
    if points >= 12:
        priority = "Critical"
    elif points >= 7:
        priority = "High"
    else:
        priority = "Moderate"

    return {
        "id": issue_counter["total"] + 1,
        "user_id": user_id,  # track which user submitted it
        "message": message,
        "emergency_type": emergency_type,
        "urgency": urgency,
        "location": location,
        "priority": priority,
        "department": department,
        "status": "Pending",
        "linked_id": None # Added for duplicate linking
    }

# --- Clustering Function for Admin View ---
def get_clustered_issues(issues):
    """Groups active issues into clusters based on semantic similarity."""
    if not model:
        return [{"topic_label": "Semantic Clustering Offline", "issues": issues}]

    # Filter out resolved issues for relevant clustering
    active_issues = [i for i in issues if i["status"] != "Resolved"]
    
    if not active_issues:
        # Case 1: No active issues (all are resolved or total is 0)
        return [] 
        
    if len(active_issues) == 1:
        # Case 2: Only one active issue. Return it as a single cluster.
        issue = active_issues[0]
        topic_label = f"Topic: {issue['emergency_type']} - {issue['message'][:50].strip()}..."
        return [{
            "topic_label": topic_label, 
            "issues": [issue]
        }]

    # Case 3: Two or more active issues (proceed with clustering)
    messages = [issue["message"] for issue in active_issues]
    
    # 1. Generate Embeddings
    embeddings = model.encode(messages)
    
    # 2. Perform Agglomerative Clustering
    Z = linkage(embeddings, method='ward')
    
    # 3. Form clusters using a distance cutoff (tune this based on your data)
    CLUSTER_DISTANCE_CUTOFF = 1.5 
    clusters = fcluster(Z, CLUSTER_DISTANCE_CUTOFF, criterion='distance')
    
    # 4. Group issues by cluster ID and assign a "Topic Label"
    clustered_data = {}
    for i, issue in enumerate(active_issues):
        cluster_id = int(clusters[i])
        
        if cluster_id not in clustered_data:
            clustered_data[cluster_id] = {
                "topic_id": cluster_id,
                "issues": []
            }
        clustered_data[cluster_id]["issues"].append(issue)

    # Set the Topic Label to the message of the highest priority report in the cluster
    for cluster in clustered_data.values():
         if cluster["issues"]:
             # Sort issues by priority for a meaningful topic label (Critical > High > Moderate > Duplicate)
             priority_map = {"Critical": 3, "High": 2, "Moderate": 1, "Duplicate": 0, "Pending": 1}
             sorted_issues = sorted(
                 cluster['issues'], 
                 key=lambda x: priority_map.get(x['priority'], 0), 
                 reverse=True
             )
             
             # Use the message of the highest priority issue as the topic label
             primary_issue = sorted_issues[0]
             cluster["topic_label"] = f"Topic: {primary_issue['emergency_type']} - {primary_issue['message'][:50].strip()}..."
             
    # Return a list of clusters
    return list(clustered_data.values())


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    msg = None
    # Assign session user id if not already set
    if "user_id" not in session:
        session["user_id"] = len(session) + 1  # simple user id

    user_id = session["user_id"]

    if request.method == "POST":
        message = request.form["message"].strip()
        duplicate_id = is_duplicate(message) 
        
        # Check for semantic duplicate
        if duplicate_id:
            # The message is a semantic duplicate. Link it to the original for clustering/admin view.
            msg = f"⚠️ Similar report already exists! (Linked to Issue ID: {duplicate_id})"
            result = analyze_message(message, user_id)
            result["status"] = "Duplicate"
            result["priority"] = "Low" # Duplicates can be de-prioritized
            result["linked_id"] = duplicate_id
            issues.append(result)
            issue_counter["total"] += 1
        else:
            result = analyze_message(message, user_id)
            issues.append(result)
            issue_counter["total"] += 1
            issue_counter["pending"] += 1
            msg = "✅ Your issue has been submitted successfully!"

    # Filter only issues reported by this user
    user_issues = [i for i in issues if i["user_id"] == user_id]

    return render_template("user.html", msg=msg, user_issues=user_issues, stats=issue_counter)

@app.route("/admin")
def admin():
    # Pass the clustered data to the template
    clustered_issues = get_clustered_issues(issues)

    return render_template("admin.html", 
                           issues=issues, 
                           stats=issue_counter,
                           clustered_issues=clustered_issues) # New variable

@app.route("/update/<int:issue_id>/<action>")
def update_issue(issue_id, action):
    for issue in issues:
        if issue["id"] == issue_id:
            if action == "process":
                if issue["status"] == "Pending":
                    issue["status"] = "Under Process"
                    issue_counter["pending"] -= 1
                    issue_counter["under_process"] += 1
            elif action == "resolve":
                if issue["status"] in ["Pending", "Under Process", "Duplicate"]:
                    # Adjust counters based on original status
                    if issue["status"] == "Pending":
                        issue_counter["pending"] -= 1
                    elif issue["status"] == "Under Process":
                        issue_counter["under_process"] -= 1
                    
                    # Duplicates do not affect pending/process counts, so just update status
                    
                    issue["status"] = "Resolved"
                    issue_counter["resolved"] += 1
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)'''

















from flask import Flask, render_template, request, redirect, url_for, session
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # New for Semantic Similarity
from scipy.cluster.hierarchy import linkage, fcluster # New for Clustering
import numpy as np                                   # New for numerical operations

app = Flask(__name__)
app.secret_key = "secret123"  # needed for session tracking

# NLP model
nlp = spacy.load("en_core_web_sm")

# NEW: Sentence Transformer model for Semantic Similarity
# Using 'all-MiniLM-L6-v2' for a good balance of speed and performance
# Note: This requires the 'sentence-transformers' package
try:
    model = SentenceTransformer('all-MiniLM-L6-v2') 
except OSError:
    print("Warning: SentenceTransformer model failed to load. Ensure 'all-MiniLM-L6-v2' is downloaded.")
    model = None


# Storage for crisis issues
issues = []
issue_counter = {"total": 0, "pending": 0, "under_process": 0, "resolved": 0}

# Departments mapping
departments_map = {
    "Fire": "🔥 Fire Department",
    "Flood": "💧 Disaster Management",
    "Accident": "🚑 Ambulance Services",
    "Earthquake": "🏢 National Disaster Team",
    "Storm": "⛈️ Weather Response",
    "Landslide": "⛰️ Geological Response"
}

# TF-IDF vectorizer (kept, but we use semantic duplicate check)
vectorizer = TfidfVectorizer()

# --- Advanced Semantic Duplicate Check ---
def is_duplicate_semantic(new_message):
    """Checks for semantic similarity against active reports."""
    if not model:
        print("Semantic model unavailable. Skipping semantic check.")
        return False

    if not issues:
        return False
        
    # Exclude already resolved or duplicate issues from the comparison set
    # Duplicates are checked against the original (non-duplicate) reports
    comparable_issues = [i for i in issues if i["status"] not in ["Resolved", "Duplicate"]]
    if not comparable_issues:
        return False

    messages = [issue["message"] for issue in comparable_issues]
    
    # 1. Generate Embeddings for all relevant messages (existing + new)
    all_messages = messages + [new_message]
    embeddings = model.encode(all_messages)
    
    new_embedding = embeddings[-1]
    existing_embeddings = embeddings[:-1]
    
    # 2. Calculate Cosine Similarity
    # np.newaxis ensures compatible dimensions for cosine_similarity
    sim_scores = cosine_similarity(new_embedding[np.newaxis, :], existing_embeddings)[0]
    
    # 3. Check against a semantic similarity threshold (tune this value)
    SEMANTIC_THRESHOLD = 0.75 
    
    if np.any(sim_scores > SEMANTIC_THRESHOLD):
        # Return the ID of the detected original issue
        duplicate_index = np.argmax(sim_scores)
        return comparable_issues[duplicate_index]["id"]

    return False

# Alias the new function
def is_duplicate(new_message):
    return is_duplicate_semantic(new_message)


def analyze_message(message, user_id):
    """Performs NLP analysis on the message to determine type, urgency, and priority."""
    doc = nlp(message)

    # Initialize points
    points = 0

    # 1️⃣ Emergency type points
    emergency_type = "Unknown"
    message_lower = message.lower()
    
    # Improved type detection (matches departments_map keys)
    if "fire" in message_lower:
        emergency_type = "Fire"
        points += 5
    elif "flood" in message_lower:
        emergency_type = "Flood"
        points += 4
    elif "accident" in message_lower:
        emergency_type = "Accident"
        points += 3
    elif "earthquake" in message_lower:
        emergency_type = "Earthquake"
        points += 6
    elif "storm" in message_lower or "hurricane" in message_lower:
        emergency_type = "Storm"
        points += 3
    elif "landslide" in message_lower or "mudslide" in message_lower:
        emergency_type = "Landslide"
        points += 5


    # 2️⃣ Urgency points
    urgency_words = ["urgent", "emergency", "immediate", "help", "dying", "danger"]
    if any(word in message_lower for word in urgency_words):
        urgency = "High"
        points += 4
    else:
        urgency = "Low"
        points += 1

    # 3️⃣ Location points
    location = "Unknown"
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LOC"]:
            location = ent.text
            # Higher point for sensitive locations
            if any(word in location.lower() for word in ["school", "hospital", "market", "power plant"]):
                points += 3
            else:
                points += 1
            break

    # 4️⃣ Severity keywords
    severity_keywords = ["injured", "death", "fatal", "major", "critical", "collapsed", "trapped"]
    if any(word in message_lower for word in severity_keywords):
        points += 5

    # 5️⃣ Assign department
    department = departments_map.get(emergency_type, "📞 General Helpline")

    # 6️⃣ Determine Priority
    if points >= 12:
        priority = "Critical"
    elif points >= 7:
        priority = "High"
    else:
        priority = "Moderate"

    return {
        "id": issue_counter["total"] + 1,
        "user_id": user_id,  # track which user submitted it
        "message": message,
        "emergency_type": emergency_type,
        "urgency": urgency,
        "location": location,
        "priority": priority,
        "department": department,
        "status": "Pending",
        "linked_id": None # Added for duplicate linking
    }

# --- Clustering Function for Admin View ---
def get_clustered_issues(issues):
    """Groups active issues into clusters based on semantic similarity."""
    if not model:
        return [{"topic_label": "Semantic Clustering Offline", "issues": issues}]

    # Filter out resolved issues for relevant clustering
    active_issues = [i for i in issues if i["status"] != "Resolved"]
    
    if not active_issues:
        # Case 1: No active issues (all are resolved or total is 0)
        return [] 
        
    if len(active_issues) == 1:
        # Case 2: Only one active issue. Return it as a single cluster.
        issue = active_issues[0]
        topic_label = f"Topic: {issue['emergency_type']} - {issue['message'][:50].strip()}..."
        return [{
            "topic_label": topic_label, 
            "issues": [issue]
        }]

    # Case 3: Two or more active issues (proceed with clustering)
    messages = [i["message"] for i in active_issues]
    
    # 1. Generate Embeddings
    embeddings = model.encode(messages)
    
    # 2. Perform Agglomerative Clustering
    Z = linkage(embeddings, method='ward')
    
    # 3. Form clusters using a distance cutoff (tune this based on your data)
    CLUSTER_DISTANCE_CUTOFF = 1.5 
    clusters = fcluster(Z, CLUSTER_DISTANCE_CUTOFF, criterion='distance')
    
    # 4. Group issues by cluster ID and assign a "Topic Label"
    clustered_data = {}
    for i, issue in enumerate(active_issues):
        cluster_id = int(clusters[i])
        
        if cluster_id not in clustered_data:
            clustered_data[cluster_id] = {
                "topic_id": cluster_id,
                "issues": []
            }
        clustered_data[cluster_id]["issues"].append(issue)

    # Set the Topic Label to the message of the highest priority report in the cluster
    for cluster in clustered_data.values():
         if cluster["issues"]:
             # Sort issues by priority for a meaningful topic label (Critical > High > Moderate > Duplicate)
             priority_map = {"Critical": 3, "High": 2, "Moderate": 1, "Duplicate": 0, "Pending": 1, "Under Process": 2} # Added Under Process
             sorted_issues = sorted(
                 cluster['issues'], 
                 key=lambda x: priority_map.get(x['priority'], 0), 
                 reverse=True
             )
             
             # Use the message of the highest priority issue as the topic label
             primary_issue = sorted_issues[0]
             cluster["topic_label"] = f"Topic: {primary_issue['emergency_type']} - {primary_issue['message'][:50].strip()}..."
             
    # Return a list of clusters
    return list(clustered_data.values())


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    msg = None
    # Assign session user id if not already set
    if "user_id" not in session:
        session["user_id"] = len(session) + 1  # simple user id

    user_id = session["user_id"]

    if request.method == "POST":
        message = request.form["message"].strip()
        duplicate_id = is_duplicate(message) 
        
        # Check for semantic duplicate
        if duplicate_id:
            # The message is a semantic duplicate. Link it to the original for clustering/admin view.
            msg = f"⚠️ Similar report already exists! (Linked to Issue ID: {duplicate_id})"
            result = analyze_message(message, user_id)
            result["status"] = "Duplicate"
            result["priority"] = "Low" # Duplicates can be de-prioritized
            result["linked_id"] = duplicate_id
            issues.append(result)
            issue_counter["total"] += 1
        else:
            result = analyze_message(message, user_id)
            issues.append(result)
            issue_counter["total"] += 1
            issue_counter["pending"] += 1
            msg = "✅ Your issue has been submitted successfully!"

    # Filter only issues reported by this user
    user_issues = [i for i in issues if i["user_id"] == user_id]

    return render_template("user.html", msg=msg, user_issues=user_issues, stats=issue_counter)

@app.route("/admin")
def admin():
    # Pass the clustered data to the template
    clustered_issues = get_clustered_issues(issues)

    return render_template("admin.html", 
                           issues=issues, 
                           stats=issue_counter,
                           clustered_issues=clustered_issues) # New variable

@app.route("/update/<int:issue_id>/<action>")
def update_issue(issue_id, action):
    # Find the issue being updated
    current_issue = next((issue for issue in issues if issue["id"] == issue_id), None)
    
    if not current_issue:
        return redirect(url_for("admin"))

    if action == "process":
        # Check if the current issue is Pending (only move Pending reports to Under Process)
        if current_issue["status"] == "Pending":
            current_issue["status"] = "Under Process"
            issue_counter["pending"] -= 1
            issue_counter["under_process"] += 1

            # --- NEW LOGIC: Update related duplicates ---
            # 1. Determine the ID of the 'original' report for this cluster.
            original_id = current_issue.get("linked_id") or current_issue["id"]
            
            # 2. Find the original report (or the report that was just processed)
            primary_report = next((issue for issue in issues if issue["id"] == original_id), current_issue)
            
            # 3. Get the department and status to propagate
            propagated_status = primary_report["status"]
            propagated_department = primary_report["department"]

            for issue in issues:
                # Update all reports that are duplicates of the original
                if issue.get("linked_id") == original_id and issue["status"] == "Duplicate":
                    issue["status"] = propagated_status + " (Linked)"
                    issue["department"] = propagated_department # <-- Propagate the department

    elif action == "resolve":
        # Resolve logic remains the same for the target issue
        if current_issue["status"] in ["Pending", "Under Process", "Duplicate", "Under Process (Linked)"]:
            
            # Adjust counters based on original status before resolving
            if current_issue["status"] == "Pending":
                issue_counter["pending"] -= 1
            elif current_issue["status"] == "Under Process":
                issue_counter["under_process"] -= 1
            
            # Resolve the current issue
            current_issue["status"] = "Resolved"
            issue_counter["resolved"] += 1
            
            # --- NEW LOGIC: Resolve all related reports ---
            # Determine the ID of the 'original' report for this cluster.
            original_id = current_issue.get("linked_id") or current_issue["id"]

            for issue in issues:
                # Resolve the original report AND all duplicates linked to it
                if issue["id"] == original_id or issue.get("linked_id") == original_id:
                    # Only resolve if not already resolved, and update counters if applicable
                    if issue["status"] != "Resolved":
                        if issue["status"] == "Pending":
                            issue_counter["pending"] -= 1
                        elif issue["status"] == "Under Process":
                            issue_counter["under_process"] -= 1

                        issue["status"] = "Resolved"
                        issue_counter["resolved"] += 1

    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)









