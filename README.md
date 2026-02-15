# Gollemer

Gollemer is an intelligent coding assistant and project orchestrator designed to help you build Go applications, specifically focusing on web servers and WASM frontends.

## Interactive Menu System

Gollemer includes a comprehensive interactive menu to guide you through project creation, management, and AI training. You can access this menu by typing `menu` in the Gollemer shell.

### Main Menu Options

#### 1. 🚀 Start a New Project (Webserver)
Initializes a new Go webserver project.
- **Action:** Prompts for a project name.
- **Result:** Creates a directory with `main.go` (including SQLite setup and a basic handler) and initializes `go.mod`.

#### 2. ➕ Add a Feature
Adds components to your existing project.
- **a. Handler (Backend logic):** Creates a new Go handler function and registers it in your `main.go`.
- **b. Page (Frontend view):** Generates a WASM-compatible Go page using the internal UI framework and registers it in the WASM router.
- **c. Database (Storage):** Creates a new SQLite database file or adds tables if fields are specified.

#### 3. 📂 Manage Files
Basic file system operations within your project context.
- **a. Create File:** Creates a new file (can use templates if learned).
- **b. Create Folder:** Creates a new directory.

#### 4. ▶️ Run Project
Builds and runs your application.
- **Action:** Prompts for the webserver name (defaults to current context).
- **Result:** Compiles the Go code, builds any WASM components, and starts the server.

#### 5. 🧠 Learning & Training
Manage the AI and learning capabilities of Gollemer.
- **1. Show Learning Status:** Displays loaded models, vocabulary sizes, and the current learning source path.
- **2. Change Learning Source:** Updates the directory Gollemer scans for templates and code patterns.
- **3. Teach New Object Word:** Manually adds a new noun/object to the knowledge base.
- **4. Run Training Commands:** Access advanced model training and visualization tools:
    1. Train Word2Vec
    2. Train MoE (Mixture of Experts)
    3. Train Intent Classifier
    4. Train NER (Named Entity Recognition)
    5. Custom Training Module
    6. Visualize Neural Network
    7. Visualize Word2Vec Model
    8. Search Word Neighbors
    9. Visualize Word Relationship
    10. Visualize Word Distribution (2D Plot)
    11. Inspect Model Weights
    12. Visualize Attention Mechanism
    13. Visualize Word Similarity (One vs List)

#### 6. 🎓 Tutorial
Starts an interactive, step-by-step tutorial that guides you through creating a folder, a file, a webserver, and running it.

#### 7. ❓ Help
Displays the general help text with command syntax and examples.

#### 8. 🚪 Exit
Closes the Gollemer application.