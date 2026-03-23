package llm

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func (r *Runner) handleMenu() {
	r.InMenuMode = true
	fmt.Println("\n--- 📋 Main Menu ---")
	fmt.Println("1. 🚀 Start a New Project (Webserver)")
	fmt.Println("2. ➕ Add a Feature (Handler, Page, Database)")
	fmt.Println("3. 📂 Manage Files (Create, Delete, Move)")
	fmt.Println("4. ▶️  Run Project")
	fmt.Println("5. 🧠 Learning & Training")
	fmt.Println("6. 🎓 Tutorial")
	fmt.Println("7. ❓ Help")
	fmt.Println("8. 🚪 Exit")
	fmt.Println("9. 💬 Interactive Mode")
	fmt.Println("10. ⚙️ Model Configuration")
	fmt.Println("11. 🧪 Test Model")
	fmt.Println("12. 🔍 Run Audit")
	fmt.Println("13. 📄 Project Profile")

	cwd, _ := os.Getwd()
	if cwd != "" {
		fmt.Printf("\n💡 Tip: You're currently in %s. I can help you scaffold a new file here!\n", filepath.Base(cwd))
	} else {
		fmt.Println("\n💡 Tip: You haven't initialized a project yet. Type '1' or 'create' to start a new webserver.")
	}

	fmt.Print("\n/ʕ◔ϖ◔ʔ/ > Select an option (1-13): ")

	choice, _ := r.Reader.ReadString('\n')
	choice = strings.TrimSpace(choice)

	r.processMenuChoice(choice)
}

func (r *Runner) processMenuChoice(choice string) {
	var query string
	switch choice {
	case "11":
		r.testModel()
		return
	case "13":
		name := detectWebserverName(r.ProjectRoot)
		if name == "" {
			name = filepath.Base(r.ProjectRoot)
		}
		cwd, _ := os.Getwd()
		_ = ScanAndSaveProfile(name, cwd, r.DB)
		ShowProjectProfile(name, r.DB, r.Mascot)
		return
	case "12":
		r.Client.RunAudit(r.Mascot)
		return
	case "1":
		archChoice := r.Mascot.AskArchitecture()
		r.Mascot.ScaffoldProject(archChoice)
		return
	case "2":
		query = r.menuAddFeature()
	case "3":
		query = r.menuManageFiles()
	case "4":
		query = r.menuRunProject()
	case "5":
		r.menuLearningAndTraining()
		return
	case "6":
		r.Mascot.RunMoETutorial()
		return
	case "7":
		query = "help"
	case "8":
		os.Exit(0)
	case "9":
		r.InMenuMode = false
		fmt.Println("Returning to interactive mode...")
		return
	case "10":
		r.menuModelConfiguration()
		return
	default:
		fmt.Println("Invalid option.")
		return
	}

	if query != "" {
		fmt.Printf("Executing: %s\n", query)
		r.handleInteractiveQuery(query)
	}
}

func (r *Runner) testModel() {
	fmt.Println("\n--- 🧪 Test Model ---")
	fmt.Print("Enter a query to test the model: ")
	testQuery, _ := r.Reader.ReadString('\n')
	testQuery = strings.TrimSpace(testQuery)

	if testQuery != "" {
		intentData := r.Resolver.Resolve(testQuery, nil)
		fmt.Printf("Predicted Intent: %s (Confidence: %.2f)\n", intentData.Intent, intentData.Confidence)
		fmt.Println("Extracted Entities:")
		for k, v := range intentData.Parameters {
			fmt.Printf("  %s: %v\n", k, v)
		}
		if len(intentData.Missing) > 0 {
			fmt.Printf("Missing Parameters: %v\n", intentData.Missing)
		}
	} else {
		fmt.Println("No query entered.")
	}
}

func (r *Runner) menuAddFeature() string {
	fmt.Println("\nWhat do you want to add?")
	fmt.Println("a. Handler (Backend logic)")
	fmt.Println("b. Page (Frontend view)")
	fmt.Println("c. Database (Storage)")
	fmt.Print("Select (a/b/c): ")
	sub, _ := r.Reader.ReadString('\n')
	sub = strings.TrimSpace(sub)
	switch sub {
	case "a":
		fmt.Print("Handler Name: ")
		n, _ := r.Reader.ReadString('\n')
		return "create handler " + strings.TrimSpace(n)
	case "b":
		fmt.Print("Page Name: ")
		n, _ := r.Reader.ReadString('\n')
		return "create page " + strings.TrimSpace(n)
	case "c":
		fmt.Print("Database Name: ")
		n, _ := r.Reader.ReadString('\n')
		return "create database " + strings.TrimSpace(n)
	}
	return ""
}

func (r *Runner) menuManageFiles() string {
	fmt.Println("\nWhat file operation?")
	fmt.Println("a. Create File")
	fmt.Println("b. Create Folder")
	fmt.Print("Select (a/b): ")
	sub, _ := r.Reader.ReadString('\n')
	sub = strings.TrimSpace(sub)
	switch sub {
	case "a":
		fmt.Print("File Name: ")
		n, _ := r.Reader.ReadString('\n')
		return "create file " + strings.TrimSpace(n)
	case "b":
		fmt.Print("Folder Name: ")
		n, _ := r.Reader.ReadString('\n')
		return "create folder " + strings.TrimSpace(n)
	}
	return ""
}

func (r *Runner) menuRunProject() string {
	fmt.Print("Enter webserver name to run (or press enter for current): ")
	n, _ := r.Reader.ReadString('\n')
	n = strings.TrimSpace(n)
	if n != "" {
		return "run webserver " + n
	}
	return "run webserver"
}

func (r *Runner) menuLearningAndTraining() {
	fmt.Println("\n--- 🧠 Learning & Training ---")
	if r.KB.LearningPath != "" {
		fmt.Printf("Current Learning Path: %s\n", r.KB.LearningPath)
	}
	fmt.Println("1. Show Learning Status (Data & Vocab)")
	fmt.Println("2. Change Learning Source (Folder)")
	fmt.Println("3. Teach New Object Word")
	fmt.Println("4. Run Training Commands")
	fmt.Print("Select (1-4): ")
	sub, _ := r.Reader.ReadString('\n')
	sub = strings.TrimSpace(sub)

	switch sub {
	case "1":
		r.showLearningStatus()
	case "2":
		fmt.Print("Enter path to learning folder (e.g., ./templates): ")
		path, _ := r.Reader.ReadString('\n')
		r.handleInteractiveQuery("learn from " + strings.TrimSpace(path))
	case "3":
		fmt.Print("Enter object name to learn: ")
		obj, _ := r.Reader.ReadString('\n')
		r.handleInteractiveQuery("learn object " + strings.TrimSpace(obj))
	case "4":
		r.menuRunTraining()
	}
}

func (r *Runner) showLearningStatus() {
	fmt.Println("\n--- 📊 Learning Status ---")
	fmt.Printf("Knowledge Base: %s\n", kbFilename)
	fmt.Printf("Templates Source: %s\n", r.KB.LearningPath)

	fmt.Println("\n[Training Data & Vocab]")
	checkPath := func(name, path string) {
		fullPath := filepath.Join(r.ProjectRoot, path)
		if _, err := os.Stat(fullPath); err == nil {
			fmt.Printf("  ✅ %s: %s\n", name, path)
		} else {
			fmt.Printf("  ❌ %s: %s (Not found)\n", name, path)
		}
	}
	checkPath("Word2Vec", r.KB.ModelConfig.Word2VecPath)
	checkPath("MoE", r.KB.ModelConfig.MoEPath)
	checkPath("NER", r.KB.ModelConfig.NERPath)
	checkPath("Query Vocab", r.KB.ModelConfig.QueryVocabPath)
	checkPath("Semantic Vocab", r.KB.ModelConfig.SemanticVocabPath)

	qVocabPath := filepath.Join(r.ProjectRoot, r.KB.ModelConfig.QueryVocabPath)
	if qVocab, err := mainvocab.LoadVocabulary(qVocabPath); err == nil {
		fmt.Printf("\n  📝 Query Vocabulary: %d words\n", len(qVocab.WordToToken))
	}
	sVocabPath := filepath.Join(r.ProjectRoot, r.KB.ModelConfig.SemanticVocabPath)
	if sVocab, err := mainvocab.LoadVocabulary(sVocabPath); err == nil {
		fmt.Printf("  📝 Semantic Output Vocabulary: %d tokens\n", len(sVocab.WordToToken))
	}
}

func (r *Runner) menuRunTraining() {
	fmt.Println("\n--- 🏋️ Run Training ---")
	fmt.Println("1. Train Word2Vec")
	fmt.Println("2. Train MoE")
	fmt.Println("3. Train Intent Classifier")
	fmt.Println("4. Train NER")
	fmt.Println("5. Custom Training Module")
	fmt.Println("6. Visualize Neural Network")
	fmt.Println("7. Visualize Word2Vec Model")
	fmt.Println("8. Search Word Neighbors")
	fmt.Println("9. Visualize Word Relationship")
	fmt.Println("10. Visualize Word Distribution (2D Plot)")
	fmt.Println("11. Inspect Model Weights")
	fmt.Println("12. Visualize Attention Mechanism")
	fmt.Println("13. Visualize Word Similarity (One vs List)")
	fmt.Print("Select (1-13): ")
	trainSub, _ := r.Reader.ReadString('\n')
	trainSub = strings.TrimSpace(trainSub)

	switch trainSub {
	case "1", "2", "3", "4", "5":
		paths := map[string]string{"1": "cmd/train_word2vec", "2": "cmd/train_moe", "3": "cmd/train_intent_classifier", "4": "cmd/train_ner", "5": "cmd/train_custom"}
		r.executeTrainingCommand(paths[trainSub])
	case "6":
		r.visualizeNeuralNetwork()
	case "7":
		r.visualizeWord2Vec()
	case "8":
		r.searchWordNeighbors()
	case "9":
		r.visualizeWordRelationship()
	case "10":
		r.visualizeWordDistribution()
	case "11":
		r.inspectModelWeights()
	case "12":
		r.visualizeAttention()
	case "13":
		r.visualizeWordSimilarityList()
	}
}

func (r *Runner) executeTrainingCommand(cmdPath string) {
	fmt.Printf("Running %s...\n", cmdPath)
	c := exec.Command("go", "run", "./"+cmdPath)
	c.Dir = r.ProjectRoot
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	if err := c.Run(); err != nil {
		fmt.Printf("Error running training: %v\n", err)
	} else {
		fmt.Println("Training completed.")
	}
}

func (r *Runner) visualizeNeuralNetwork() {
	nn := r.IntentModel
	if nn == nil {
		fmt.Println("❌ Model not loaded.")
		return
	}
	fmt.Println("\n--- 🕸️ Neural Network Architecture ---")
	fmt.Println("")
	fmt.Println("       [ Input Query ]")
	fmt.Println("             ⬇")
	if nn.Embedding != nil {
		fmt.Printf("  ╔═══════════════════════╗\n  ║    Embedding Layer    ║  Dimension: %d\n  ╚═══════════════════════╝\n", nn.Embedding.DimModel)
	}
	fmt.Println("             ⬇")
	fmt.Println("  ╔═══════════════════════╗")
	encoderType := fmt.Sprintf("%T", nn.Encoder)
	fmt.Printf("  ║        Encoder        ║  Type: %s\n  ╚═══════════════════════╝\n", encoderType)
	fmt.Println("             ⬇")
	if nn.Decoder != nil && nn.Decoder.LSTM != nil {
		fmt.Printf("  ╔═══════════════════════╗\n  ║        Decoder        ║  Hidden Size: %d\n  ╚═══════════════════════╝\n", nn.Decoder.LSTM.HiddenSize)
	}
	fmt.Printf("             ⬇\n  ╔═══════════════════════╗\n  ║     Output Vocab      ║  Size: %d\n  ╚═══════════════════════╝\n", nn.SentenceVocabSize)
	fmt.Println("             ⬇\n      [ Predicted Intent ]\n")
}

func (r *Runner) visualizeWord2Vec() {
	if r.W2V == nil {
		fmt.Println("❌ Word2Vec model not loaded.")
		return
	}
	fmt.Println("\n--- 🔤 Word2Vec Model Visualization ---")
	fmt.Printf("Vector Size: %d\n", r.W2V.VectorSize)
	fmt.Printf("Vocabulary Count: %d words\n", len(r.W2V.Vocabulary))
	fmt.Println("---------------------------------------")
}

func (r *Runner) searchWordNeighbors() {
	if r.W2V == nil {
		fmt.Println("❌ Word2Vec model not loaded.")
		return
	}
	fmt.Print("Enter word: ")
	word, _ := r.Reader.ReadString('\n')
	word = strings.TrimSpace(word)
	targetIdx, ok := r.W2V.Vocabulary[word]
	if !ok {
		fmt.Printf("❌ Word '%s' not found.\n", word)
		return
	}
	targetVec := r.W2V.WordVectors[targetIdx]
	type res struct {
		word  string
		score float64
	}
	var results []res
	for w, idx := range r.W2V.Vocabulary {
		if w == word {
			continue
		}
		results = append(results, res{w, cosineSimilarity(targetVec, r.W2V.WordVectors[idx])})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	fmt.Printf("\n--- Neighbors for '%s' ---\n", word)
	for i := 0; i < min(len(results), 10); i++ {
		fmt.Printf("  %d. %s (%.4f)\n", i+1, results[i].word, results[i].score)
	}
}

func (r *Runner) visualizeWordRelationship() {
	if r.W2V == nil {
		return
	}
	fmt.Print("Word 1: ")
	w1, _ := r.Reader.ReadString('\n')
	fmt.Print("Word 2: ")
	w2, _ := r.Reader.ReadString('\n')
	w1, w2 = strings.TrimSpace(w1), strings.TrimSpace(w2)
	idx1, ok1 := r.W2V.Vocabulary[w1]
	idx2, ok2 := r.W2V.Vocabulary[w2]
	if ok1 && ok2 {
		fmt.Printf("Similarity: %.4f\n", cosineSimilarity(r.W2V.WordVectors[idx1], r.W2V.WordVectors[idx2]))
	}
}

func (r *Runner) visualizeWordDistribution() {
	if r.W2V == nil {
		return
	}
	fmt.Println("Generating distribution visualization...")
	limit := 500
	var words []string
	var vectors [][]float64
	count := 0
	for w, idx := range r.W2V.Vocabulary {
		if count >= limit {
			break
		}
		words = append(words, w)
		vectors = append(vectors, r.W2V.WordVectors[idx])
		count++
	}
	html := generateWordVizHTML(words, vectors)
	os.WriteFile("word_distribution.html", []byte(html), 0644)
	fmt.Println("✅ Generated word_distribution.html")
}

func (r *Runner) inspectModelWeights() {
	if r.IntentModel == nil {
		return
	}
	fmt.Println("\n--- ⚖️ Weight Inspection ---")
	fmt.Println("1. Embedding")
	fmt.Println("2. Encoder")
	fmt.Println("3. Decoder")
	sub, _ := r.Reader.ReadString('\n')
	sub = strings.TrimSpace(sub)
	switch sub {
	case "1":
		inspectStruct(r.IntentModel.Embedding, "  ")
	case "2":
		inspectStruct(r.IntentModel.Encoder, "  ")
	case "3":
		inspectStruct(r.IntentModel.Decoder, "  ")
	}
}

func (r *Runner) visualizeAttention() {
	if r.IntentModel != nil {
		findAndVisualizeAttention(r.IntentModel)
	}
}

func (r *Runner) visualizeWordSimilarityList() {
	if r.W2V == nil {
		return
	}
	fmt.Print("Target word: ")
	target, _ := r.Reader.ReadString('\n')
	fmt.Print("Compare to (comma-separated): ")
	list, _ := r.Reader.ReadString('\n')
	target = strings.TrimSpace(target)
	tIdx, ok := r.W2V.Vocabulary[target]
	if !ok {
		return
	}
	tVec := r.W2V.WordVectors[tIdx]
	for _, w := range strings.Split(list, ",") {
		w = strings.TrimSpace(w)
		if idx, ok := r.W2V.Vocabulary[w]; ok {
			fmt.Printf("  %s: %.4f\n", w, cosineSimilarity(tVec, r.W2V.WordVectors[idx]))
		}
	}
}

func (r *Runner) menuModelConfiguration() {
	fmt.Println("\n--- ⚙️ Model Configuration ---")
	fields := []string{"Word2Vec Model", "MoE Model", "Query Vocab", "Semantic Vocab", "NER Model"}
	paths := []*string{&r.KB.ModelConfig.Word2VecPath, &r.KB.ModelConfig.MoEPath, &r.KB.ModelConfig.QueryVocabPath, &r.KB.ModelConfig.SemanticVocabPath, &r.KB.ModelConfig.NERPath}

	for i, f := range fields {
		fmt.Printf("%d. %s: %s\n", i+1, f, *paths[i])
	}
	fmt.Println("6. Back")
	fmt.Print("Select (1-6): ")
	choice, _ := r.Reader.ReadString('\n')
	idx := strings.TrimSpace(choice)
	if idx == "6" {
		return
	}
	i := 0
	fmt.Sscanf(idx, "%d", &i)
	if i >= 1 && i <= 5 {
		fmt.Printf("Enter new path for %s: ", fields[i-1])
		newPath, _ := r.Reader.ReadString('\n')
		*paths[i-1] = strings.TrimSpace(newPath)
		r.KB.Save()
		fmt.Println("Saved.")
	}
}
