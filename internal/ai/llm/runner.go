package llm

import (
	"bufio"
	"database/sql"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/platform/discovery"
	"github.com/golangast/gollemer/internal/platform/sqlite_db"
	"github.com/golangast/gollemer/internal/platform/ui"
)

type Runner struct {
	Mascot         *ui.Mascot
	ProjectRoot    string
	DB             *sql.DB
	KB             *KnowledgeBase
	Reader         *bufio.Reader
	W2V            *word2vec.SimpleWord2Vec
	IntentModel    *moe.IntentMoE
	Client         *GollemerMoEClient
	Resolver       *HybridIntentResolver
	CommandHistory []string
	SessionState   ConversationState
	TutorialState  TutorialState
	InMenuMode     bool
}

func NewRunner() (*Runner, error) {
	mascot := ui.NewMascot()

	projectRoot, err := FindProjectRoot()
	if err != nil {
		log.Printf("❌ Failed to find project root: %v", err)
		return nil, fmt.Errorf("failed to find project root: %v", err)
	}
	log.Printf("✅ Detected Project Root: %s", projectRoot)

	absoluteLastDirConfigPath = filepath.Join(projectRoot, "last_dir.txt")

	dbFileName := "data/db/gollemer.db"
	db, err := sqlite_db.InitDB(dbFileName)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize database: %v", err)
	}

	kb := LoadKnowledgeBase()

	r := &Runner{
		Mascot:      mascot,
		ProjectRoot: projectRoot,
		DB:          db,
		KB:          kb,
		Reader:      bufio.NewReader(os.Stdin),
	}
	log.Printf("🔧 Initializing Gollemer Runner (Root: %s)", r.ProjectRoot)

	return r, nil
}

func (r *Runner) Init() {
	if r.KB.FirstRun {
		r.Mascot.WelcomeSequence()
		printIntro()
		r.KB.FirstRun = false
		r.KB.Save()
	}

	// Load last directory on startup
	lastDir, err := loadLastDirectory()
	if err == nil {
		_ = os.Chdir(lastDir)
	}

	r.initModels()

	// Load social model if available
	var socialModel *moe.IntentMoE
	socialModelPath := filepath.Join(r.ProjectRoot, "data/models/gob_models/moe_social_model.gob")
	if loaded, err := moe.LoadIntentMoEModelFromGOB(socialModelPath); err == nil {
		socialModel = loaded
		log.Printf("✅ Loaded social-only model from %s", socialModelPath)
	} else {
		log.Printf("⚠️  Social model not found at %s (training with -train-social to create it)", socialModelPath)
	}

	// Initialize Intent Resolver
	r.Client = &GollemerMoEClient{KB: r.KB, Model: r.IntentModel, SocialModel: socialModel, W2V: r.W2V}
	if r.W2V != nil && r.W2V.VocabSize > 0 {
		r.Client.CommandAnchors = map[string][]float64{
			"create_file":    r.Client.getSentenceEmbedding("create a new file or scaffold code"),
			"START_TRAINING": r.Client.getSentenceEmbedding("start training the model or run the training cycle"),
			"MENU":           r.Client.getSentenceEmbedding("show the main menu with options"),
		}
	}
	r.Client.LoadChatBank(filepath.Join(r.ProjectRoot, "data/training/trainingdata/conversing.csv"))
	r.Resolver = NewHybridIntentResolver(r.Client)

	// Initialize Tutorial State
	if step, active := sqlite_db.GetCurrentStep(r.DB); active {
		r.TutorialState.Step = step
		r.TutorialState.Active = active
	}

	r.Mascot.AuditProjectSize(r.ProjectRoot)
}

func (r *Runner) initModels() {
	// Try to load Word2Vec model
	if r.KB.ModelConfig.Word2VecPath != "" {
		loadedW2V, err := word2vec.LoadModel(r.KB.ModelConfig.Word2VecPath)
		if err == nil {
			r.W2V = loadedW2V
		} else {
			log.Printf("⚠️  Failed to load Word2Vec: %v", err)
			// Create a default/dummy model
			r.W2V = &word2vec.SimpleWord2Vec{
				Vocabulary:  make(map[string]int),
				VocabSize:   0,
				VectorSize:  64,
				WordVectors: make(map[int][]float64),
			}
			basics := []string{"create", "webserver", "UNK", "<pad>", "<s>", "</s>"}
			for i, w := range basics {
				r.W2V.Vocabulary[w] = i
				r.W2V.WordVectors[i] = make([]float64, 64)
			}
			r.W2V.VocabSize = len(basics)
		}
	}

	vocabSize := 2547   // Final vocab size from TrainChat
	embeddingDim := 768 // Match Transformer training
	if r.W2V != nil {
		vocabSize = r.W2V.VocabSize
		// If W2V was loaded, it may be 64d but the model should still be 768d
	}

	// Try to load trained MoE model - prioritize the latest trained model or checkpoint
	if r.KB.ModelConfig.MoEPath != "" {
		paths := []string{
			filepath.Join(r.ProjectRoot, "data/models/checkpoints/latest_periodic.gob"),
			filepath.Join(r.ProjectRoot, "data/models/gob_models/moe_classification_model.gob"),
			filepath.Join(r.ProjectRoot, "data/models/gob_models/golden_checkpoint.gob"),
			filepath.Join(r.ProjectRoot, r.KB.ModelConfig.MoEPath),
		}

		for _, p := range paths {
			if r.IntentModel != nil {
				break
			}
			if _, err := os.Stat(p); err != nil {
				continue
			}

			// Try as checkpoint first (compressed)
			ckpt, err := moe.LoadIntentMoECheckpoint(p)
			if err == nil && ckpt != nil && ckpt.Model != nil {
				r.IntentModel = ckpt.Model
				log.Printf("✅ Success: Loaded 768d MoE model from compressed Checkpoint: %s (Steps: %d)", filepath.Base(p), ckpt.StepCount)
				break
			} else if err != nil && !strings.Contains(err.Error(), "invalid header") {
				log.Printf("❌ Error loading checkpoint %s: %v", filepath.Base(p), err)
			}

			// Try as raw GOB model (uncompressed)
			loadedModel, err := moe.LoadIntentMoEModelFromGOB(p)
			if err == nil && loadedModel != nil {
				r.IntentModel = loadedModel
				log.Printf("✅ Success: Loaded 768d MoE model from raw GOB: %s", filepath.Base(p))
				break
			} else if err != nil {
				log.Printf("⚠️  Candidate %s failed (GOB): %v", filepath.Base(p), err)
			}
		}

		if r.IntentModel == nil {
			log.Printf("⚠️  No trained MoE weights could be loaded from any of the %d candidates.", len(paths))
		}
	}

	if r.IntentModel == nil {
		intentModel, err := moe.NewHybridIntentMoE(
			vocabSize,
			embeddingDim,
			4,    // numExperts
			100,  // parentVocabSize
			100,  // childVocabSize
			1000, // sentenceVocabSize
			4,    // maxAttentionHeads
			r.W2V,
		)
		if err == nil {
			r.IntentModel = intentModel
		} else {
			log.Printf("❌ Failed to initialize new MoE model: %v", err)
		}
	}

	if r.IntentModel != nil && r.IntentModel.SentenceVocab == nil {
		vocabPath := filepath.Join(r.ProjectRoot, r.KB.ModelConfig.SemanticVocabPath)
		if v, err := mainvocab.LoadVocabulary(vocabPath); err == nil {
			r.IntentModel.SentenceVocab = v
		} else {
			v := mainvocab.NewVocabulary()
			tokens := []string{"create", "webserver", "handler", "page", "database", "<s>", "</s>"}
			for _, t := range tokens {
				v.AddToken(t)
			}
			v.BosID = v.GetTokenID("<s>")
			v.EosID = v.GetTokenID("</s>")
			if r.W2V != nil {
				for w := range r.W2V.Vocabulary {
					v.AddToken(w)
				}
			}
			r.IntentModel.SentenceVocab = v
		}
	}
}

func (r *Runner) Run() {
	projectCtx := discovery.ScanProject()
	lastDirState := &discovery.FolderState{}

	initialMsg := discovery.GetExpertAdvice(projectCtx)
	if initialMsg == "Ready to code! What's the focus for this session?" {
		// Suppress redundant greeting if we already did WelcomeSequence
		initialMsg = ""
	}
	if initialMsg != "" {
		r.Mascot.Speak(ui.MoodHappy, initialMsg)
	}

	for {
		r.Mascot.WellnessCheck()
		r.SessionState.JustConfirmed = false

		if r.InMenuMode {
			r.handleMenu()
			continue
		}

		// Periodic Proactive Scan
		if discovery.QuickCheck(".", lastDirState) {
			newCtx := discovery.ScanProject()
			if newCtx.IsGollemer != projectCtx.IsGollemer || newCtx.HasModel != projectCtx.HasModel {
				projectCtx = newCtx
				advice := discovery.GetExpertAdvice(projectCtx)
				r.Mascot.Speak(ui.MoodThink, "Wait, I noticed something changed! "+advice)
			}
		}

		mood := ui.MoodIdle
		if r.SessionState.WaitingForConfirm {
			mood = ui.MoodWaiting
		}
		r.Mascot.ShowMascot(mood)
		query, _ := r.Reader.ReadString('\n')
		query = strings.TrimSpace(query)

		// --- Strip Mascot Prefix if present (e.g., copied from logs or mimicry) ---
		if strings.HasPrefix(query, "/") && strings.Contains(query, "/ >") {
			parts := strings.SplitN(query, "/ >", 2)
			if len(parts) == 2 {
				query = strings.TrimSpace(parts[1])
			}
		}

		if query == "" {
			continue
		}

		r.CommandHistory = append(r.CommandHistory, query)
		r.handleInput(query)
	}
}

func (r *Runner) handleInput(query string) {
	if query == "menu" {
		r.InMenuMode = true
		return
	}
	if query == "exit" {
		r.Mascot.Shutdown(r.ProjectRoot)
		os.Exit(0)
	}
	if query == "clear" {
		cmd := exec.Command("clear")
		cmd.Stdout = os.Stdout
		cmd.Run()
		return
	}
	if query == "doctor" || query == "fix system" {
		r.Client.RunDoctor(r.Mascot)
		return
	}
	if query == "audit" || query == "scan project" {
		r.Client.RunAudit(r.Mascot)
		return
	}
	if query == "commit" || query == "push changes" {
		r.Client.MascotCommit(r.Mascot, r.Reader)
		return
	}
	if query == "profile" || query == "show profile" || query == "project status" {
		name := detectWebserverName(r.ProjectRoot)
		if name == "" {
			name = filepath.Base(r.ProjectRoot)
		}
		cwd, _ := os.Getwd()
		_ = ScanAndSaveProfile(name, cwd, r.DB)
		ShowProjectProfile(name, r.DB, r.Mascot)
		return
	}
	if query == "watch" || query == "monitor" || query == "guard" {
		r.Client.StartBackgroundWatcher(r.Mascot, r.ProjectRoot)
		r.Mascot.Speak(ui.MoodHappy, "I'm on guard duty! I'll watch the workspace for changes.")
		return
	}
	if query == "tutorial" || query == "start tutorial" {
		if r.TutorialState.Active {
			r.Mascot.Speak(ui.MoodHappy, fmt.Sprintf("Hi, welcome to gollemer! Resuming tutorial from Step %d. (Type 'reset' to start over)", r.TutorialState.Step))
			return
		}
		r.TutorialState.Step = 1
		r.TutorialState.Active = true
		sqlite_db.SyncStep(r.DB, 1, true)
		loc, _ := r.Mascot.CalculateProjectSize(r.ProjectRoot)
		r.Mascot.DrawHUD(1, 4, loc)
		r.Mascot.Speak(ui.Happy, "Hi, welcome to gollemer! Tutorial started.\n\nStep 1: Create a folder. Try: 'create folder mynews'")
		return
	}
	if query == "restart tutorial" || query == "restart" || query == "reset tutorial" || query == "reset" || query == "clear tutorial" || query == "clear progress" {
		r.TutorialState.Step = 1
		r.TutorialState.Active = true
		sqlite_db.SyncStep(r.DB, 1, true)
		loc, _ := r.Mascot.CalculateProjectSize(r.ProjectRoot)
		r.Mascot.DrawHUD(1, 4, loc)
		r.Mascot.Speak(ui.Happy, "Hi, welcome to gollemer! Tutorial cleared and restarted from Step 1.\n\nStep 1: Create a folder. Try: 'create folder mynews'")
		return
	}
	r.handleInteractiveQuery(query)
}
