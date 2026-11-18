package chat

import (
	"strings"
	"testing"
)

func TestGuidedWalkLifecycle(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()
	guideActive, guideIdx, guideRoot = false, 0, ""

	oldProj, oldRoot := lastAnalyzeProject, unifiedProjectRoot
	defer func() { lastAnalyzeProject, unifiedProjectRoot = oldProj, oldRoot }()
	lastAnalyzeProject = nil
	unifiedProjectRoot = t.TempDir() // empty: no Go files; demo falls back in
	tourStops = nil

	if !handleBeginnerCommand("start", nil) {
		t.Fatal("'start' should begin the guided walk")
	}
	if !guideActive || lastAnalyzeProject == nil {
		t.Fatal("guide should be active with a project open")
	}
	root := guideRoot

	// Walk to the end with 'next' (curriculum + graduation).
	for i := 0; i < len(guideCurriculum); i++ {
		if !handleBeginnerCommand("next", nil) {
			t.Fatal("'next' during the walk should be handled")
		}
	}
	if guideActive {
		t.Fatal("guide should finish after the last stop")
	}
	if _, ok := friendLoad().Guided[root]; ok {
		t.Fatal("bookmark should clear at graduation")
	}

	// The Friend stop must have credited the Friend on the scoreboard.
	brainMu.Lock()
	friendOK := 0
	if st := brainStats["friend"]; st != nil {
		friendOK = st.OK
	}
	brainMu.Unlock()
	if friendOK == 0 {
		t.Fatal("the guided Friend stop should record the Friend brain")
	}
}

func TestGuidedWalkResumesFromNotebook(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()
	guideActive, guideIdx, guideRoot = false, 0, ""

	oldProj, oldRoot := lastAnalyzeProject, unifiedProjectRoot
	defer func() { lastAnalyzeProject, unifiedProjectRoot = oldProj, oldRoot }()
	lastAnalyzeProject = nil
	unifiedProjectRoot = t.TempDir() // empty: no Go files; demo falls back in
	tourStops = nil

	handleBeginnerCommand("start", nil) // shows stop 1, bookmark -> 1
	handleBeginnerCommand("next", nil)  // shows stop 2, bookmark -> 2
	root := guideRoot
	if got := friendLoad().Guided[root]; got != 2 {
		t.Fatalf("bookmark after two stops = %d, want 2", got)
	}
	// Simulate leaving and returning: state resets, notebook persists.
	guideActive, guideIdx = false, 0
	handleStart(false)
	if !guideActive || guideIdx != 3 { // resumed at stop 3 (index 2 shown -> idx 3)
		t.Fatalf("resume should land past stop 2: active=%v idx=%d", guideActive, guideIdx)
	}
	guideActive = false
}

func TestHelpIsSituationsFirst(t *testing.T) {
	if !isHelpLine("help") || isHelpAllLine("help") {
		t.Fatal("'help' should be the short situational menu")
	}
	if !isHelpAllLine("help all") {
		t.Fatal("'help all' should select the full catalog")
	}
	if !strings.Contains(translateBeginnerSentence("im new here"), "/start") {
		t.Fatal("a beginner announcing themselves should reach the guide")
	}
	if got := translateBeginnerSentence("stop guiding"); got != "/guidestop" {
		t.Fatalf("stop guiding -> %q", got)
	}
}
