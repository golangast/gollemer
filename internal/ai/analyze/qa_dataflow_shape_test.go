package analyze

import "testing"

// IsDataflowQuestion is the shape gate the chat's intent layer
// puts on its dataflow dispatch: a line earns a varflow report
// by its shape, never by a classifier label plus a token that
// happens to be tracked. These pins hold both directions —
// every dataflow shape the Q&A layer answers is a shape here,
// and plain statements, however many tracked words they carry,
// are not.
func TestIsDataflowQuestion(t *testing.T) {
	shaped := []string{
		"where does Items get its data",
		"where does Items get its data from",
		"where does Items come from",
		"where does Items change",
		"what changes Items",
		"what updates the config",
		"where is Items set",
		"where is the config set?",
		"who sets Items",
		"so where does the total come from",
	}
	for _, q := range shaped {
		if !IsDataflowQuestion(q) {
			t.Errorf("IsDataflowQuestion(%q) = false, want true", q)
		}
	}
	unshaped := []string{
		// The live theft (2026-11-09): a resources ask whose
		// first word is a tracked local in the analyzed repo.
		"i need resources for a database",
		"i need a database in jim",
		"Items is a slice of strings",
		"where does the team meet",
		// Codebase questions, but not dataflow ones: the wider
		// IsCodebaseQuestion set must not leak in here.
		"what is Items for",
		"why does Add call Save",
		"what does Items do",
	}
	for _, q := range unshaped {
		if IsDataflowQuestion(q) {
			t.Errorf("IsDataflowQuestion(%q) = true, want false", q)
		}
	}
}
