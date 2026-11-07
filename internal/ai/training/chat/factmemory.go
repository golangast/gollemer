package chat

// The fact-memory seat: facts the user states about
// themselves, captured and recalled deterministically.
//
// Why this exists: the echo/recall data round (2026-11-07)
// missed its bar because the social model cannot reliably
// echo a fact it was told earlier in a conversation — the
// value echo took, the name echo did not. This seat stops
// asking the model to remember. A fact stated in an ordinary
// statement ("my dog is Bella", "I live in Omaha", "my
// favorite color is forest green") is captured by a
// deterministic grammar into the long-term memory store
// (memory_state.go, global side — facts about the person,
// never per-project), and a fact question ("what's my dog's
// name?") is answered from the store exactly, or honestly
// not-known. One general fact mechanism, not per-fact
// patches: pets are one shape with a subject slot, family
// relations are one shape, favorites are one shape with a
// category slot.
//
// The two halves have opposite claim shapes, on purpose:
//
//   - CAPTURE IS A HOOK, NEVER A CLAIM. The statement still
//     gets exactly the conversational reply it gets today;
//     the hook only stores the fact and the loop prints one
//     short disclosure line after the reply (the store's
//     standing no-silent-capture rule). The disclosure is
//     printed, never added to the conversation, so the
//     model's own history conditioning is untouched.
//   - RECALL IS A CLAIM, seated with the talk seat's
//     deterministic arms (ahead of the intent layer in the
//     unified loop, and as an arm of the seat's responder
//     ahead of the social arm). A question the grammar owns
//     is answered from the store even when no fact is
//     stored — the honest "you haven't told me" — and never
//     falls through to a model that would invent a name.
//     Invented names are the failure this seat exists to
//     kill.
//
// Near-miss discipline is the heart of the grammar: every
// shape is anchored to the whole line, values are validated
// per kind, and a statement the grammar cannot read as a
// fact falls through to conversation byte-unchanged,
// storing nothing. "my dog is tired" is a statement about
// the dog's evening, not a name; "I live in interesting
// times" is not a city. A capture that fires on non-facts
// is worse than none.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// ----- stored shape -----

// longTermFact is one stored fact about the user. Label is
// the listing phrase ("your dog's name"), Value the fact as
// the user stated it (their casing kept), Answer the exact
// recall sentence, rendered at capture time so recall never
// re-derives it. The store key (map key in
// longTermMemoryData.Facts) carries the kind and subject:
// "name", "pet:dog", "family:wife", "city", "job",
// "favorite:color", "birthday", "learning".
type longTermFact struct {
	Label  string `json:"label"`
	Value  string `json:"value"`
	Answer string `json:"answer"`
}

func setLongTermFact(key string, f longTermFact) (old longTermFact, had bool, err error) {
	d, err := loadLongTermMemory()
	if err != nil {
		return longTermFact{}, false, err
	}
	old, had = d.Facts[key]
	d.Facts[key] = f
	if err := saveLongTermMemory(d); err != nil {
		return longTermFact{}, false, err
	}
	return old, had, nil
}

func getLongTermFact(key string) (longTermFact, bool) {
	// Recall reads the store fresh: it is rare, and a fact
	// captured moments ago in this session — or by another
	// session sharing the home — must be the one answered.
	d, err := loadLongTermMemory()
	if err != nil {
		return longTermFact{}, false
	}
	f, ok := d.Facts[key]
	return f, ok
}

func sortedFactKeys(facts map[string]longTermFact) []string {
	keys := make([]string, 0, len(facts))
	for k := range facts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// factForgetPhrases are the normalized phrases a "forget my
// …" line may use for one fact: its full label, the label
// without the leading "your", the bare subject for slotted
// keys, and a few spoken extras for the unslotted kinds.
func factForgetPhrases(key string, f longTermFact) []string {
	label := normalizeMemoryMatch(f.Label)
	phrases := []string{label}
	if rest := strings.TrimPrefix(label, "your "); rest != label {
		phrases = append(phrases, rest)
	}
	if i := strings.Index(key, ":"); i >= 0 {
		phrases = append(phrases, normalizeMemoryMatch(key[i+1:]))
	}
	switch key {
	case "name":
		phrases = append(phrases, "name")
	case "city":
		phrases = append(phrases, "city", "where i live")
	case "learning":
		phrases = append(phrases, "learning")
	}
	return phrases
}

// forgetLongTermFact mirrors forgetLongTermPreference's
// contract: an unambiguous match removes just that fact and
// returns it; several matches return their labels as
// candidates and change nothing; no match returns neither.
// One refinement preferences don't need: fact labels are
// structured, so an exact phrase match outranks a substring
// one — "forget my wife's name" names the wife fact exactly
// and must not be made ambiguous by the plain name fact's
// shorter "name" phrase being a substring of it. Only when
// no phrase matches exactly does the preference-style
// substring-both-ways pass run.
func forgetLongTermFact(query string) (removed *longTermFact, candidates []string, err error) {
	q := normalizeMemoryMatch(query)
	if q == "" {
		return nil, nil, nil
	}
	d, err := loadLongTermMemory()
	if err != nil {
		return nil, nil, err
	}
	var matched []string
	for _, key := range sortedFactKeys(d.Facts) {
		for _, p := range factForgetPhrases(key, d.Facts[key]) {
			if p == q {
				matched = append(matched, key)
				break
			}
		}
	}
	if len(matched) == 0 {
		for _, key := range sortedFactKeys(d.Facts) {
			for _, p := range factForgetPhrases(key, d.Facts[key]) {
				if strings.Contains(p, q) || strings.Contains(q, p) {
					matched = append(matched, key)
					break
				}
			}
		}
	}
	if len(matched) == 0 {
		return nil, nil, nil
	}
	if len(matched) > 1 {
		for _, key := range matched {
			candidates = append(candidates, d.Facts[key].Label)
		}
		return nil, candidates, nil
	}
	f := d.Facts[matched[0]]
	delete(d.Facts, matched[0])
	if err := saveLongTermMemory(d); err != nil {
		return nil, nil, err
	}
	return &f, nil, nil
}

// ----- value discipline -----

var (
	factNameTokenRe = regexp.MustCompile(`^[A-Za-z][A-Za-z'’-]*$`)
	factWordTokenRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9'&+.-]*$`)
	factDateTokenRe = regexp.MustCompile(`^[0-9]{1,2}([/-][0-9]{1,2}){1,2}$`)
	factMonthRe     = regexp.MustCompile(`(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b`)
)

// factStateWords are states and evaluations, never values:
// the words that make "my dog is tired" and "my job is
// hard" statements about a state instead of declarations
// of a fact. A value containing one is declined.
var factStateWords = map[string]bool{
	"tired": true, "sick": true, "ill": true, "sleepy": true,
	"hungry": true, "thirsty": true, "asleep": true, "awake": true,
	"happy": true, "sad": true, "cute": true, "big": true,
	"small": true, "little": true, "old": true, "young": true,
	"good": true, "bad": true, "great": true, "fine": true,
	"ok": true, "okay": true, "well": true, "here": true,
	"gone": true, "lost": true, "dead": true, "alive": true,
	"home": true, "ready": true, "busy": true, "playful": true,
	"friendly": true, "scared": true, "nervous": true, "excited": true,
	"hard": true, "tough": true, "easy": true, "boring": true,
	"stressful": true, "fun": true, "rewarding": true, "terrible": true,
	"awful": true, "new": true, "temporary": true,
}

func factHasStateWord(words []string) bool {
	for _, w := range words {
		if factStateWords[strings.ToLower(w)] {
			return true
		}
	}
	return false
}

// factNameValue validates a name value: one to three words,
// each a plain name token. Casing is the user's, kept.
func factNameValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 3 {
		return "", false
	}
	for _, w := range words {
		if !factNameTokenRe.MatchString(w) {
			return "", false
		}
	}
	return strings.Join(words, " "), true
}

// factPetBareValue validates the value of the bare pet
// shape ("my dog is X"), the ambiguous one: it must read as
// a name — one or two name words, no leading article, no
// state word. "my dog is Bella" is a name; "my dog is
// tired" and "my dog is a good boy" are not. A value opening
// with the name-introducer verbs themselves is the called
// shape with its value missing ("my cat is called"), not a
// name.
func factPetBareValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 2 {
		return "", false
	}
	switch strings.ToLower(words[0]) {
	case "a", "an", "the", "called", "named":
		return "", false
	}
	if factHasStateWord(words) {
		return "", false
	}
	return factNameValue(v)
}

func factWordValue(v string, max int) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > max {
		return "", false
	}
	for _, w := range words {
		if !factWordTokenRe.MatchString(w) {
			return "", false
		}
	}
	return strings.Join(words, " "), true
}

// factJobValue is the value of "my job is X": word-shaped,
// and no state word — "my job is teaching" is a job, "my
// job is hard" is a review.
func factJobValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 4 {
		return "", false
	}
	if factHasStateWord(words) {
		return "", false
	}
	return factWordValue(v, 4)
}

// factFavoriteValue: a favorite's value is a thing — a few
// word tokens, and not a clause ("pizza is great" is a
// sentence about pizza, not a favorite food).
func factFavoriteValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 4 {
		return "", false
	}
	padded := " " + strings.ToLower(strings.Join(words, " ")) + " "
	if strings.Contains(padded, " is ") || strings.Contains(padded, " was ") {
		return "", false
	}
	return factWordValue(v, 4)
}

// factCityValue: a single word is a place as stated
// ("Omaha", "crestwood"); a multi-word place must be
// capitalized as typed ("New York", "San Francisco") —
// lowercase multi-word values are phrases, not places
// ("interesting times", "a small town"), and are declined.
func factCityValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 3 {
		return "", false
	}
	for _, w := range words {
		if !factNameTokenRe.MatchString(w) {
			return "", false
		}
	}
	if len(words) == 1 {
		return words[0], true
	}
	for _, w := range words {
		if w[0] < 'A' || w[0] > 'Z' {
			return "", false
		}
	}
	return strings.Join(words, " "), true
}

func factBirthdayValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 3 {
		return "", false
	}
	joined := strings.Join(words, " ")
	if !factMonthRe.MatchString(joined) && !strings.ContainsAny(joined, "0123456789") {
		return "", false
	}
	for _, w := range words {
		if !factWordTokenRe.MatchString(w) && !factDateTokenRe.MatchString(w) {
			return "", false
		}
	}
	return joined, true
}

func factLearningValue(v string) (string, bool) {
	words := strings.Fields(v)
	if len(words) == 0 || len(words) > 4 {
		return "", false
	}
	if strings.ToLower(words[0]) == "to" {
		return "", false // "i'm learning to let go" is not a subject
	}
	return factWordValue(v, 4)
}

// ----- capture grammar -----

const factRelations = `wife|husband|son|daughter|kid|child|mom|dad|mother|father|brother|sister|friend`
const factPets = `dog|cat|pet`

var (
	factActuallyRe  = regexp.MustCompile(`(?i)^actually[\s,]+`)
	factSelfNameRe  = regexp.MustCompile(`(?i)^my name is\s+(.+?)\s*[.!]?\s*$`)
	factPetNamedRe  = regexp.MustCompile(`(?i)^my (dog|cat|pet)['’]s name is\s+(.+?)\s*[.!]?\s*$`)
	factPetCalledRe = regexp.MustCompile(`(?i)^my (dog|cat|pet) is (?:called|named)\s+(.+?)\s*[.!]?\s*$`)
	factPetBareRe   = regexp.MustCompile(`(?i)^my (dog|cat|pet) is\s+(.+?)\s*[.!]?\s*$`)
	factFamNamedRe  = regexp.MustCompile(`(?i)^my (` + factRelations + `)['’]s name is\s+(.+?)\s*[.!]?\s*$`)
	factFamCalledRe = regexp.MustCompile(`(?i)^my (` + factRelations + `) is (?:called|named)\s+(.+?)\s*[.!]?\s*$`)
	factCityRe      = regexp.MustCompile(`(?i)^i live in\s+(.+?)\s*[.!]?\s*$`)
	factJobAsRe     = regexp.MustCompile(`(?i)^i work as\s+(.+?)\s*[.!]?\s*$`)
	factJobIsRe     = regexp.MustCompile(`(?i)^my job is\s+(.+?)\s*[.!]?\s*$`)
	factFavoriteRe  = regexp.MustCompile(`(?i)^my favorite\s+([a-z][a-z-]*)\s+is\s+(.+?)\s*[.!]?\s*$`)
	factBirthdayRe  = regexp.MustCompile(`(?i)^my birthday is\s+(?:(on|in)\s+)?(.+?)\s*[.!]?\s*$`)
	factLearningRe  = regexp.MustCompile(`(?i)^i(?:'m| am) learning\s+(.+?)\s*[.!]?\s*$`)
)

// parseFactCapture reads one ordinary statement as a fact
// declaration. It returns the store key and the rendered
// fact, or ok=false — the caller then treats the line as
// plain conversation and stores nothing. The shapes are
// tried most-explicit first: a pet's "is called" and
// "'s name is" forms are read before the bare "is", whose
// value must additionally read as a name.
func parseFactCapture(line string) (string, longTermFact, bool) {
	s := strings.TrimSpace(line)
	// A leading "actually" is the correction marker ("actually
	// my dog is Rex"), not part of the statement.
	s = factActuallyRe.ReplaceAllString(s, "")
	mk := func(key, label, value, answer string) (string, longTermFact, bool) {
		return key, longTermFact{Label: label, Value: value, Answer: answer}, true
	}
	if m := factSelfNameRe.FindStringSubmatch(s); m != nil {
		if v, ok := factNameValue(m[1]); ok {
			return mk("name", "your name", v, "Your name is "+v+".")
		}
	}
	if m := factPetNamedRe.FindStringSubmatch(s); m != nil {
		sub := strings.ToLower(m[1])
		if v, ok := factNameValue(m[2]); ok {
			return mk("pet:"+sub, "your "+sub+"'s name", v, "Your "+sub+"'s name is "+v+".")
		}
	}
	if m := factPetCalledRe.FindStringSubmatch(s); m != nil {
		sub := strings.ToLower(m[1])
		if v, ok := factNameValue(m[2]); ok {
			return mk("pet:"+sub, "your "+sub+"'s name", v, "Your "+sub+"'s name is "+v+".")
		}
	}
	if m := factFamNamedRe.FindStringSubmatch(s); m != nil {
		sub := strings.ToLower(m[1])
		if v, ok := factNameValue(m[2]); ok {
			return mk("family:"+sub, "your "+sub+"'s name", v, "Your "+sub+"'s name is "+v+".")
		}
	}
	if m := factFamCalledRe.FindStringSubmatch(s); m != nil {
		sub := strings.ToLower(m[1])
		if v, ok := factNameValue(m[2]); ok {
			return mk("family:"+sub, "your "+sub+"'s name", v, "Your "+sub+"'s name is "+v+".")
		}
	}
	if m := factPetBareRe.FindStringSubmatch(s); m != nil {
		sub := strings.ToLower(m[1])
		if v, ok := factPetBareValue(m[2]); ok {
			return mk("pet:"+sub, "your "+sub+"'s name", v, "Your "+sub+"'s name is "+v+".")
		}
	}
	if m := factCityRe.FindStringSubmatch(s); m != nil {
		if v, ok := factCityValue(m[1]); ok {
			return mk("city", "where you live", v, "You live in "+v+".")
		}
	}
	if m := factJobAsRe.FindStringSubmatch(s); m != nil {
		if v, ok := factWordValue(m[1], 4); ok {
			return mk("job", "your job", v, "You work as "+v+".")
		}
	}
	if m := factJobIsRe.FindStringSubmatch(s); m != nil {
		if v, ok := factJobValue(m[1]); ok {
			return mk("job", "your job", v, "Your job is "+v+".")
		}
	}
	if m := factFavoriteRe.FindStringSubmatch(s); m != nil {
		cat := strings.ToLower(m[1])
		if v, ok := factFavoriteValue(m[2]); ok {
			return mk("favorite:"+cat, "your favorite "+cat, v, "Your favorite "+cat+" is "+v+".")
		}
	}
	if m := factBirthdayRe.FindStringSubmatch(s); m != nil {
		if v, ok := factBirthdayValue(m[2]); ok {
			if m[1] != "" {
				v = strings.ToLower(m[1]) + " " + v
			}
			return mk("birthday", "your birthday", v, "Your birthday is "+v+".")
		}
	}
	if m := factLearningRe.FindStringSubmatch(s); m != nil {
		if v, ok := factLearningValue(m[1]); ok {
			return mk("learning", "what you're learning", v, "You're learning "+v+".")
		}
	}
	return "", longTermFact{}, false
}

// captureFactStatement is the loop's capture hook: it
// stores the fact a statement declares and returns the
// short disclosure line for the loop to print AFTER the
// statement's own reply — "" when the line declares no
// fact. A restated fact overwrites, and the disclosure
// says so.
func captureFactStatement(line string) string {
	key, f, ok := parseFactCapture(line)
	if !ok {
		return ""
	}
	old, had, err := setLongTermFact(key, f)
	if err != nil {
		return "I couldn't save that to my memory store: " + err.Error()
	}
	body := strings.TrimSuffix(f.Answer, ".")
	if had && old.Value != f.Value {
		return "I'll remember: " + lowerFirstFact(body) + " (updated from " + old.Value + ")."
	}
	return "I'll remember: " + lowerFirstFact(body) + "."
}

func lowerFirstFact(s string) string {
	if s == "" {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}

// printFactNote prints the capture hook's disclosure under
// the talk tag. It is deliberately NOT recorded in the
// conversation: the disclosure is bookkeeping about the
// store, and the model's history conditioning must stay
// exactly what it was before this seat existed.
func printFactNote(note string) {
	if note == "" {
		return
	}
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), note)
}

// ----- recall grammar -----

var (
	factQNameRe      = regexp.MustCompile(`(?i)^\s*(?:what['’]s my name|what is my name|who am i)\s*[?.!]*\s*$`)
	factQPetNamedRe  = regexp.MustCompile(`(?i)^\s*what(?:['’]s| is) my (dog|cat|pet)['’]s name\s*[?.!]*\s*$`)
	factQPetCalledRe = regexp.MustCompile(`(?i)^\s*what(?:['’]s| is) my (dog|cat|pet) called\s*[?.!]*\s*$`)
	factQFamNamedRe  = regexp.MustCompile(`(?i)^\s*what(?:['’]s| is) my (` + factRelations + `)['’]s name\s*[?.!]*\s*$`)
	factQFamWhoRe    = regexp.MustCompile(`(?i)^\s*who(?:['’]s| is) my (` + factRelations + `)\s*[?.!]*\s*$`)
	factQCityRe      = regexp.MustCompile(`(?i)^\s*where do i live\s*[?.!]*\s*$`)
	factQJobRe       = regexp.MustCompile(`(?i)^\s*(?:what do i do for (?:work|a living)|what(?:['’]s| is) my job)\s*[?.!]*\s*$`)
	factQFavoriteRe  = regexp.MustCompile(`(?i)^\s*what(?:['’]s| is) my favorite\s+([a-z][a-z-]*)\s*[?.!]*\s*$`)
	factQBirthdayRe  = regexp.MustCompile(`(?i)^\s*(?:when is my birthday|what(?:['’]s| is) my birthday)\s*[?.!]*\s*$`)
	factQLearningRe  = regexp.MustCompile(`(?i)^\s*what am i learning\s*[?.!]*\s*$`)
)

// factRecallAnswer answers a fact question from the store:
// the stored fact's own sentence when one is stored, and
// the honest not-told-me sentence when the question's
// shape is owned but no fact is. ok=false means the line
// is not a fact question this seat owns — it declines
// narrowly, and the line proceeds untouched. The shapes
// are anchored whole-line and first-person: "what is the
// capital…" never reaches the store, "what's my dog
// dreaming about" is not the name question, and "where do
// I live-stream" is not the city question.
func factRecallAnswer(line string) (string, bool) {
	key, honest := "", ""
	switch {
	case factQNameRe.MatchString(line):
		key, honest = "name", "You haven't told me your name yet."
	case factQPetNamedRe.MatchString(line):
		sub := strings.ToLower(factQPetNamedRe.FindStringSubmatch(line)[1])
		key, honest = "pet:"+sub, "You haven't told me your "+sub+"'s name yet."
	case factQPetCalledRe.MatchString(line):
		sub := strings.ToLower(factQPetCalledRe.FindStringSubmatch(line)[1])
		key, honest = "pet:"+sub, "You haven't told me your "+sub+"'s name yet."
	case factQFamNamedRe.MatchString(line):
		sub := strings.ToLower(factQFamNamedRe.FindStringSubmatch(line)[1])
		key, honest = "family:"+sub, "You haven't told me your "+sub+"'s name yet."
	case factQFamWhoRe.MatchString(line):
		sub := strings.ToLower(factQFamWhoRe.FindStringSubmatch(line)[1])
		key, honest = "family:"+sub, "You haven't told me your "+sub+"'s name yet."
	case factQCityRe.MatchString(line):
		key, honest = "city", "You haven't told me where you live yet."
	case factQJobRe.MatchString(line):
		key, honest = "job", "You haven't told me what you do for work yet."
	case factQFavoriteRe.MatchString(line):
		cat := strings.ToLower(factQFavoriteRe.FindStringSubmatch(line)[1])
		key, honest = "favorite:"+cat, "You haven't told me your favorite "+cat+" yet."
	case factQBirthdayRe.MatchString(line):
		key, honest = "birthday", "You haven't told me your birthday yet."
	case factQLearningRe.MatchString(line):
		key, honest = "learning", "You haven't told me what you're learning yet."
	default:
		return "", false
	}
	if f, ok := getLongTermFact(key); ok {
		return f.Answer, true
	}
	return honest, true
}

// tryFactRecallAnswer is the fact seat's claim in the
// unified loop, seated beside tryTalkStateAnswer — ahead
// of the intent layer and of every model — for the same
// reason the state arm sits there: a fact question is a
// fact, and a classifier or a model must never get the
// chance to invent its answer.
func tryFactRecallAnswer(line string, conv *Conversation) bool {
	out, ok := factRecallAnswer(line)
	if !ok {
		return false
	}
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), out)
	if conv != nil {
		conv.AddUser(line)
		conv.AddReply(out, TalkDomain, false)
	}
	return true
}
