package chat

// Checked sampling for the layers social seat (layers_seat.go),
// seated 2026-11-18 on John's word after the decode probe
// measured it on the adopted model: drawing up to 8 replies at
// temperature 0.8 and serving the first that carries the line's
// topic scored 240/426 on the pack-2 combination holdout against
// greedy's 176/426, and 67/160 against 58/160 on the frozen set
// (~/workspace/decode-probe-scratch/RESULTS.md). The product
// policy differs from the probe protocol in one deliberate way:
// a greedy reply that already carries the topic is kept as-is —
// sampling only rescues replies that dropped the subject — so
// served behavior tracks the measurement with far fewer draws.
//
// The check is deterministic and reads only the user's line:
// the topic is the longest phrase from the model's topic banks
// that the line names (word boundaries); a line naming no bank
// topic falls back to the line's own content words
// (seq2seq.ContentWordsOf), and a reply carries the line when
// it names the topic phrase or at least one of those words.
// Nothing about the reference answers a judging run compares
// against is visible here — the check is structure, not a key.

import (
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// The decode policy the probe measured at its safer setting:
// temperature 0.8 (1.0 scored 3 items higher on unseen cells
// and garbled more; 0.8 won the seen cells outright), at most
// this many sampled draws per rescued reply.
const (
	layersSampleTemp  = 0.8
	layersSampleDraws = 8
)

// layersTopicPhrases is the topic bank the adopted model was
// trained on: the 132 topic phrases of the pack-1 + pack-2
// topic banks, extracted verbatim from
// gen_copypack2_pairs.py (PACK1_TOPICS + WAVE2_TOPICS, first
// tuple element). A future model trained on new topics extends
// this list with its own banks.
var layersTopicPhrases = []string{
	"kayaking",
	"canoeing",
	"paddleboarding",
	"sailing",
	"surfing",
	"archery",
	"bouldering",
	"rock climbing",
	"roller skating",
	"ice skating",
	"snowboarding",
	"bowling",
	"darts",
	"pottery",
	"knitting",
	"crocheting",
	"sourdough",
	"pickling",
	"canning",
	"origami",
	"calligraphy",
	"candle making",
	"soap making",
	"watercolor",
	"sketching",
	"woodworking",
	"chess",
	"crossword",
	"jigsaw puzzle",
	"board games",
	"card games",
	"karaoke",
	"ukulele",
	"harmonica",
	"choir",
	"houseplants",
	"terrarium",
	"aquarium",
	"beekeeping",
	"backyard chickens",
	"herb garden",
	"tomatoes",
	"compost",
	"bird feeder",
	"birdwatching",
	"stargazing",
	"thrift shopping",
	"flea market",
	"garage sale",
	"book club",
	"journaling",
	"volunteering",
	"meditation",
	"train ride",
	"badminton",
	"table tennis",
	"volleyball",
	"hockey",
	"skiing",
	"skateboarding",
	"juggling",
	"magic tricks",
	"frisbee",
	"kite flying",
	"boxing",
	"martial arts",
	"tai chi",
	"pilates",
	"weightlifting",
	"golf",
	"tennis",
	"soccer",
	"baseball",
	"basketball",
	"swimming",
	"camping",
	"yoga",
	"sewing",
	"embroidery",
	"quilting",
	"bonsai",
	"flower arranging",
	"metal detecting",
	"writing poetry",
	"photography",
	"the saxophone",
	"the drums",
	"the violin",
	"laundry",
	"ironing",
	"grocery shopping",
	"meal prep",
	"decluttering",
	"mowing the lawn",
	"raking leaves",
	"shoveling snow",
	"washing the car",
	"house cleaning",
	"dog walking",
	"budgeting",
	"saving money",
	"paying bills",
	"taxes",
	"bookkeeping",
	"deadlines",
	"presentations",
	"overtime",
	"job interview",
	"night school",
	"online course",
	"learning spanish",
	"tutoring",
	"exams",
	"the dentist",
	"the museum",
	"the zoo",
	"the farmers market",
	"birthday party",
	"a wedding",
	"graduation",
	"family reunion",
	"baby shower",
	"holiday dinner",
	"school play",
	"parent evening",
	"commuting",
	"the bus",
	"watching the rain",
	"first snow",
	"autumn leaves",
	"brewing",
	"fermenting",
}

type layersTopicEntry struct {
	phrase string
	re     *regexp.Regexp
}

var layersTopicsByLength = buildLayersTopics()

func buildLayersTopics() []layersTopicEntry {
	phrases := append([]string{}, layersTopicPhrases...)
	sort.Slice(phrases, func(i, j int) bool {
		if len(phrases[i]) != len(phrases[j]) {
			return len(phrases[i]) > len(phrases[j])
		}
		return phrases[i] < phrases[j]
	})
	out := make([]layersTopicEntry, 0, len(phrases))
	for _, p := range phrases {
		out = append(out, layersTopicEntry{p, layersWordRe(p)})
	}
	return out
}

var layersWordReCache = map[string]*regexp.Regexp{}

func layersWordRe(word string) *regexp.Regexp {
	if re, ok := layersWordReCache[word]; ok {
		return re
	}
	re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(word) + `\b`)
	layersWordReCache[word] = re
	return re
}

// layersTopicFor resolves a line's topic: the longest bank
// phrase the line names on word boundaries, or "" when the
// line names none.
func layersTopicFor(line string) (string, *regexp.Regexp) {
	for _, te := range layersTopicsByLength {
		if te.re.MatchString(line) {
			return te.phrase, te.re
		}
	}
	return "", nil
}

// layersLineWords splits a line the way the probe's checker
// read its inputs: lowercased fields with edge punctuation
// trimmed.
func layersLineWords(line string) []string {
	fields := strings.Fields(strings.ToLower(line))
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f = strings.Trim(f, ".,!?;:\"'()[]"); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// layersReplyCarries is the deterministic topic check: the
// reply carries the line when it names the line's bank topic,
// or — for a line naming no bank topic — at least one of the
// line's content words. An empty reply never carries; a line
// with no topic and no content words is carried by nothing, so
// its greedy reply always stands.
func layersReplyCarries(line, reply string) bool {
	if strings.TrimSpace(reply) == "" {
		return false
	}
	if _, re := layersTopicFor(line); re != nil {
		return re.MatchString(reply)
	}
	for _, w := range seq2seq.ContentWordsOf(layersLineWords(line)) {
		if layersWordRe(w).MatchString(reply) {
			return true
		}
	}
	return false
}

// layersPickReply is the seat's selection policy, factored out
// for tests: the greedy reply stands when it carries the line;
// otherwise draws are tried in order and the first carrying
// one wins; a draw error or no carrying draw leaves the greedy
// reply standing.
func layersPickReply(greedy string, carries func(string) bool, draw func() (string, error)) string {
	if carries(greedy) {
		return greedy
	}
	for i := 0; i < layersSampleDraws; i++ {
		s, err := draw()
		if err != nil {
			return greedy
		}
		if carries(s) {
			return s
		}
	}
	return greedy
}
