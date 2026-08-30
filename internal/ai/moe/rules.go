package moe

import (
	"encoding/csv"
	"strings"
	"sync"
)

// IntentRule defines the structural and vocabulary expectations for a specific intent.
type IntentRule struct {
	ParentIntent string
	ChildIntent  string

	// GrammarSkeleton defines the expected sequence of POS types (simplified)
	// Example: ["PRON", "VERB", "ADJ"] -> "I am happy"
	GrammarSkeleton []string

	// RequiredKeywords are words that SHOULD be present for this intent to be valid.
	RequiredKeywords []string

	// ForbiddenPatterns are sequences that make the sentence incoherent for this intent.
	ForbiddenPatterns []string
}

// EvaluateWindow evaluates a sliding window of three consecutive tokens (Tri-grams)
// to catch complex errors that bigrams miss, such as missing words in verb phrases
// or improper punctuation placement.
func (r *IntentRule) EvaluateWindow(prev, curr, next string) float32 {
	var penalty float32 = 0.0

	// Missing words in verb phrases
	if prev == "AUX" && curr != "VERB" && curr != "ADJ" && curr != "PREP" && curr != "PRON" {
		penalty += 0.3
	}

	// Improper punctuation placement (prevent ending on an AUX or PREP)
	if next == "EOS" && (curr == "AUX" || curr == "PREP" || curr == "INTERROGATIVE") {
		penalty += 0.4
	}

	// Double pronoun without verb/prep
	if prev == "PRON" && curr == "PRON" {
		penalty += 0.2
	}

	return penalty
}

// RuleBook is the sophisticated collection of linguistic rules for the MoE model.
type RuleBook struct {
	Rules        map[string]IntentRule
	EntityRoutes map[string]string // entity keyword -> cartridge path
}

// NewRuleBook initializes the rule system with standard conversational grammar.
func NewRuleBook() *RuleBook {
	rb := &RuleBook{
		Rules:        make(map[string]IntentRule),
		EntityRoutes: make(map[string]string),
	}

	// Rule: Greeting — "hi there! how can i help you today?"
	// CSV grammar: GREET OTHER OTHER AUX PRON VERB PRON OTHER
	rb.Rules["social:greeting"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "greeting",
		GrammarSkeleton:  []string{"GREET", "OTHER", "OTHER", "INTERROGATIVE", "AUX", "PRON", "VERB", "PRON", "OTHER", "OTHER"},
		RequiredKeywords: []string{"hi", "hello", "help", "how", "today"},
	}

	// Rule: Identity — "i am gollemer, your ai assistant."
	// CSV grammar: PRON VERB NOUN PRON NOUN NOUN
	rb.Rules["social:identity"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "identity",
		GrammarSkeleton:  []string{"PRON", "VERB", "NOUN", "PRON", "NOUN", "NOUN"},
		RequiredKeywords: []string{"gollemer", "ai", "assistant"},
	}

	// Rule: Status Check — "i am doing well, thank you for asking!"
	// CSV grammar: PRON VERB OTHER ADJ OTHER PRON PREP OTHER
	rb.Rules["social:status_check"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "status_check",
		GrammarSkeleton:  []string{"PRON", "VERB", "OTHER", "ADJ", "OTHER", "PRON", "PREP", "OTHER"},
		RequiredKeywords: []string{"doing", "well"},
	}

	// Rule: Polite — "you are very welcome! i am happy to help."
	// CSV grammar: PRON VERB OTHER OTHER PRON VERB ADJ PREP VERB
	rb.Rules["social:polite"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "polite",
		GrammarSkeleton:  []string{"PRON", "VERB", "OTHER", "OTHER", "PRON", "VERB", "ADJ", "PREP", "VERB"},
		RequiredKeywords: []string{"welcome", "happy", "help"},
	}

	// Rule: Farewell — "goodbye! it was nice talking to you today."
	// CSV grammar: GREET PRON VERB ADJ OTHER PREP PRON OTHER
	rb.Rules["social:farewell"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "farewell",
		GrammarSkeleton:  []string{"GREET", "PRON", "VERB", "ADJ", "OTHER", "PREP", "PRON", "OTHER"},
		RequiredKeywords: []string{"goodbye"},
	}

	// Rule: Capabilities — "i can answer questions, tell jokes, and help you with your code."
	// CSV grammar: PRON AUX OTHER NOUN OTHER OTHER PREP VERB PRON PREP PRON NOUN
	rb.Rules["social:capabilities"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "capabilities",
		GrammarSkeleton:  []string{"PRON", "AUX", "OTHER", "NOUN", "OTHER", "VERB", "PRON", "PREP", "PRON", "NOUN"},
		RequiredKeywords: []string{"can", "help"},
	}

	// Rule: Emotional Support — "i hope you can get some rest soon."
	// CSV grammar: PRON OTHER PRON AUX VERB OTHER OTHER OTHER
	rb.Rules["social:emotional_support"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "emotional_support",
		GrammarSkeleton:  []string{"PRON", "OTHER", "PRON", "AUX", "VERB", "OTHER", "OTHER"},
		RequiredKeywords: []string{},
	}

	// Rule: Support/Help — "i would be happy to help!"
	// CSV grammar: PRON AUX VERB ADJ PREP VERB
	rb.Rules["social:support"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "support",
		GrammarSkeleton:  []string{"PRON", "AUX", "VERB", "ADJ", "PREP", "VERB"},
		RequiredKeywords: []string{"help"},
	}

	// Rule: General Social — simple subject-verb structure
	rb.Rules["social:social_chat"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "social_chat",
		GrammarSkeleton:  []string{"PRON", "VERB", "OTHER"},
		RequiredKeywords: []string{},
	}

	// Rule: Trivia / Knowledge
	rb.Rules["social:trivia"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "trivia",
		GrammarSkeleton:  []string{"OTHER", "PRON", "OTHER", "OTHER", "OTHER", "VERB"},
		RequiredKeywords: []string{},
	}

	// Rule: Small Talk
	rb.Rules["social:small_talk"] = IntentRule{
		ParentIntent:     "social",
		ChildIntent:      "small_talk",
		GrammarSkeleton:  []string{"PRON", "OTHER", "OTHER", "PRON", "VERB"},
		RequiredKeywords: []string{},
	}

	// Entity Routes: map subject keywords to specialized cartridges.
	rb.EntityRoutes["channel"] = "data/models/intents/channel.cartridge"
	rb.EntityRoutes["goroutine"] = "data/models/intents/goroutine.cartridge"
	rb.EntityRoutes["mutex"] = "data/models/intents/mutex.cartridge"
	rb.EntityRoutes["interface"] = "data/models/intents/interface.cartridge"
	rb.EntityRoutes["error"] = "data/models/intents/error.cartridge"
	rb.EntityRoutes["context"] = "data/models/intents/context.cartridge"
	rb.EntityRoutes["slice"] = "data/models/intents/slice.cartridge"
	rb.EntityRoutes["map"] = "data/models/intents/map.cartridge"
	rb.EntityRoutes["defer"] = "data/models/intents/defer.cartridge"
	rb.EntityRoutes["init"] = "data/models/intents/init.cartridge"
	rb.EntityRoutes["package"] = "data/models/intents/package.cartridge"
	rb.EntityRoutes["module"] = "data/models/intents/module.cartridge"
	rb.EntityRoutes["struct"] = "data/models/intents/struct.cartridge"
	rb.EntityRoutes["function"] = "data/models/intents/function.cartridge"
	rb.EntityRoutes["garbage collector"] = "data/models/intents/gc.cartridge"
	rb.EntityRoutes["test"] = "data/models/intents/testing.cartridge"
	rb.EntityRoutes["http"] = "data/models/intents/http.cartridge"
	rb.EntityRoutes["panic"] = "data/models/intents/panic.cartridge"
	rb.EntityRoutes["race"] = "data/models/intents/race.cartridge"
	rb.EntityRoutes["build"] = "data/models/intents/build.cartridge"
	rb.EntityRoutes["config"] = "data/models/intents/config.cartridge"

	return rb
}

// GetRuleByIntent retrieves the rule for a specific intent pair.
func (rb *RuleBook) GetRuleByIntent(parent, child string) (IntentRule, bool) {
	key := parent + ":" + child
	r, ok := rb.Rules[key]
	return r, ok
}

// GetEntityRoute returns the cartridge path for a given entity keyword.
func (rb *RuleBook) GetEntityRoute(keyword string) (string, bool) {
	route, ok := rb.EntityRoutes[strings.ToLower(keyword)]
	return route, ok
}

var lexiconCSV = `word,type
hello,GREET
hi,GREET
hey,GREET
greetings,GREET
afternoon,GREET
hiya,GREET
howdy,GREET
yo,GREET
sup,GREET
hii,GREET
heya,GREET
ohh,GREET
oh,GREET
wow,GREET
haha,GREET
lol,GREET
hehe,GREET
yep,GREET
yup,GREET
yeah,GREET
yea,GREET
yes,GREET
nope,GREET
nah,GREET
cool,GREET
awesome,GREET
amazing,GREET
interesting,GREET
true,GREET
right,GREET
exactly,GREET
indeed,GREET
absolutely,GREET
there,GREET
here,GREET
now,GREET
i,PRON
me,PRON
my,PRON
mine,PRON
myself,PRON
you,PRON
your,PRON
yours,PRON
yourself,PRON
he,PRON
him,PRON
his,PRON
himself,PRON
she,PRON
her,PRON
hers,PRON
herself,PRON
it,PRON
its,PRON
itself,PRON
we,PRON
us,PRON
our,PRON
ours,PRON
ourselves,PRON
they,PRON
them,PRON
their,PRON
theirs,PRON
themselves,PRON
this,PRON
that,PRON
these,PRON
those,PRON
someone,PRON
anyone,PRON
everyone,PRON
nobody,PRON
somebody,PRON
am,VERB
is,VERB
are,VERB
was,VERB
were,VERB
be,VERB
been,VERB
being,VERB
seem,VERB
seems,VERB
seemed,VERB
feel,VERB
feels,VERB
felt,VERB
looks,VERB
sounded,VERB
become,VERB
became,VERB
stay,VERB
stayed,VERB
remain,VERB
remains,VERB
think,VERB
thinking,VERB
thought,VERB
know,VERB
knowing,VERB
knew,VERB
want,VERB
wanted,VERB
wanting,VERB
like,VERB
liked,VERB
liking,VERB
love,VERB
loved,VERB
loving,VERB
make,VERB
made,VERB
making,VERB
go,VERB
went,VERB
gone,VERB
start,VERB
started,VERB
starting,VERB
use,VERB
used,VERB
using,VERB
work,VERB
worked,VERB
working,VERB
try,VERB
tried,VERB
trying,VERB
learn,VERB
learned,VERB
learning,VERB
take,VERB
took,VERB
taken,VERB
taking,VERB
put,VERB
keep,VERB
kept,VERB
come,VERB
came,VERB
coming,VERB
run,VERB
ran,VERB
running,VERB
see,VERB
saw,VERB
seen,VERB
seeing,VERB
look,VERB
looked,VERB
looking,VERB
find,VERB
found,VERB
finding,VERB
give,VERB
gave,VERB
given,VERB
giving,VERB
tell,VERB
told,VERB
telling,VERB
ask,VERB
asked,VERB
asking,VERB
talk,VERB
talked,VERB
talking,VERB
say,VERB
said,VERB
saying,VERB
help,VERB
helped,VERB
helping,VERB
plan,VERB
planned,VERB
planning,VERB
buy,VERB
bought,VERB
buying,VERB
eat,VERB
ate,VERB
eaten,VERB
eating,VERB
drink,VERB
drank,VERB
drunk,VERB
drinking,VERB
read,VERB
reading,VERB
write,VERB
wrote,VERB
written,VERB
writing,VERB
play,VERB
played,VERB
playing,VERB
cook,VERB
cooked,VERB
cooking,VERB
build,VERB
built,VERB
building,VERB
enjoy,VERB
enjoyed,VERB
enjoying,VERB
listen,VERB
listened,VERB
listening,VERB
practice,VERB
practiced,VERB
practicing,VERB
need,VERB
needed,VERB
needing,VERB
check,VERB
checked,VERB
checking,VERB
share,VERB
shared,VERB
sharing,VERB
hear,VERB
heard,VERB
hearing,VERB
remember,VERB
remembered,VERB
remembering,VERB
spend,VERB
spent,VERB
spending,VERB
live,VERB
lived,VERB
living,VERB
grow,VERB
grew,VERB
grown,VERB
growing,VERB
move,VERB
moved,VERB
moving,VERB
done,VERB
finished,VERB
completed,VERB
will,AUX
would,AUX
can,AUX
could,AUX
should,AUX
shall,AUX
may,AUX
might,AUX
must,AUX
ought,AUX
dare,AUX
do,AUX
does,AUX
did,AUX
doing,AUX
have,AUX
has,AUX
had,AUX
having,AUX
get,AUX
got,AUX
gotten,AUX
getting,AUX
going,AUX
gonna,AUX
gotta,AUX
wanna,AUX
good,ADJ
fine,ADJ
well,ADJ
great,ADJ
excellent,ADJ
bad,ADJ
okay,ADJ
ok,ADJ
happy,ADJ
sad,ADJ
excited,ADJ
bored,ADJ
tired,ADJ
busy,ADJ
free,ADJ
ready,ADJ
easy,ADJ
hard,ADJ
difficult,ADJ
complex,ADJ
fun,ADJ
funny,ADJ
nice,ADJ
beautiful,ADJ
lovely,ADJ
wonderful,ADJ
big,ADJ
small,ADJ
large,ADJ
little,ADJ
long,ADJ
short,ADJ
new,ADJ
old,ADJ
hot,ADJ
cold,ADJ
warm,ADJ
clean,ADJ
fresh,ADJ
quick,ADJ
slow,ADJ
better,ADJ
worse,ADJ
best,ADJ
worst,ADJ
more,ADJ
most,ADJ
less,ADJ
least,ADJ
very,ADJ
really,ADJ
quite,ADJ
just,ADJ
still,ADJ
already,ADJ
again,ADJ
always,ADJ
never,ADJ
often,ADJ
sometimes,ADJ
usually,ADJ
actually,ADJ
definitely,ADJ
probably,ADJ
maybe,ADJ
perhaps,ADJ
sure,ADJ
only,ADJ
also,ADJ
too,ADJ
even,ADJ
much,ADJ
many,ADJ
few,ADJ
lot,ADJ
lots,ADJ
pretty,ADJ
fairly,ADJ
kind,ADJ
sort,ADJ
bit,ADJ
enough,ADJ
so,ADJ
such,ADJ
both,ADJ
each,ADJ
every,ADJ
all,ADJ
any,ADJ
some,ADJ
local,ADJ
different,ADJ
same,ADJ
next,ADJ
last,ADJ
other,ADJ
own,ADJ
full,ADJ
whole,ADJ
main,ADJ
basic,ADJ
simple,ADJ
special,ADJ
first,ADJ
second,ADJ
several,ADJ
single,ADJ
important,ADJ
natural,ADJ
classic,ADJ
healthy,ADJ
perfect,ADJ
name,NOUN
gollemer,NOUN
bot,NOUN
assistant,NOUN
system,NOUN
ai,NOUN
machine,NOUN
human,NOUN
person,NOUN
people,NOUN
friend,NOUN
friends,NOUN
family,NOUN
thing,NOUN
things,NOUN
stuff,NOUN
way,NOUN
ways,NOUN
time,NOUN
day,NOUN
days,NOUN
week,NOUN
month,NOUN
year,NOUN
place,NOUN
home,NOUN
house,NOUN
room,NOUN
job,NOUN
school,NOUN
class,NOUN
project,NOUN
idea,NOUN
ideas,NOUN
food,NOUN
water,NOUN
coffee,NOUN
tea,NOUN
book,NOUN
books,NOUN
music,NOUN
life,NOUN
world,NOUN
city,NOUN
town,NOUN
country,NOUN
morning,NOUN
night,NOUN
habit,NOUN
hobby,NOUN
skill,NOUN
goal,NOUN
goals,NOUN
garden,NOUN
kitchen,NOUN
dog,NOUN
cat,NOUN
recipe,NOUN
hiking,NOUN
weekend,NOUN
vacation,NOUN
holiday,NOUN
trip,NOUN
weather,NOUN
sleep,NOUN
rest,NOUN
exercise,NOUN
health,NOUN
mind,NOUN
body,NOUN
heart,NOUN
energy,NOUN
money,NOUN
phone,NOUN
computer,NOUN
app,NOUN
game,NOUN
show,NOUN
movie,NOUN
podcast,NOUN
video,NOUN
photo,NOUN
message,NOUN
email,NOUN
question,NOUN
answer,NOUN
story,NOUN
point,NOUN
reason,NOUN
moment,NOUN
today,NOUN
yesterday,NOUN
tomorrow,NOUN
the,PREP
a,PREP
an,PREP
in,PREP
on,PREP
at,PREP
to,PREP
for,PREP
with,PREP
by,PREP
from,PREP
of,PREP
about,PREP
above,PREP
after,PREP
before,PREP
between,PREP
during,PREP
into,PREP
near,PREP
off,PREP
out,PREP
over,PREP
through,PREP
under,PREP
until,PREP
up,PREP
upon,PREP
within,PREP
without,PREP
and,PREP
but,PREP
or,PREP
nor,PREP
yet,PREP
because,PREP
if,PREP
when,PREP
while,PREP
although,PREP
though,PREP
since,PREP
unless,PREP
as,PREP
than,PREP
then,PREP
not,PREP
no,PREP
how,INTERROGATIVE
what,INTERROGATIVE
where,INTERROGATIVE
why,INTERROGATIVE
who,INTERROGATIVE
which,INTERROGATIVE
whose,INTERROGATIVE
whom,INTERROGATIVE`

var (
	lexiconMap  map[string]string
	lexiconOnce sync.Once
)

func initLexicon() {
	lexiconOnce.Do(func() {
		lexiconMap = make(map[string]string)
		r := csv.NewReader(strings.NewReader(lexiconCSV))
		records, err := r.ReadAll()
		if err == nil {
			for i, rec := range records {
				if i == 0 {
					continue
				}
				if len(rec) >= 2 {
					lexiconMap[strings.ToLower(strings.TrimSpace(rec[0]))] = strings.TrimSpace(rec[1])
				}
			}
		}
	})
}

// MapWordToGrammarType returns a coarse-grained tag for a word.
// This uses the Dynamic CSV Lexicon Loader.
func MapWordToGrammarType(w string) string {
	initLexicon()
	w = strings.ToLower(strings.Trim(w, ".,!?;:\"'()[]"))
	if t, ok := lexiconMap[w]; ok {
		return t
	}
	return "OTHER"
}
