// gen_conversing_pb generates data/training/trainingdata/conversing.pb from
// a hard-coded set of multi-turn conversations encoded in the
// datasetpb.ConversationDataset protobuf format.
//
// Each turn mirrors the conversing.csv columns:
//
//	conversation_id, turn_sequence, role, content
//
// Assistant turns embed the predictive reasoning trace inline in content,
// using the [PREDICTIVE_REASONING] ... [RESPONSE] layout.
//
// Usage:
//
//	go run ./cmd/tools/gen_conversing_pb
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
)

// reasoningContent formats a predictive reasoning trace the same way the
// conversing.csv rows do.
func reasoningContent(entities string, currentState, intent string, outcomes []string, goal, response string) string {
	s := "[PREDICTIVE_REASONING]\n"
	s += "- ENTITIES: " + entities + "\n"
	s += "- CURRENT_STATE: " + currentState + "\n"
	s += "- PREDICTED_USER_INTENT: " + intent + "\n"
	s += "- SIMULATED_OUTCOMES:\n"
	for _, o := range outcomes {
		s += "  * " + o + "\n"
	}
	s += "- TARGET_GOAL: " + goal + "\n"
	s += "[RESPONSE] " + response
	return s
}

func userTurn(seq int32, content string) *datasetpb.ConversationTurn {
	return &datasetpb.ConversationTurn{TurnSequence: seq, Role: datasetpb.Role_ROLE_USER, Content: content}
}

func assistantTurn(seq int32, content string) *datasetpb.ConversationTurn {
	return &datasetpb.ConversationTurn{TurnSequence: seq, Role: datasetpb.Role_ROLE_ASSISTANT, Content: content}
}

func main() {
	rootDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("getwd: %v", err)
	}

	ds := &datasetpb.ConversationDataset{
		Conversations: []*datasetpb.Conversation{
			{
				ConversationId: "conv_401",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "My roommate keeps leaving dirty dishes in the sink for days, and it's frustrating me."),
					assistantTurn(2, reasoningContent(
						"Subject=Roommate | Object=Dirty dishes / Shared space",
						"Conflict escalating due to unaddressed chore imbalance",
						"Wants resolution without damaging relationship",
						[]string{
							"Immediate confrontation -> Escalates defensiveness (Risk: High)",
							"Silent cleaning -> Causes user burnout (Risk: High)",
							"Proactive boundary framing -> Sets mutual expectations (Risk: Low)",
						},
						"Direct user toward proactive boundary framing",
						"Address the issue during a neutral moment rather than right when you're angry. Focus on expressing how a clean shared space benefits both of you, and propose a simple standard, like rinsing dishes or clearing the sink daily.",
					)),
				},
			},
			{
				ConversationId: "conv_402",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "A close friend asked to borrow $500, but they rarely pay people back."),
					assistantTurn(2, reasoningContent(
						"Subject=Friend | Object=Loan request ($500)",
						"Financial risk creating potential relationship strain",
						"Wants to protect finances without losing the friendship",
						[]string{
							"Blunt refusal -> Causes immediate offense and emotional distance (Risk: High)",
							"Lending money unconditionally -> High likelihood of financial loss and resentment (Risk: High)",
							"Empathetic boundary setting -> Protects finances while offering non-monetary support (Risk: Low)",
						},
						"Guide user to set financial boundaries politely without shame",
						"Express care for their situation first, but be clear that your current financial policy is not to lend money to friends. Offer to help them brainstorm other solutions or review their options instead.",
					)),
				},
			},
			{
				ConversationId: "conv_403",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "My coworker keeps taking credit for my ideas in meetings and I don't know how to handle it."),
					assistantTurn(2, reasoningContent(
						"Subject=Coworker | Object=Credit for user's ideas",
						"Professional recognition being diverted, eroding trust and morale",
						"Wants recognition while avoiding workplace conflict",
						[]string{
							"Publicly calling them out in the meeting -> Creates open conflict and may embarrass the user (Risk: High)",
							"Saying nothing -> Reinforces the behavior and breeds resentment (Risk: High)",
							"Documenting ideas and privately addressing the pattern -> Protects credit while staying professional (Risk: Low)",
						},
						"Help the user reclaim credit through calm, documented communication",
						"Start by sending your ideas in writing before meetings so there's a record of authorship. Then have a private, non-accusatory conversation with your coworker about collaboration and visibility. If it continues, loop in your manager with the documented examples.",
					)),
				},
			},
			{
				ConversationId: "conv_404",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "My parents keep commenting on my career choices and it makes me doubt myself."),
					assistantTurn(2, reasoningContent(
						"Subject=Parents | Object=Career criticism",
						"Family pressure undermining user's confidence and autonomy",
						"Wants validation of their path without severing family ties",
						[]string{
							"Arguing to defend the choices -> Turns visits into debates and deepens the rift (Risk: High)",
							"Hiding career news entirely -> Creates distance and secrecy (Risk: Medium)",
							"Setting a respectful conversational boundary -> Preserves the relationship and self-trust (Risk: Low)",
						},
						"Help the user hold a calm boundary with family while staying confident in their path",
						"Acknowledge that they likely mean well, then let them know career topics are off the table for advice. Redirect to things you can share, and remind yourself that their anxiety is not a verdict on your choices.",
					)),
				},
			},
			{
				ConversationId: "conv_405",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "I agreed to host a big holiday dinner but I'm completely overwhelmed and regretting it."),
					assistantTurn(2, reasoningContent(
						"Subject=User | Object=Holiday hosting commitment",
						"Overcommitment causing stress ahead of a family event",
						"Wants relief from the burden without disappointing everyone",
						[]string{
							"Canceling outright -> Disappoints guests and adds guilt (Risk: High)",
							"Pushing through alone -> Guarantees burnout on the day (Risk: High)",
							"Delegating dishes and scaling down the menu -> Keeps the event while cutting the load (Risk: Low)",
						},
						"Help the user scale the event down and share the load gracefully",
						"Reach out to guests and turn it into a potluck where everyone brings a dish. Cut the menu to a few essentials and accept that a simpler gathering still delivers the part people actually remember.",
					)),
				},
			},
			{
				ConversationId: "conv_406",
				Turns: []*datasetpb.ConversationTurn{
					userTurn(1, "My neighbor's dog barks all night and I haven't slept properly in a week."),
					assistantTurn(2, reasoningContent(
						"Subject=Neighbor | Object=Nighttime dog barking",
						"Sleep deprivation straining a relationship that must remain workable",
						"Wants quiet nights without triggering a neighborhood feud",
						[]string{
							"Leaving an angry note -> Invites defensiveness and escalation (Risk: High)",
							"Calling authorities immediately -> Poisons the relationship permanently (Risk: High)",
							"Friendly conversation first, escalating only if needed -> Solves most cases while preserving goodwill (Risk: Low)",
						},
						"Guide the user toward a friendly, stepwise resolution",
						"Knock on their door at a reasonable hour and mention it kindly; they may genuinely not hear it. Suggest white noise machines or keeping the dog indoors at night, and only involve the landlord or authorities if the pattern continues after a fair warning.",
					)),
				},
			},
		},
	}

	outPath := filepath.Join(rootDir, "data", "training", "trainingdata", "conversing.pb")
	if err := datasetpb.SaveConversationDatasetToProto(outPath, ds); err != nil {
		log.Fatalf("save conversing.pb: %v", err)
	}
	fmt.Printf("\u2705 wrote %s (%d conversations)\n", outPath, len(ds.Conversations))
}
