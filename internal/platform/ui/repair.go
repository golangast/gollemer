package ui

import (
	"fmt"
	"strings"
)

// ProposeMove interacts with the user to move a misplaced webserver
func (m *Mascot) ProposeMove(name string, moveFunc func() error) {
	m.Say(Shocked, fmt.Sprintf("I found '%s' in the root, but I usually look in 'cmd/'.", name))
	fmt.Printf("%s >> Should I move it to cmd/%s for better organization? (y/n): %s", m.Color, name, ColorReset)

	var input string
	fmt.Scanln(&input)

	if strings.ToLower(input) == "y" || strings.ToLower(input) == "yes" {
		m.Say(Thinking, "Restructuring project...")
		if err := moveFunc(); err != nil {
			m.Say(Disturbed, "I failed to move it: "+err.Error())
		} else {
			m.Say(Happy, "Moved! Now running '"+name+"' from its proper home.")
		}
	} else {
		m.Say(Neutral, "Okay, keeping it where it is. I'll adapt my search next time!")
	}
}

// ConfirmRepair asks for permission to execute a repair function
func (m *Mascot) ConfirmRepair(issue string, repairFunc func() error) {
	m.Say(Alert, fmt.Sprintf("I noticed an issue: %s", issue))
	fmt.Printf("%s >> Should I attempt to fix this automatically? (y/n): %s", m.Color, ColorReset)

	var input string
	fmt.Scanln(&input)

	if strings.ToLower(input) == "y" || strings.ToLower(input) == "yes" {
		m.Say(Thinking, "Applying fix...")
		if err := repairFunc(); err != nil {
			m.Say(Disturbed, "Repair failed: "+err.Error())
		} else {
			m.Say(Happy, "Done! The system should be healthy now.")
		}
	}
}
