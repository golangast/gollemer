package colors

import (
	"fmt"
	"time"
)

// Colorize applies foreground and background colors to a string and prints it.
func Colorize(ForegroundColor, BackgroundColor, line, prefix string, blink bool) {
	fcolor, ok := fgColors[ForegroundColor]
	if !ok {
		fcolor = fgColors["default"] // Default to reset
	}

	bcolor, ok := bgColors[BackgroundColor]
	if !ok {
		bcolor = bgColors["default"] // Default to reset
	}

	blinkCode := ""
	if blink {
		blinkCode = "5;"
	}

	combineCode := ""
	if bcolor != "" {
		combineCode = bcolor
	}
	if fcolor != 0 {
		if combineCode != "" {
			combineCode += ";"
		}
		combineCode += fmt.Sprintf("%d", fcolor)
	}
	if blinkCode != "" {
		if combineCode != "" {
			combineCode += ";"
		}
		combineCode += blinkCode[:len(blinkCode)-1] // remove trailing semicolon
	}

	fmt.Printf("%s\x1b[%sm%s\x1b[0m", prefix, combineCode, line)
}

// ShowSpinner displays a terminal spinner with a message for a given duration.
func ShowSpinner(message, fgColor, bgColor string, duration time.Duration) {
	spinnerChars := []rune{'—', '\\', '|', '/'}
	i := 0
	clearLine := "\r\033[K" // Carriage return and clear line from cursor to end

	startTime := time.Now()
	for time.Since(startTime) < duration {
		spinnerChar := string(spinnerChars[i%len(spinnerChars)])

		fcolor, ok := fgColors[fgColor]
		if !ok {
			fcolor = fgColors["default"]
		}

		bcolor, ok := bgColors[bgColor]
		if !ok {
			bcolor = bgColors["default"]
		}

		fmt.Printf("%s\x1b[%s;%dm%s %s\x1b[0m", clearLine, bcolor, fcolor, spinnerChar, message)

		i++
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Print(clearLine) // Clear the spinner line after duration
}

// AnimatedOutput first shows a spinner for a duration, then prints the final output.
func AnimatedOutput(ForegroundColor, BackgroundColor, line string, animationDuration time.Duration) {
	ShowSpinner("Processing...", ForegroundColor, BackgroundColor, animationDuration)
	ColorizeOutPut(ForegroundColor, BackgroundColor, line)
}

var fgColors = map[string]int{
	"black":   30,
	"red":     31,
	"green":   32,
	"yellow":  33,
	"blue":    34,
	"magenta": 35,
	"cyan":    36,
	"white":   37,
	"default": 39, // Reset foreground color

	// Bright colors
	"bblack":   90,
	"bred":     91,
	"bgreen":   92,
	"byellow":  93,
	"bblue":    94,
	"bmagenta": 95,
	"bcyan":    96,
	"bwhite":   97,
}

var bgColors = map[string]string{
	"black":   "40",
	"red":     "41",
	"green":   "42",
	"yellow":  "43",
	"blue":    "44",
	"magenta": "45",
	"cyan":    "46",
	"white":   "47",
	"default": "49", // Reset background color

	// Bright background colors
	"bblack":   "100",
	"bred":     "101",
	"bgreen":   "102",
	"byellow":  "103",
	"bblue":    "104",
	"bmagenta": "105",
	"bcyan":    "106",
	"bwhite":   "107",
}

func ColorizeCol(ForegroundColor, BackgroundColor, line string) {
	Colorize(ForegroundColor, BackgroundColor, line, "", false)
	fmt.Printf("\x1b[0m\n") // Ensure reset and newline after Col
}

func ColorizeOutPut(ForegroundColor, BackgroundColor, line string) {
	Colorize(ForegroundColor, BackgroundColor, line, " ", false)
	fmt.Printf("\x1b[0m\n") // Ensure reset and newline after Output
}
