package analyze

// The Material look for the webpage piece (editwebpage.go's
// "material" template, and the component piece that adds one
// component at a time to an existing page): Google's
// Material Design 3, written by hand in plain CSS — no
// framework, no library, nothing loaded from the web. The
// whole sheet is embedded in the page itself, in the
// package's self-contained skeleton style.
//
// The sheet is assembled from named chunks — the color
// roles, the base/type block, and one chunk per component —
// because the two pieces share it: the material template
// embeds every chunk at once, and the component piece
// inserts only the chunks a page is missing. One source,
// so the values can never drift between them.

import "strings"

// MaterialStyleMarker is the word the material styles carry
// in their opening comment. A page containing it already
// holds the color-role variables (written by the material
// template, or inserted with a first component); the
// component piece checks for it before adding CSS.
const MaterialStyleMarker = "gollemer-material"

// materialCSSVariables is the Material color system: the
// baseline light scheme as CSS variables named for the job
// each color does (Material's "color roles"), and the
// baseline dark scheme flipping the same variables when
// the reader's system asks for dark.
const materialCSSVariables = `/* Material Design 3, written by hand in plain CSS — no framework,
   no library, nothing loaded from elsewhere. The colors are
   variables named for the job each does in Material (its "color
   roles"), so changing one variable restyles every component
   that reads it. gollemer-material */
:root {
  --primary: #6750A4; /* the main color: filled buttons, the switch when on */
  --on-primary: #FFFFFF; /* words and symbols drawn on the primary color */
  --primary-container: #EADDFF; /* a soft wash of the primary color: tonal buttons, the floating button, picked chips */
  --on-primary-container: #21005D; /* words drawn on the primary container */
  --secondary: #625B71; /* a quieter second color, for less loud accents */
  --surface: #FEF7FF; /* the page's background */
  --on-surface: #1D1B20; /* the main words */
  --surface-variant: #E7E0EC; /* quiet fills: text fields */
  --outline: #79747E; /* borders on outlined things */
  --surface-container: #F3EDF7; /* raised areas: the app bar, cards */
}
@media (prefers-color-scheme: dark) {
  :root {
    --primary: #D0BCFF;
    --on-primary: #381E72;
    --primary-container: #4F378B;
    --on-primary-container: #EADDFF;
    --secondary: #CCC2DC;
    --surface: #141218;
    --on-surface: #E6E0E9;
    --surface-variant: #49454F;
    --outline: #938F99;
    --surface-container: #211F26;
  }
}
`

// materialCSSBase is the page frame and Material's type
// scale, as classes a beginner can read off the scale's own
// names: a headline, a title, body words, labels.
const materialCSSBase = `body { font-family: Roboto, "Segoe UI", Arial, sans-serif; font-size: 16px; line-height: 1.5; background: var(--surface); color: var(--on-surface); margin: 0 auto; max-width: 44em; padding: 0 1em 5em; }
.headline { font-size: 32px; line-height: 40px; font-weight: 400; }
.section-title { font-size: 22px; line-height: 28px; font-weight: 400; margin-top: 1.5em; }
.label { font-size: 14px; line-height: 20px; font-weight: 500; letter-spacing: 0.1px; }
`

// The per-component chunks. Each opens with its own marker
// comment — the component piece wraps its insertions the
// same way, so carried styles are recognizable in the page.

const materialCSSAppBar = `/* gollemer-material: app bar */
.app-bar { background: var(--surface-container); margin: 0 -1em 1em; padding: 0.75em 1em; }
.app-bar-title { font-size: 22px; line-height: 28px; }
`

const materialCSSButtons = `/* gollemer-material: buttons */
.btn { font-size: 14px; font-weight: 500; letter-spacing: 0.1px; border: none; border-radius: 999px; padding: 0.65em 1.6em; cursor: pointer; }
.btn-filled { background: var(--primary); color: var(--on-primary); }
.btn-tonal { background: var(--primary-container); color: var(--on-primary-container); }
.btn-outlined { background: transparent; color: var(--primary); border: 1px solid var(--outline); }
.btn-text { background: transparent; color: var(--primary); }
`

const materialCSSCard = `/* gollemer-material: card */
.card { background: var(--surface-container); border-radius: 12px; padding: 1em; }
.card-title { font-size: 22px; line-height: 28px; margin: 0 0 0.25em; }
.card p { margin: 0.25em 0; }
`

const materialCSSFab = `/* gollemer-material: floating button */
.fab { position: fixed; right: 1em; bottom: 1em; width: 56px; height: 56px; border: none; border-radius: 16px; background: var(--primary-container); color: var(--on-primary-container); font-size: 24px; cursor: pointer; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3); }
`

const materialCSSChips = `/* gollemer-material: chips */
.chip { display: inline-block; border: 1px solid var(--outline); border-radius: 8px; padding: 0.35em 0.9em; font-size: 14px; font-weight: 500; }
.chip-selected { background: var(--primary-container); color: var(--on-primary-container); border-color: transparent; }
`

const materialCSSField = `/* gollemer-material: text field */
.field label { display: block; font-size: 14px; font-weight: 500; margin-bottom: 0.25em; }
.field input { width: 100%; box-sizing: border-box; font-size: 16px; padding: 0.8em 1em; color: var(--on-surface); background: var(--surface-variant); border: 1px solid var(--outline); border-radius: 4px 4px 0 0; }
`

const materialCSSSwitch = `/* gollemer-material: switch */
.switch { appearance: none; -webkit-appearance: none; width: 52px; height: 32px; border-radius: 999px; border: 2px solid var(--outline); background: var(--surface-variant); position: relative; vertical-align: middle; cursor: pointer; margin: 0; }
.switch::after { content: ""; position: absolute; top: 50%; left: 5px; transform: translateY(-50%); width: 16px; height: 16px; border-radius: 50%; background: var(--outline); }
.switch:checked { background: var(--primary); border-color: var(--primary); }
.switch:checked::after { left: 26px; width: 20px; height: 20px; background: var(--on-primary); }
.switch-row { cursor: pointer; }
`

// materialCSSAll is the whole sheet the material template
// embeds: variables, base, then every component chunk.
func materialCSSAll() string {
	return materialCSSVariables + materialCSSBase +
		materialCSSAppBar + materialCSSButtons + materialCSSCard +
		materialCSSFab + materialCSSChips + materialCSSField + materialCSSSwitch
}

// The components' markup, one snippet each — the same
// strings the material template's demo sections are built
// from and the component piece inserts into a page.

func materialAppBarSnippet(title string) string {
	if title == "" {
		title = "TODO: the page's name"
	}
	return "<header class=\"app-bar\"><span class=\"app-bar-title\">" + title + "</span></header>\n"
}

const materialButtonSnippet = "<button class=\"btn btn-filled\" type=\"button\">TODO: the button's words</button>\n"

const materialTonalButtonSnippet = "<button class=\"btn btn-tonal\" type=\"button\">TODO: the button's words</button>\n"

const materialOutlinedButtonSnippet = "<button class=\"btn btn-outlined\" type=\"button\">TODO: the button's words</button>\n"

const materialTextButtonSnippet = "<button class=\"btn btn-text\" type=\"button\">TODO: the button's words</button>\n"

const materialCardSnippet = "<div class=\"card\">\n" +
	"<p class=\"card-title\">TODO: the card's heading</p>\n" +
	"<p>TODO: the card's words — a card groups one thing: a person, a product, a short note.</p>\n" +
	"</div>\n"

const materialFabSnippet = "<button class=\"fab\" type=\"button\" aria-label=\"TODO: say what this button does\">+</button>\n"

const materialChipSnippet = "<span class=\"chip\">TODO: the chip's words</span>\n"

const materialFieldSnippet = "<div class=\"field\">\n" +
	"<label>TODO: the field's label</label>\n" +
	"<input type=\"text\" placeholder=\"TODO: a hint for what to type\">\n" +
	"</div>\n"

const materialSwitchSnippet = "<label class=\"switch-row\"><input class=\"switch\" type=\"checkbox\"> TODO: what this switch turns on</label>\n"

// materialBody assembles the material template's page: the
// app bar, an intro in the skeleton voice, and one labeled
// section per component, so a beginner sees what each part
// is called. The floating button rides last, fixed to the
// corner by its CSS wherever it sits in the markup.
func materialBody(title string) string {
	var b strings.Builder
	b.WriteString(materialAppBarSnippet(title))
	b.WriteString("<p>TODO: what this page is — a starter page in Google's Material Design 3 look, written by hand in plain CSS in this file's style block: no framework, no library, and not one line of JavaScript. Below is the starter set of parts, each under its own name.</p>\n")
	b.WriteString("<h2 class=\"section-title\">Buttons</h2>\n")
	b.WriteString("<p>\n" + materialButtonSnippet + materialTonalButtonSnippet + materialOutlinedButtonSnippet + materialTextButtonSnippet + "</p>\n")
	b.WriteString("<h2 class=\"section-title\">Card</h2>\n")
	b.WriteString(materialCardSnippet)
	b.WriteString("<h2 class=\"section-title\">Chips</h2>\n")
	b.WriteString("<p><span class=\"chip chip-selected\">TODO: a picked chip</span> <span class=\"chip\">TODO: a chip</span> <span class=\"chip\">TODO: another chip</span></p>\n")
	b.WriteString("<h2 class=\"section-title\">Text field</h2>\n")
	b.WriteString(materialFieldSnippet)
	b.WriteString("<h2 class=\"section-title\">Switch</h2>\n")
	b.WriteString("<p>" + materialSwitchSnippet + "</p>\n")
	b.WriteString(materialFabSnippet)
	return b.String()
}
