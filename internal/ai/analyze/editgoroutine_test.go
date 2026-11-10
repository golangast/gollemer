package analyze

import (
	"strings"
	"testing"
)

// The goroutine-result planner's pins: the exact assembled
// transformation, the collision discipline, the position
// pick, and every v1 refusal — each a plain answer, never
// a silent bend.

const goroutineGoMod = "module example.com/goroutine\n\ngo 1.21\n"

func planGoroutine(t *testing.T, proj *Project, req EditRequest) (*EditPlan, string) {
	t.Helper()
	req.Kind = EditGoroutineResult
	return proj.PlanEdit(req)
}

// John's live shape: main holding one anonymous goroutine
// with a TODO body. The whole assembled file is pinned.
func TestPlanGoroutineResultGolden(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"jim/main.go": "package main\n\nfunc main() {\n" +
			"\tgo func() {\n" +
			"\t\t// TODO: work goes here\n" +
			"\t}()\n" +
			"}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "jim/main.go", GoroutineType: "string",
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	if len(plan.Files) != 1 {
		t.Fatalf("plan files = %+v, want the one file", plan.Files)
	}
	want := "package main\n\nfunc main() {\n" +
		"\tresultCh := make(chan string)\n" +
		"\tgo func() {\n" +
		"\t\t// TODO: work goes here\n" +
		"\t\t// TODO: the goroutine's real result goes here — \"\" is only the zero value, sent until the goroutine has one\n" +
		"\t\tresultCh <- \"\"\n" +
		"\t}()\n" +
		"\tresult := <-resultCh\n" +
		"\t_ = result\n" +
		"}\n"
	if got := string(plan.Files[0].After); got != want {
		t.Errorf("assembled file:\n%s\nwant:\n%s", got, want)
	}
	if !strings.Contains(plan.Summary, "Give the goroutine in main") {
		t.Errorf("summary = %q", plan.Summary)
	}
	if !strings.Contains(plan.VerifyDesc, "sends its string back on the resultCh channel") {
		t.Errorf("verify desc = %q", plan.VerifyDesc)
	}
	applyServePlan(t, dir, plan)
}

// A one-line empty literal opens up: the send lands inside
// it, framed by the TODO.
func TestPlanGoroutineResultOneLineLiteral(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":  goroutineGoMod,
		"main.go": "package main\n\nfunc main() {\n\tgo func() {}()\n}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "main.go", GoroutineType: "string",
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	got := string(plan.Files[0].After)
	for _, want := range []string{
		"\tresultCh := make(chan string)\n",
		"\tgo func() {\n",
		"\t\tresultCh <- \"\"\n",
		"\t}()\n\tresult := <-resultCh\n\t_ = result\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("assembled file missing %q:\n%s", want, got)
		}
	}
}

// A goroutine nested in an if: the channel lines land at
// the go statement's own indentation, inside the block.
func TestPlanGoroutineResultNestedInIf(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"main.go": "package main\n\nfunc main() {\n" +
			"\tready := true\n" +
			"\tif ready {\n" +
			"\t\tgo func() {\n" +
			"\t\t\tprintln(\"work\")\n" +
			"\t\t}()\n" +
			"\t}\n" +
			"\t_ = ready\n" +
			"}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "main.go", GoroutineType: "int",
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	got := string(plan.Files[0].After)
	for _, want := range []string{
		"\t\tresultCh := make(chan int)\n",
		"\t\t\tresultCh <- 0\n",
		"\t\tresult := <-resultCh\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("assembled file missing %q:\n%s", want, got)
		}
	}
	applyServePlan(t, dir, plan)
}

// Names the function already uses are skipped: the pair
// counts up together, deterministically.
func TestPlanGoroutineResultNameCollision(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"main.go": "package main\n\nfunc main() {\n" +
			"\tresultCh := make(chan int)\n" +
			"\tresult := 0\n" +
			"\tgo func() {\n" +
			"\t\tprintln(\"work\")\n" +
			"\t}()\n" +
			"\t_ = resultCh\n" +
			"\t_ = result\n" +
			"}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "main.go", GoroutineType: "string",
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	got := string(plan.Files[0].After)
	for _, want := range []string{
		"resultCh2 := make(chan string)",
		"resultCh2 <- \"\"",
		"result2 := <-resultCh2",
		"_ = result2",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("assembled file missing %q:\n%s", want, got)
		}
	}
}

// The position picks exactly one goroutine — the second —
// and leaves the first untouched.
func TestPlanGoroutineResultOrdinal(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"main.go": "package main\n\nfunc main() {\n" +
			"\tgo func() {\n\t\tprintln(\"one\")\n\t}()\n" +
			"\tgo func() {\n\t\tprintln(\"two\")\n\t}()\n" +
			"}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "main.go", GoroutineType: "bool", GoroutineOrdinal: 2,
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	got := string(plan.Files[0].After)
	if n := strings.Count(got, "resultCh <- false"); n != 1 {
		t.Fatalf("sends = %d, want exactly 1:\n%s", n, got)
	}
	two := strings.Index(got, `println("two")`)
	send := strings.Index(got, "resultCh <- false")
	if send < two {
		t.Errorf("the send landed in the wrong goroutine:\n%s", got)
	}
}

// A function name scopes the search: the goroutine in
// helper is found while main has none.
func TestPlanGoroutineResultNamedFunction(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"main.go": "package main\n\nfunc main() { helper() }\n\n" +
			"func helper() {\n\tgo func() {\n\t\tprintln(\"work\")\n\t}()\n}\n",
	})
	plan, refusal := planGoroutine(t, proj, EditRequest{
		File: "main.go", FuncName: "helper", GoroutineType: "string",
	})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	if !strings.Contains(plan.VerifyDesc, "the goroutine in helper") {
		t.Errorf("verify desc = %q", plan.VerifyDesc)
	}
	// And the bare-file ask declines: main alone has no
	// goroutine, but the file does — with no function
	// named the scope is the file, so it plans on helper's.
	plan2, refusal2 := planGoroutine(t, proj, EditRequest{
		File: "main.go", GoroutineType: "string",
	})
	if refusal2 != "" || plan2 == nil {
		t.Fatalf("file-scoped ask refused: %s", refusal2)
	}
}

// Every refusal, pinned by its telling words.
func TestPlanGoroutineResultRefusals(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		req   EditRequest
		want  string
	}{
		{
			name: "no goroutine",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() { println(\"hi\") }\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "main.go — this piece gives an existing goroutine",
		},
		{
			name: "no goroutine in the named function",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() { println(\"hi\") }\n",
			},
			req:  EditRequest{File: "main.go", FuncName: "main", GoroutineType: "string"},
			want: "no goroutine in main in",
		},
		{
			name: "function not found",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {}\n",
			},
			req:  EditRequest{File: "main.go", FuncName: "nope", GoroutineType: "string"},
			want: "no function named \"nope\"",
		},
		{
			name: "named call",
			files: map[string]string{
				"main.go": "package main\n\nfunc work() {}\n\nfunc main() {\n\tgo work()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "runs work — its result would have to come from inside work",
		},
		{
			name: "takes a parameter",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tgo func(n int) {\n\t\t_ = n\n\t}(3)\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "takes a value in",
		},
		{
			name: "declares a result",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tgo func() int {\n\t\treturn 1\n\t}()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "int"},
			want: "already declares a result",
		},
		{
			name: "already sends",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tdone := make(chan bool)\n\tgo func() {\n\t\tdone <- true\n\t}()\n\t<-done\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "already sends on a channel",
		},
		{
			name: "several, no position",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tgo func() {}()\n\tgo func() {}()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "say which one, by its position",
		},
		{
			name: "position past the end",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tgo func() {}()\n\tgo func() {}()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string", GoroutineOrdinal: 3},
			want: "no third one",
		},
		{
			name: "type outside the set",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tgo func() {}()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "byte"},
			want: "isn't a result type I can build this with yet",
		},
		{
			name: "go statement shares its line",
			files: map[string]string{
				"main.go": "package main\n\nfunc main() {\n\tx := 1; go func() {\n\t\tprintln(x)\n\t}()\n}\n",
			},
			req:  EditRequest{File: "main.go", GoroutineType: "string"},
			want: "shares its line with other code",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			files := map[string]string{"go.mod": goroutineGoMod}
			for k, v := range tc.files {
				files[k] = v
			}
			_, proj := serveProject(t, files)
			plan, refusal := planGoroutine(t, proj, tc.req)
			if plan != nil {
				t.Fatalf("planned despite %s", tc.name)
			}
			if !strings.Contains(refusal, tc.want) {
				t.Errorf("refusal = %q, want it to say %q", refusal, tc.want)
			}
		})
	}
}

// The site listing the chat layer's question reads from.
func TestGoroutineResultSites(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": goroutineGoMod,
		"main.go": "package main\n\n" +
			"func main() {\n\tgo func() {}()\n}\n\n" +
			"func helper() {\n\tgo func() {}()\n\tgo work()\n}\n\n" +
			"func work() {}\n",
	})
	all := proj.GoroutineResultSites("main.go", "")
	if len(all) != 3 || all[0].Func != "main" || all[1].Func != "helper" || all[2].Func != "helper" {
		t.Fatalf("sites = %+v, want main + helper + helper", all)
	}
	if all[0].Line != 4 || all[1].Line != 8 {
		t.Errorf("site lines = %d, %d — want 4, 8", all[0].Line, all[1].Line)
	}
	help := proj.GoroutineResultSites("main.go", "helper")
	if len(help) != 2 {
		t.Fatalf("helper sites = %+v, want 2", help)
	}
	if got := proj.GoroutineResultSites("main.go", "nope"); got != nil {
		t.Errorf("unknown function sites = %+v, want nil", got)
	}
}
