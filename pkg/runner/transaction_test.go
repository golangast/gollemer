package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func txFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/txdemo\n\ngo 1.26.0\n")
	write("add.go", "package demo\n\n// Add returns the sum of a and b.\nfunc Add(a, b int) int { return a + b }\n")
	return dir
}

func readFile(t *testing.T, dir, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(data)
}

func TestApplyPatchSetCommitsOnPass(t *testing.T) {
	dir := txFixture(t)
	var ps PatchSet
	ps.Add("add_test.go", "package demo\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif got := Add(2, 2); got != 4 {\n\t\tt.Errorf(\"got %d, want 4\", got)\n\t}\n}\n")
	if ps.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", ps.Len())
	}
	res, err := ApplyPatchSet(dir, ps)
	if err != nil {
		t.Fatalf("ApplyPatchSet: %v", err)
	}
	if res == nil || !res.Passed {
		t.Fatalf("expected pass, got %+v", res)
	}
	if got := readFile(t, dir, "add_test.go"); !strings.Contains(got, "func TestAdd") {
		t.Errorf("committed file missing test:\n%s", got)
	}
}

func TestApplyPatchSetNoCommitOnFailure(t *testing.T) {
	dir := txFixture(t)
	var ps PatchSet
	ps.Add("broken_test.go", "package demo\n\nimport \"testing\"\n\nfunc TestBroken(t *testing.T) {\n\tif Add(1) != 1 {\n\t\tt.Error(\"wrong\")\n\t}\n}\n")
	res, err := ApplyPatchSet(dir, ps)
	if err != nil {
		t.Fatalf("ApplyPatchSet: %v", err)
	}
	if res == nil || res.Passed {
		t.Fatalf("expected validation failure, got %+v", res)
	}
	if len(res.Errors) == 0 {
		t.Error("expected parsed errors in the result")
	}
	if _, err := os.Stat(filepath.Join(dir, "broken_test.go")); !os.IsNotExist(err) {
		t.Error("failed patch was committed to disk")
	}
	// The pre-existing file must be untouched.
	if got := readFile(t, dir, "add.go"); !strings.Contains(got, "func Add") {
		t.Error("pre-existing add.go was modified")
	}
}

func TestApplyPatchSetValidationErrors(t *testing.T) {
	dir := txFixture(t)
	if _, err := ApplyPatchSet("", PatchSet{Files: map[string]string{"a.go": "package demo\n"}}); err == nil {
		t.Error("expected error for empty targetDir")
	}
	if _, err := ApplyPatchSet(dir, PatchSet{}); err == nil {
		t.Error("expected error for empty patch set")
	}
	var ps PatchSet
	ps.Add("../escape.go", "package demo\n")
	if _, err := ApplyPatchSet(dir, ps); err == nil {
		t.Error("expected error for escaping path")
	}
}

func TestCommitPatchesRollsBack(t *testing.T) {
	dir := txFixture(t)
	// "sub" is a regular file, so creating sub/x.go must fail.
	if err := os.WriteFile(filepath.Join(dir, "sub"), []byte("not a dir"), 0o644); err != nil {
		t.Fatal(err)
	}
	keepOrig := "package demo\n\n// Keep is precious.\nfunc Keep() int { return 1 }\n"
	if err := os.WriteFile(filepath.Join(dir, "keep.go"), []byte(keepOrig), 0o644); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"a.go":     "package demo\n",
		"keep.go":  "package demo\n\nfunc Keep() int { return 2 }\n",
		"sub/x.go": "package demo\n",
	}
	err := commitPatches(dir, files, []string{"a.go", "keep.go", "sub/x.go"})
	if err == nil {
		t.Fatal("expected commit failure")
	}
	if _, serr := os.Stat(filepath.Join(dir, "a.go")); !os.IsNotExist(serr) {
		t.Error("rollback did not remove newly created a.go")
	}
	if got := readFile(t, dir, "keep.go"); got != keepOrig {
		t.Errorf("rollback did not restore keep.go:\n%s", got)
	}
	if got := readFile(t, dir, "sub"); got != "not a dir" {
		t.Error("rollback damaged the blocking file")
	}
}
