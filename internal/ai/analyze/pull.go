package analyze

import (
	"path"
	"strings"
)

// DescribeFile returns a one-line plain-words description of what the
// project-relative Go file is for, used by the pull-request changed-files
// view. It describes what the file itself declares (its headline types
// and functions); test files are labeled as tests.
func (p *Project) DescribeFile(rel string) string {
	base := path.Base(rel)
	if strings.HasSuffix(base, "_test.go") {
		if pkg := p.packageForFile(rel); pkg != nil {
			return "tests for package " + pkg.Name
		}
		return "tests"
	}
	var types []*Type
	var funcs []*Func
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.File == rel {
				types = append(types, t)
			}
		}
		for _, f := range pkg.Funcs {
			if f.File == rel && !f.IsTest && !f.IsMain && !f.IsInit {
				funcs = append(funcs, f)
			}
		}
	}
	// Prefer an exported type with a doc comment: "the Config struct —
	// the app's settings".
	for _, t := range types {
		if t.Exported && t.Doc != "" {
			return "the " + t.Name + " " + t.Kind + " — " + firstSentence(t.Doc)
		}
	}
	for _, f := range funcs {
		if f.Exported && f.Doc != "" && f.Receiver == "" {
			return firstSentence(f.Doc)
		}
	}
	for _, t := range types {
		if t.Exported {
			return "the " + t.Name + " " + t.Kind
		}
	}
	for _, f := range funcs {
		if f.Exported && f.Receiver == "" {
			return f.Name + "()"
		}
	}
	if pkg := p.packageForFile(rel); pkg != nil {
		if pkg.Doc != "" {
			return "package " + pkg.Name + " — " + firstSentence(pkg.Doc)
		}
		return "part of package " + pkg.Name
	}
	return ""
}

// packageForFile returns the package containing the project-relative file.
func (p *Project) packageForFile(rel string) *Package {
	dir := path.Dir(rel)
	if dir == "." {
		dir = ""
	}
	best := (*Package)(nil)
	for _, pkg := range p.Packages {
		if pkg.Dir == dir || strings.HasPrefix(dir, pkg.Dir+"/") {
			if best == nil || len(pkg.Dir) > len(best.Dir) {
				best = pkg
			}
		}
	}
	return best
}
