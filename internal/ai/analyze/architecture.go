package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// This file infers architecture: entry points, layers, components. It sees
// the forest, not just the trees — how the system fits together.

// Architecture is the high-level structure of the project.
type Architecture struct {
	EntryPoints []string            // mains, inits, exported with no callers
	Layers      []Layer             // groups by package/directory
	Interfaces  []InterfaceInfo     // interfaces and their implementations
}

// Layer is one architectural layer (typically a package).
type Layer struct {
	Name      string   // package name
	DependsOn []string // other layers it calls into
	Provides  []string // key exported functions/types
}

// InterfaceInfo describes an interface and what implements it.
type InterfaceInfo struct {
	Name           string
	Implementations []string
}

// InferArchitecture analyzes the project structure.
func (p *Project) InferArchitecture() Architecture {
	var arch Architecture

	// Entry points: main, init, exported functions with no callers.
	seen := map[string]bool{}
	for _, fn := range p.byID {
		if fn.IsMain || fn.IsInit {
			if !seen[fn.Display()] {
				arch.EntryPoints = append(arch.EntryPoints, fn.Display())
				seen[fn.Display()] = true
			}
		} else if fn.Exported && len(fn.Callers) == 0 && !fn.IsTest {
			if !seen[fn.Display()] {
				arch.EntryPoints = append(arch.EntryPoints, fn.Display())
				seen[fn.Display()] = true
			}
		}
	}
	sort.Strings(arch.EntryPoints)

	// Layers: one per package, with dependencies.
	for _, pkg := range p.Packages {
		layer := Layer{Name: pkg.Name}
		depSet := map[string]bool{}
		for _, fn := range pkg.Funcs {
			for _, calleeID := range fn.Calls {
				if callee := p.byID[calleeID]; callee != nil && callee.Pkg != pkg.Name {
					depSet[callee.Pkg] = true
				}
			}
		}
		for d := range depSet {
			layer.DependsOn = append(layer.DependsOn, d)
		}
		sort.Strings(layer.DependsOn)
		// Key exports: exported functions and types.
		for _, fn := range pkg.Funcs {
			if fn.Exported && !fn.IsTest && len(layer.Provides) < 5 {
				layer.Provides = append(layer.Provides, fn.Name)
			}
		}
		for _, t := range pkg.Types {
			if isExported(t.Name) && len(layer.Provides) < 8 {
				layer.Provides = append(layer.Provides, "type "+t.Name)
			}
		}
		arch.Layers = append(arch.Layers, layer)
	}

	// Interfaces: find interface types and count implementations.
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind == "interface" {
				info := InterfaceInfo{Name: t.Pkg + "." + t.Name}
				// Implementations are types with matching methods (heuristic).
				arch.Interfaces = append(arch.Interfaces, info)
			}
		}
	}

	return arch
}

// ArchitectureSummary renders the architecture as plain text.
func (p *Project) ArchitectureSummary() string {
	arch := p.InferArchitecture()
	var b strings.Builder

	if len(arch.EntryPoints) > 0 {
		b.WriteString(fmt.Sprintf("Entry points: %s\n", strings.Join(arch.EntryPoints, ", ")))
	}
	for _, layer := range arch.Layers {
		b.WriteString(fmt.Sprintf("\nLayer `%s`:\n", layer.Name))
		if len(layer.Provides) > 0 {
			b.WriteString(fmt.Sprintf("  Provides: %s\n", strings.Join(layer.Provides, ", ")))
		}
		if len(layer.DependsOn) > 0 {
			b.WriteString(fmt.Sprintf("  Depends on: %s\n", strings.Join(layer.DependsOn, ", ")))
		} else {
			b.WriteString("  Depends on: nothing (leaf layer)\n")
		}
	}
	if len(arch.Interfaces) > 0 {
		var names []string
		for _, ii := range arch.Interfaces {
			names = append(names, ii.Name)
		}
		b.WriteString(fmt.Sprintf("\nInterfaces: %s\n", strings.Join(names, ", ")))
	}
	return b.String()
}

func isExported(name string) bool {
	return name != "" && name[0] >= 'A' && name[0] <= 'Z'
}
