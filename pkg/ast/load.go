// Shared module loading for AST analyses (impact, miner).
//
// loadModulePackages loads every package in the module containing dir
// (found via go.mod walk-up) with full type information. Analyses that
// need cross-package references share this loader so the load mode
// stays in sync in one place.
package ast

import (
	"fmt"
	"path/filepath"

	"golang.org/x/tools/go/packages"
)

// loadModulePackages returns the module root and all loaded packages
// for the module containing dir.
func loadModulePackages(dir string) (moduleRoot string, pkgs []*packages.Package, err error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", nil, fmt.Errorf("resolve dir %q: %w", dir, err)
	}
	root, err := findModuleRoot(abs)
	if err != nil {
		return "", nil, err
	}
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedTypes |
			packages.NeedTypesInfo |
			packages.NeedSyntax |
			packages.NeedModule,
		Dir: root,
	}
	pkgs, err = packages.Load(cfg, "./...")
	if err != nil {
		return "", nil, fmt.Errorf("packages.Load: %w", err)
	}
	return root, pkgs, nil
}
