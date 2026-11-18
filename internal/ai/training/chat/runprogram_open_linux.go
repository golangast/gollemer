//go:build linux

package chat

import "os/exec"

// platformOpenBrowser opens url in the user's default
// browser with xdg-open — the Linux desktop's own launcher
// (on John's Chromebook it hands the address to Chrome). The
// launcher is started, not waited on: it forks the browser
// and returns; a Wait goroutine reaps it so it never lingers
// as a zombie.
func platformOpenBrowser(url string) error {
	cmd := exec.Command("xdg-open", url)
	if err := cmd.Start(); err != nil {
		return err
	}
	go cmd.Wait()
	return nil
}
