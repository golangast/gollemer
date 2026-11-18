//go:build !linux

package chat

import "os/exec"

// platformOpenBrowser opens url in the user's default
// browser with the macOS `open` launcher. Started, not
// waited on; a Wait goroutine reaps it. On a platform with
// no such launcher the Start error surfaces and the caller
// prints the address plainly instead — the honest fallback
// everywhere.
func platformOpenBrowser(url string) error {
	cmd := exec.Command("open", url)
	if err := cmd.Start(); err != nil {
		return err
	}
	go cmd.Wait()
	return nil
}
