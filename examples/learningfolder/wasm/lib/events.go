//go:build js && wasm


package lib

import "syscall/js"

func AlertUser(msg string) {
	js.Global().Get("alert").Invoke(msg)
}
