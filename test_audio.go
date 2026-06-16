package main
import (
    "fmt"
    "github.com/gordonklaus/portaudio"
)
func main() {
    portaudio.Initialize()
    defer portaudio.Terminate()
    in := make([]int16, 1024)
    stream, err := portaudio.OpenDefaultStream(1, 0, 16000, len(in), in)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer stream.Close()
    err = stream.Start()
    if err != nil {
        fmt.Println("Start Error:", err)
        return
    }
    fmt.Println("Successfully opened portaudio stream while dabri is recording!")
}
