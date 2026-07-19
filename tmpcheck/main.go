package main
import (
  "fmt"
  "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)
func main(){
  v:=vocab.NewVocabulary()
  for _, word := range []string{"hello","world","the","quick","brown","fox","jumped","over","lazy","dog","today","tomorrow","learn","model","token","routing","expert"} {
    v.AddToken(word)
  }
  fmt.Println("hello id", v.GetTokenID("hello"), "word", v.GetWord(v.GetTokenID("hello")))
  fmt.Println("world id", v.GetTokenID("world"), "word", v.GetWord(v.GetTokenID("world")))
  fmt.Println("tokens", v.TokenToWord)
}
