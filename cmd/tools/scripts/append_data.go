package main

import (
	"encoding/csv"
	"os"
)

func main() {
	newRows := [][]string{
		{"conv_051", "1", "system", "You are a helpful assistant."},
		{"conv_051", "2", "user", "My dog's name is Barnaby."},
		{"conv_051", "3", "assistant", "Nice to meet you! Barnaby is a great name for a dog. What breed is he?"},
		{"conv_051", "4", "user", "He is a golden retriever. Also, I live in Seattle."},
		{"conv_051", "5", "assistant", "Seattle is a beautiful city, though it rains a lot. Barnaby must love running in the parks there."},
		{"conv_051", "6", "user", "What is my dog's name and where do I live?"},
		{"conv_051", "7", "assistant", "Your dog's name is Barnaby and you live in Seattle."},

		{"conv_052", "1", "system", "You are a helpful assistant."},
		{"conv_052", "2", "user", "I am planning a trip to Japan next month."},
		{"conv_052", "3", "assistant", "That sounds exciting! Japan is wonderful. Which cities are you visiting?"},
		{"conv_052", "4", "user", "I will be visiting Tokyo, Kyoto, and Osaka."},
		{"conv_052", "5", "assistant", "A classic itinerary! Tokyo is bustling, Kyoto is historic, and Osaka has amazing food."},
		{"conv_052", "6", "user", "Which three cities did I say I was going to visit?"},
		{"conv_052", "7", "assistant", "You mentioned you will be visiting Tokyo, Kyoto, and Osaka."},

		{"conv_053", "1", "system", "You are a helpful assistant."},
		{"conv_053", "2", "user", "My favorite color is cerulean blue, and my lucky number is 42."},
		{"conv_053", "3", "assistant", "Cerulean blue is a very calming color, and 42 is famous as the answer to the ultimate question of life, the universe, and everything!"},
		{"conv_053", "4", "user", "I am thinking of painting my bedroom."},
		{"conv_053", "5", "assistant", "Are you considering painting it cerulean blue, since that is your favorite color?"},
		{"conv_053", "6", "user", "Yes! Can you remind me what my lucky number is, maybe I will paint that on the door."},
		{"conv_053", "7", "assistant", "Your lucky number is 42, which would look great painted on your bedroom door."},

		{"conv_054", "1", "system", "You are a helpful assistant."},
		{"conv_054", "2", "user", "I just bought a new car. It is a red Toyota Corolla."},
		{"conv_054", "3", "assistant", "Congratulations on the new car! The Toyota Corolla is very reliable. Do you enjoy the red color?"},
		{"conv_054", "4", "user", "I love it. Oh, and I named the car Ruby."},
		{"conv_054", "5", "assistant", "Ruby is a perfect name for a red car. Have you taken Ruby on a road trip yet?"},
		{"conv_054", "6", "user", "What kind of car do I drive and what did I name it?"},
		{"conv_054", "7", "assistant", "You drive a red Toyota Corolla and you named it Ruby."},
	}

	f, err := os.OpenFile("/home/zendrulat/g/gollemer/data/training/trainingdata/conversations.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	// Force all fields to be quoted to match the existing format
	writer.UseCRLF = false
	for _, row := range newRows {
		if err := writer.Write(row); err != nil {
			panic(err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		panic(err)
	}
}
