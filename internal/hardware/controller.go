package hardware

import (
	"fmt"
)

// IntentPayload reflects the structure of your trained CSV/JSON data
type IntentPayload struct {
	Intent string            `json:"intent"`
	Roles  map[string]string `json:"roles"`
}

// HandleIntent routes the model's decision to the OS driver layer
func HandleIntent(payload IntentPayload) error {
	switch payload.Intent {
	case "CAMERA_CAPTURE", "CAMERA_CAPTURE_ANALYZE":
		return TriggerCameraCapture(payload.Roles["target"])

	case "AUDIO_VOLUME_UP", "AUDIO_VOLUME_DOWN":
		return AdjustSpeakerVolume(payload.Intent)

	case "MICROPHONE_MUTE":
		return ToggleMicrophone(false)

	case "MICROPHONE_UNMUTE":
		return ToggleMicrophone(true)

	default:
		return fmt.Errorf("intent %s recognized but no hardware routine mapped yet", payload.Intent)
	}
}

func TriggerCameraCapture(targetFile string) error {
	// When hardware arrives, this executes: fswebcam /dev/video0 output.jpg
	if targetFile == "" {
		targetFile = "output"
	}
	fmt.Printf("[MOCK] Executing: fswebcam over /dev/video0 creating %s.jpg\n", targetFile)
	return nil
}

func AdjustSpeakerVolume(intent string) error {
	// When hardware arrives, this executes: amixer set Master 5%+
	if intent == "AUDIO_VOLUME_UP" {
		fmt.Println("[MOCK] Executing: amixer set Master 10%+")
	} else {
		fmt.Println("[MOCK] Executing: amixer set Master 10%-")
	}
	return nil
}

func ToggleMicrophone(enable bool) error {
	// Toggles ALSA capture lines
	if enable {
		fmt.Println("[MOCK] Executing: amixer set Capture cap")
	} else {
		fmt.Println("[MOCK] Executing: amixer set Capture nocap")
	}
	return nil
}
