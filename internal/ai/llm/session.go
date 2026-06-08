package llm

import (
	"log"
	"strings"
	"sync"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Message & Conversation types
// ─────────────────────────────────────────────────────────────────────────────

// Message is a single turn in a conversation, carrying the speaker's role,
// the content, and a lightweight token-count estimate for context-window math.
type Message struct {
	Role      string    `json:"role"`             // "system", "user", or "assistant"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Tokens    int       `json:"tokens,omitempty"` // Rough estimate: len(fields)
}

// estimateTokens returns a fast word-count approximation of BPE token length.
// Accurate enough for sliding-window bookkeeping without running a real tokeniser.
func estimateTokens(text string) int {
	n := len(strings.Fields(text))
	if n == 0 {
		return 1
	}
	return n
}

// Conversation represents one continuous dialogue session identified by a
// unique SessionID. It is safe for concurrent access via an internal RWMutex.
type Conversation struct {
	ID        string
	Messages  []Message
	UpdatedAt time.Time
	mu        sync.RWMutex
}

// AddMessage appends a new turn to the conversation and stamps UpdatedAt.
func (c *Conversation) AddMessage(role, content string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Messages = append(c.Messages, Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Tokens:    estimateTokens(content),
	})
	c.UpdatedAt = time.Now()
}

// GetContextForInference returns the slice of Messages that fits inside
// maxTokens, always preserving the system-prompt at index 0 and preferring
// the most recent turns (sliding-window strategy).
//
//   - The system prompt is never counted against the budget after the first pass.
//   - Messages are collected newest-first then reversed so the returned slice
//     is in chronological order (system → oldest kept turn → newest turn).
func (c *Conversation) GetContextForInference(maxTokens int) []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.Messages) == 0 {
		return nil
	}

	var systemMsg *Message
	tokenCount := 0

	// Preserve system prompt regardless of budget.
	if c.Messages[0].Role == "system" {
		systemMsg = &c.Messages[0]
		tokenCount += systemMsg.Tokens
	}

	// Walk backwards through history, accumulating until we hit the limit.
	var recent []Message
	for i := len(c.Messages) - 1; i >= 0; i-- {
		msg := c.Messages[i]
		if msg.Role == "system" {
			continue // handled separately
		}
		if tokenCount+msg.Tokens > maxTokens {
			break
		}
		tokenCount += msg.Tokens
		recent = append([]Message{msg}, recent...) // prepend → keeps chrono order
	}

	var payload []Message
	if systemMsg != nil {
		payload = append(payload, *systemMsg)
	}
	payload = append(payload, recent...)
	return payload
}

// BuildContextString formats the conversation window as the token string that
// gollemer's encoder already understands (__system__ / __user__ / __assistant__).
func (c *Conversation) BuildContextString(maxTokens int) string {
	msgs := c.GetContextForInference(maxTokens)
	var sb strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "system":
			sb.WriteString("<s> __system__ ")
			sb.WriteString(m.Content)
			sb.WriteString(" </s> ")
		case "user":
			sb.WriteString("__user__ ")
			sb.WriteString(strings.ToLower(m.Content))
			sb.WriteString(" ")
		case "assistant":
			sb.WriteString("__assistant__ ")
			sb.WriteString(strings.ToLower(m.Content))
			sb.WriteString(" </s> ")
		}
	}
	return strings.TrimSpace(sb.String())
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionManager
// ─────────────────────────────────────────────────────────────────────────────

const (
	// defaultSystemPrompt is injected into every new conversation.
	defaultSystemPrompt = "You are Gollemer, a helpful Go development assistant."

	// defaultMaxTokens is the context-window budget passed to GetContextForInference
	// when no override is supplied.  Sized conservatively for the current small model.
	defaultMaxTokens = 512

	// sessionTTL is how long a session can be idle before the eviction goroutine
	// removes it from memory.
	sessionTTL = 30 * time.Minute

	// evictionInterval is how often the cleaner goroutine runs.
	evictionInterval = 5 * time.Minute
)

// SessionManager tracks all active Conversations in a concurrent-safe map and
// runs a background goroutine to evict stale sessions after sessionTTL.
type SessionManager struct {
	sessions map[string]*Conversation
	mu       sync.RWMutex
	stopCh   chan struct{}
}

// NewSessionManager creates a SessionManager and starts the TTL eviction loop.
func NewSessionManager() *SessionManager {
	sm := &SessionManager{
		sessions: make(map[string]*Conversation),
		stopCh:   make(chan struct{}),
	}
	go sm.evictionLoop()
	return sm
}

// GetOrCreate returns an existing Conversation for sessionID, or creates a new
// one pre-seeded with the system prompt.
func (sm *SessionManager) GetOrCreate(sessionID string) *Conversation {
	// Fast path: read lock.
	sm.mu.RLock()
	if conv, ok := sm.sessions[sessionID]; ok {
		sm.mu.RUnlock()
		return conv
	}
	sm.mu.RUnlock()

	// Slow path: write lock.
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Double-check after acquiring write lock.
	if conv, ok := sm.sessions[sessionID]; ok {
		return conv
	}

	conv := &Conversation{
		ID:        sessionID,
		Messages:  []Message{},
		UpdatedAt: time.Now(),
	}
	// Bootstrap with the system prompt so it is always context slot 0.
	conv.Messages = append(conv.Messages, Message{
		Role:      "system",
		Content:   defaultSystemPrompt,
		Timestamp: time.Now(),
		Tokens:    estimateTokens(defaultSystemPrompt),
	})
	sm.sessions[sessionID] = conv
	log.Printf("🆕 [Session] Created session '%s'", sessionID)
	return conv
}

// Get returns the Conversation for sessionID, or nil if it does not exist.
func (sm *SessionManager) Get(sessionID string) *Conversation {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.sessions[sessionID]
}

// Delete removes a session immediately (e.g. on explicit user logout).
func (sm *SessionManager) Delete(sessionID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	delete(sm.sessions, sessionID)
}

// ActiveCount returns the number of live sessions.
func (sm *SessionManager) ActiveCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.sessions)
}

// Stop shuts down the background eviction goroutine gracefully.
func (sm *SessionManager) Stop() {
	close(sm.stopCh)
}

// evictionLoop runs periodically and removes sessions idle longer than sessionTTL.
func (sm *SessionManager) evictionLoop() {
	ticker := time.NewTicker(evictionInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			sm.evictStale()
		case <-sm.stopCh:
			return
		}
	}
}

func (sm *SessionManager) evictStale() {
	now := time.Now()
	sm.mu.Lock()
	defer sm.mu.Unlock()
	for id, conv := range sm.sessions {
		conv.mu.RLock()
		idle := now.Sub(conv.UpdatedAt)
		conv.mu.RUnlock()
		if idle > sessionTTL {
			delete(sm.sessions, id)
			log.Printf("🗑️  [Session] Evicted idle session '%s' (idle %.0fm)", id, idle.Minutes())
		}
	}
}
