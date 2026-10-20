package main

import (
	"errors"
	"sync"
)

// Counter counts things by name. It is safe for many goroutines to
// use at the same time, because every access is guarded by a lock.
type Counter struct {
	mu     sync.Mutex
	counts map[string]int
}

// NewCounter returns an empty Counter, ready to use.
func NewCounter() *Counter {
	return &Counter{counts: make(map[string]int)}
}

// Add counts one more of the named thing.
func (c *Counter) Add(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.counts[name]++
}

// Get reports how many of the named thing have been counted. It
// returns an error when the name was never added.
func (c *Counter) Get(name string) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n, ok := c.counts[name]
	if !ok {
		return 0, errors.New("never counted: " + name)
	}
	return n, nil
}

// Total adds up everything counted so far, across all names.
func (c *Counter) Total() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	total := 0
	for _, n := range c.counts {
		total += n
	}
	return total
}
