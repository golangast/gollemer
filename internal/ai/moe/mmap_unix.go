//go:build !windows
// +build !windows

package moe

import (
	"os"
	"syscall"
	"unsafe"
)

// MmapWeights maps a weight file directly into RAM to avoid copies.
func MmapWeights(filename string) ([]float32, *os.File, error) {
	f, err := os.OpenFile(filename, os.O_RDONLY, 0)
	if err != nil {
		return nil, nil, err
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}

	size := info.Size()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, nil, err
	}

	// Use unsafe.Slice to convert []byte to []float32 safely
	floatData := unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
	return floatData, f, nil
}
