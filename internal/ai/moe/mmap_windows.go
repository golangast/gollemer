//go:build windows
// +build windows

package moe

import (
	"os"
	"reflect"
	"unsafe"
)

// MmapWeights maps a weight file directly into RAM on Windows.
// Note: Using a simplified approach for Windows compatibility.
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
	if size == 0 {
		return []float32{}, f, nil
	}

	// On Windows, syscall.Mmap is not available in the same way.
	// Fallback to reading the file for simplicity and stability on Windows
	// unless high-performance mmap is specifically required.
	data := make([]byte, size)
	_, err = f.Read(data)
	if err != nil {
		f.Close()
		return nil, nil, err
	}

	// Use unsafe to convert []byte to []float32 for consistency with the API
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Len /= 4
	header.Cap /= 4

	return *(*[]float32)(unsafe.Pointer(&header)), f, nil
}
