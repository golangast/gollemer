//go:build !windows
// +build !windows

package moe

import (
	"os"
	"reflect"
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

	// Use unsafe to convert []byte to []float32
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Len /= 4
	header.Cap /= 4

	return *(*[]float32)(unsafe.Pointer(&header)), f, nil
}
