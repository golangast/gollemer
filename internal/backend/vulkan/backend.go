//go:build !cgo

package vulkan

import (
"fmt"
"sync"

vk "github.com/gogpu/wgpu/hal/vulkan/vk"
)

// Backend implements Vulkan-based GPU computation
type Backend struct {
	instance        vk.Instance
	physDevice      vk.PhysicalDevice
	device          vk.Device
	queue           vk.Queue
	cmdPool         vk.CommandPool
	shaderModules   map[string]vk.ShaderModule
	pipelines       map[string]vk.Pipeline
	pipelineLayouts map[string]vk.PipelineLayout
	mu              sync.RWMutex
	memoryProperties vk.PhysicalDeviceMemoryProperties
	allocations     map[uintptr]vk.DeviceMemory
	allocMu         sync.Mutex
}

// New creates and initializes a Vulkan backend
func New() (*Backend, error) {
	fmt.Println("🚀 Initializing Direct Vulkan Backend (Pure Go, using system libvulkan.so)...")

	// Initialize Vulkan loader with system libvulkan.so
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan: failed to initialize loader: %w", err)
	}
	fmt.Println("   ✓ Vulkan loader initialized from /lib/x86_64-linux-gnu/libvulkan.so.1")

	// Create Vulkan instance
	appInfo := vk.ApplicationInfo{
		SType:      vk.StructureTypeApplicationInfo,
		ApiVersion: vk.ApiVersion12,
	}

	createInfo := vk.InstanceCreateInfo{
		SType:            vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: &appInfo,
	}

	var instance vk.Instance
	if res := vk.CreateInstance(&createInfo, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vulkan: CreateInstance failed: %v", res)
	}
	fmt.Println("   ✓ Vulkan instance created")

	// Find physical device
	var deviceCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &deviceCount, nil); res != vk.Success {
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: EnumeratePhysicalDevices count failed: %v", res)
	}

	if deviceCount == 0 {
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: no physical devices found")
	}

	devices := make([]vk.PhysicalDevice, deviceCount)
	if res := vk.EnumeratePhysicalDevices(instance, &deviceCount, devices); res != vk.Success {
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: EnumeratePhysicalDevices failed: %v", res)
	}

	// Select discrete GPU if available
	var selectedDevice vk.PhysicalDevice
	for _, device := range devices {
		var props vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(device, &props)
		fmt.Printf("   Found GPU: %s\n", string(props.DeviceName[:]))

		if props.DeviceType == vk.PhysicalDeviceTypeDiscreteGpu {
			selectedDevice = device
			break
		}
		if selectedDevice == nil {
			selectedDevice = device
		}
	}
	fmt.Println("   ✓ Physical device selected")

	// Find compute queue family
	var queueFamilyCount uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(selectedDevice, &queueFamilyCount, nil)
	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vk.GetPhysicalDeviceQueueFamilyProperties(selectedDevice, &queueFamilyCount, queueFamilies)

	queueFamilyIndex := uint32(0)
	found := false
	for i, family := range queueFamilies {
		if (family.QueueFlags & vk.QueueComputeBit) != 0 {
			queueFamilyIndex = uint32(i)
			found = true
			break
		}
	}

	if !found {
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: no compute queue family found")
	}

	// Create logical device
	priority := float32(1.0)
	queueCreateInfo := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamilyIndex,
		QueueCount:       1,
		PQueuePriorities: &priority,
	}

	deviceCreateInfo := vk.DeviceCreateInfo{
		SType:              vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:  &queueCreateInfo,
	}

	var device vk.Device
	if res := vk.CreateDevice(selectedDevice, &deviceCreateInfo, nil, &device); res != vk.Success {
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: CreateDevice failed: %v", res)
	}

	vk.LoadProc(device)
	fmt.Println("   ✓ Logical device created")

	// Get queue
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamilyIndex, 0, &queue)

	// Create command pool
	cmdPoolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamilyIndex,
	}

	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &cmdPoolInfo, nil, &cmdPool); res != vk.Success {
		vk.DestroyDevice(device, nil)
		vk.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: CreateCommandPool failed: %v", res)
	}

	// Get memory properties
	var memProps vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(selectedDevice, &memProps)

	b := &Backend{
		instance:        instance,
		physDevice:      selectedDevice,
		device:          device,
		queue:           queue,
		cmdPool:         cmdPool,
		shaderModules:   make(map[string]vk.ShaderModule),
		pipelines:       make(map[string]vk.Pipeline),
		pipelineLayouts: make(map[string]vk.PipelineLayout),
		allocations:     make(map[uintptr]vk.DeviceMemory),
		memoryProperties: memProps,
	}

	fmt.Println("✅ Direct Vulkan Backend initialized successfully")
	fmt.Println("   Ready for GPU-accelerated tensor operations")
	return b, nil
}

// Release cleans up Vulkan resources
func (b *Backend) Release() error {
	if b == nil {
		return nil
	}

	for _, shader := range b.shaderModules {
		vk.DestroyShaderModule(b.device, shader, nil)
	}

	for _, pipeline := range b.pipelines {
		vk.DestroyPipeline(b.device, pipeline, nil)
	}

	for _, layout := range b.pipelineLayouts {
		vk.DestroyPipelineLayout(b.device, layout, nil)
	}

	for _, mem := range b.allocations {
		vk.FreeMemory(b.device, mem, nil)
	}

	vk.DestroyCommandPool(b.device, b.cmdPool, nil)
	vk.DestroyDevice(b.device, nil)
	vk.DestroyInstance(b.instance, nil)

	return nil
}

// SupportsCompute returns true (Vulkan always supports compute)
func (b *Backend) SupportsCompute() bool {
	return true
}

// Name returns the backend name
func (b *Backend) Name() string {
	var props vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(b.physDevice, &props)
	return fmt.Sprintf("Vulkan (%s)", string(props.DeviceName[:]))
}
